from collections import deque, defaultdict
from typing import Tuple, Optional, List, Dict

from game_constants import Team, FoodType, ShopCosts, GameConstants
from robot_controller import RobotController
from item import Pan, Plate, Food
from tiles import Box

FOOD_BY_NAME = {ft.food_name: ft for ft in FoodType}
FOOD_BY_ID = {ft.food_id: ft for ft in FoodType}

DIRECTIONS = (
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
)

DEFAULT_TUNING = {
    "claim_bonus_none": 1000,
    "claim_bonus_ours": 500,
    "plate_exact_bonus": 5000,
    "plate_subset_bonus": 2000,
    "plate_mismatch_penalty": 2000,
    "infeasible_multiplier": 0.1,
    "value_penalty_weight": 0.8,
    "shop_overhead": 6,
    "travel_weight_small": 0.8,
    "travel_weight_large": 1.2,
    "future_horizon": 90,
    "future_bonus": 250,
    "small_map_simplicity_penalty": 600,
    "high_penalty_threshold": 2500,
    "high_penalty_urgent_turns": 55,
    "quick_sabotage_limit": 2,
    "quick_sabotage_cooldown": 80,
    "low_pressure_margin": 0.45,
    "low_pressure_margin_small": 0.35,
    "urgency_weight": 200.0,
    "last_order_bonus": 200,
}


class BotPlayer:
    def __init__(self, map_copy):
        self.map = map_copy
        self.map_width = map_copy.width
        self.map_height = map_copy.height

        self.tile_cache: Dict[Team, Dict[str, List[Tuple[int, int]]]] = {}
        self.workstations: Dict[Team, Dict[str, Tuple[int, int]]] = {}
        self.shop_adjacent_cache: Dict[Team, List[Tuple[int, int]]] = {}
        self.cooker_adjacent_cache: Dict[Team, List[Tuple[int, int]]] = {}
        self.map_profile: Dict[Team, Dict[str, int]] = {}
        self.bot_roles: Dict[int, str] = {}
        self.last_order_id: Optional[int] = None
        self.has_switched = False
        self._turn_id: Optional[int] = None
        self._moved_bots: set = set()
        self.disable_sabotage = False
        self.enable_switch = True
        self.tuning = dict(DEFAULT_TUNING)
        self._sabotage_mode: Optional[str] = None
        self._quick_sabotage_count: int = 0
        self._last_quick_sabotage_turn: int = -9999
        self._shop_accessible = True
        self._bfs_cache: Dict[Tuple[Tuple[int,int], Tuple[int,int]], Optional[int]] = {}
        self._map_dims_cache: Dict[Team, Tuple[int, int]] = {}
        self._walkable: Dict[Team, set] = {}
        self._team_bot_ids: Optional[List[int]] = None
        self._cached_orders_own: Optional[List] = None
        self._cached_orders_enemy: Optional[List] = None
        self._cached_orders_turn: Optional[int] = None
        # Anti-oscillation: track last actions per bot
        self._bot_last_action: Dict[int, List[str]] = {}
        self._bot_stuck_count: Dict[int, int] = {}
        # Per-turn caches (cleared each turn) for speed
        self._reachable_ws_cache: Dict[Tuple[Team, Tuple[int, int]], Dict] = {}

    def _reset_turn_state(self, turn: int) -> None:
        if self._turn_id != turn:
            self._turn_id = turn
            self._moved_bots = set()
            self._bfs_cache = {}
            self._reachable_ws_cache = {}

    def _get_map_dims(self, controller: RobotController, map_team: Team) -> Tuple[int, int]:
        cached = self._map_dims_cache.get(map_team)
        if cached is not None:
            self.map_width, self.map_height = cached
            return cached
        m = controller.get_map(map_team)
        w, h = m.width, m.height
        self._map_dims_cache[map_team] = (w, h)
        self.map_width, self.map_height = w, h
        return w, h

    # ----------------------------
    # Map helpers
    # ----------------------------
    def _chebyshev(self, a: Tuple[int, int], b: Tuple[int, int]) -> int:
        return max(abs(a[0] - b[0]), abs(a[1] - b[1]))

    def _ensure_tile_cache(self, controller: RobotController, map_team: Team) -> None:
        if map_team in self.tile_cache:
            return

        positions: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
        walkable_set: set = set()
        w, h = self._get_map_dims(controller, map_team)
        for x in range(w):
            for y in range(h):
                tile = controller.get_tile(map_team, x, y)
                if tile is None:
                    continue
                positions[tile.tile_name].append((x, y))
                if tile.is_walkable:
                    walkable_set.add((x, y))

        self.tile_cache[map_team] = positions
        self._walkable[map_team] = walkable_set

        def nearest_to(tile_name: str, targets: List[Tuple[int, int]]) -> Optional[Tuple[int, int]]:
            if not targets:
                return None
            best = None
            best_dist = 10**9
            for pos in positions.get(tile_name, []):
                for t in targets:
                    dist = self._chebyshev(pos, t)
                    if dist < best_dist:
                        best_dist = dist
                        best = pos
            return best

        submit_pos = positions.get("SUBMIT", [])
        cooker_pos = positions.get("COOKER", [])
        counter_pos = positions.get("COUNTER", [])
        shop_pos = positions.get("SHOP", [])
        sink_pos = positions.get("SINK", [])
        sinktable_pos = positions.get("SINKTABLE", [])
        trash_pos = positions.get("TRASH", [])

        prep_counter = nearest_to("COUNTER", cooker_pos) if cooker_pos else (counter_pos[0] if counter_pos else None)
        plate_counter = nearest_to("COUNTER", submit_pos) if submit_pos else (counter_pos[0] if counter_pos else None)

        self.workstations[map_team] = {
            "prep_counter": prep_counter,
            "plate_counter": plate_counter,
            "cooker": cooker_pos[0] if cooker_pos else None,
            "shop": shop_pos[0] if shop_pos else None,
            "submit": submit_pos[0] if submit_pos else None,
            "sink": sink_pos[0] if sink_pos else None,
            "sinktable": sinktable_pos[0] if sinktable_pos else None,
            "trash": trash_pos[0] if trash_pos else None,
        }

        # Precompute walkable tiles adjacent to any shop (use walkable_set to avoid get_tile calls)
        adjacent = []
        for sx, sy in shop_pos:
            for dx, dy in DIRECTIONS:
                ax, ay = sx + dx, sy + dy
                if (ax, ay) in walkable_set:
                    adjacent.append((ax, ay))
        self.shop_adjacent_cache[map_team] = adjacent

        # Walkable tiles adjacent to any cooker (COOKER is non-walkable; BFS/role assignment need reachable target).
        cooker_adjacent = []
        for cx, cy in cooker_pos:
            for dx, dy in DIRECTIONS:
                ax, ay = cx + dx, cy + dy
                if (ax, ay) in walkable_set:
                    cooker_adjacent.append((ax, ay))
        self.cooker_adjacent_cache[map_team] = cooker_adjacent

        area = w * h
        self.map_profile[map_team] = {
            "area": area,
            "counters": len(counter_pos),
            "cookers": len(cooker_pos),
            "small_map": 1 if area <= 140 else 0,
        }

    def _get_map_profile(self, controller: RobotController, map_team: Team) -> Dict[str, int]:
        self._ensure_tile_cache(controller, map_team)
        return self.map_profile.get(map_team, {"area": 0, "counters": 0, "cookers": 0, "small_map": 0})

    def _find_nearest(self, origin: Tuple[int, int], positions: List[Tuple[int, int]]) -> Optional[Tuple[int, int]]:
        if not positions:
            return None
        return min(positions, key=lambda p: self._chebyshev(origin, p))

    # ----------------------------
    # Pathfinding
    # ----------------------------
    def _has_shop_access(self, controller: RobotController, map_team: Team) -> bool:
        targets = self.shop_adjacent_cache.get(map_team, [])
        if not targets:
            return False
        bot_ids = self._team_bot_ids if self._team_bot_ids is not None else controller.get_team_bot_ids(controller.get_team())
        for bid in bot_ids:
            b = controller.get_bot_state(bid)
            if not b:
                continue
            start = (b["x"], b["y"])
            for t in targets:
                if self._bfs_distance(controller, map_team, start, t) is not None:
                    return True
        return False

    def _get_bfs_step(
        self,
        controller: RobotController,
        map_team: Team,
        start: Tuple[int, int],
        target_predicate,
        blocked: Optional[set] = None,
    ) -> Optional[Tuple[int, int]]:
        """Return one (dx, dy) step toward a cell satisfying target_predicate. Skip cells in blocked (e.g. other bots)."""
        if blocked is None:
            blocked = set()
        queue = deque([start])
        came_from = {start: None}
        w, h = self._get_map_dims(controller, map_team)
        walkable = self._walkable.get(map_team, set())
        goal = None

        while queue:
            cx, cy = queue.popleft()
            if target_predicate(cx, cy):
                goal = (cx, cy)
                break

            for dx, dy in DIRECTIONS:
                nx, ny = cx + dx, cy + dy
                if (nx, ny) in came_from:
                    continue
                if (nx, ny) in blocked:
                    continue
                if not (0 <= nx < w and 0 <= ny < h) or (nx, ny) not in walkable:
                    continue
                came_from[(nx, ny)] = (cx, cy)
                queue.append((nx, ny))

        if goal is None:
            for pos in came_from:
                if target_predicate(pos[0], pos[1]):
                    goal = pos
                    break
        if goal is None:
            return None
        # backtrack to get first step
        curr = goal
        while came_from[curr] != start and came_from[curr] is not None:
            curr = came_from[curr]
        if came_from[curr] is None:
            return None
        return (curr[0] - start[0], curr[1] - start[1])

    def _bfs_distance(
        self,
        controller: RobotController,
        map_team: Team,
        start: Tuple[int, int],
        goal: Tuple[int, int],
    ) -> Optional[int]:
        if start == goal:
            return 0
        
        # Check cache first
        cache_key = (start, goal)
        if cache_key in self._bfs_cache:
            return self._bfs_cache[cache_key]
        
        queue = deque([(start[0], start[1], 0)])
        seen = {start}
        w, h = self._get_map_dims(controller, map_team)
        walkable = self._walkable.get(map_team, set())
        while queue:
            cx, cy, dist = queue.popleft()
            for dx, dy in DIRECTIONS:
                nx, ny = cx + dx, cy + dy
                if not (0 <= nx < w and 0 <= ny < h) or (nx, ny) not in walkable:
                    continue
                if (nx, ny) in seen:
                    continue
                if (nx, ny) == goal:
                    self._bfs_cache[cache_key] = dist + 1
                    return dist + 1
                seen.add((nx, ny))
                queue.append((nx, ny, dist + 1))
        
        self._bfs_cache[cache_key] = None
        return None

    def _move_towards(self, controller: RobotController, bot_id: int, map_team: Team, target: Tuple[int, int]) -> bool:
        """Move one step toward target. Returns True if bot is now adjacent (can act on target this turn)."""
        bot_state = controller.get_bot_state(bot_id)
        if bot_state is None:
            return False
        bx, by = bot_state["x"], bot_state["y"]
        if self._chebyshev((bx, by), target) <= 1:
            return True
        if bot_id in self._moved_bots:
            return False

        # Block tiles occupied by other bots on same team (same map) so we don't try to move into them
        blocked = set()
        bot_ids = self._team_bot_ids if self._team_bot_ids is not None else controller.get_team_bot_ids(controller.get_team())
        for other_id in bot_ids:
            if other_id == bot_id:
                continue
            other = controller.get_bot_state(other_id)
            if other and other.get("map_team") == map_team.name:
                blocked.add((other["x"], other["y"]))

        def is_adjacent(x: int, y: int) -> bool:
            return self._chebyshev((x, y), target) <= 1

        step = self._get_bfs_step(controller, map_team, (bx, by), is_adjacent, blocked=blocked)
        if step is None:
            step = self._get_bfs_step(controller, map_team, (bx, by), is_adjacent, blocked=set())
        if step is None:
            # Desperation move: BFS found no path (e.g. messy map); move one step that gets closer (Chebyshev) to target.
            w, h = self._get_map_dims(controller, map_team)
            walkable = self._walkable.get(map_team, set())
            cur_dist = self._chebyshev((bx, by), target)
            for dx, dy in DIRECTIONS:
                if dx == 0 and dy == 0:
                    continue
                nx, ny = bx + dx, by + dy
                if (nx, ny) in blocked or not (0 <= nx < w and 0 <= ny < h) or (nx, ny) not in walkable:
                    continue
                if self._chebyshev((nx, ny), target) < cur_dist and controller.can_move(bot_id, dx, dy):
                    controller.move(bot_id, dx, dy)
                    self._moved_bots.add(bot_id)
                    return False
            return False
        if controller.can_move(bot_id, step[0], step[1]):
            controller.move(bot_id, step[0], step[1])
            self._moved_bots.add(bot_id)
            # After moving, we may now be adjacent — allow same-turn action
            after = controller.get_bot_state(bot_id)
            if after and self._chebyshev((after["x"], after["y"]), target) <= 1:
                return True
        else:
            # Step-aside: chosen step is blocked (e.g. other bot); move to an adjacent free cell to break deadlock (donut).
            w, h = self._get_map_dims(controller, map_team)
            walkable = self._walkable.get(map_team, set())
            for dx, dy in DIRECTIONS:
                if dx == 0 and dy == 0:
                    continue
                nx, ny = bx + dx, by + dy
                if (nx, ny) in blocked or not (0 <= nx < w and 0 <= ny < h) or (nx, ny) not in walkable:
                    continue
                if controller.can_move(bot_id, dx, dy):
                    controller.move(bot_id, dx, dy)
                    self._moved_bots.add(bot_id)
                    return False
        return False

    # ----------------------------
    # Order helpers
    # ----------------------------
    def _select_active_order(self, controller: RobotController, team: Team) -> Optional[Dict]:
        if team == controller.get_team() and self._cached_orders_turn == controller.get_turn() and self._cached_orders_own is not None:
            orders = self._cached_orders_own
        else:
            orders = controller.get_orders(team)
        if not orders:
            return None

        turn = controller.get_turn()
        bot_ids = self._team_bot_ids if (team == controller.get_team() and self._team_bot_ids is not None) else controller.get_team_bot_ids(team)
        # Filter for active orders
        active = [o for o in orders if o.get("is_active")]
        if not active:
            return None
        

        map_team = team
        plate_obj = self._plate_at_counter_or_bot(controller, map_team)
        plate_has_food = bool(self._plate_food_dicts(plate_obj))
        profile = self._get_map_profile(controller, map_team)

        if plate_has_food:
            for o in active:
                req = self._required_food_types(o)
                if self._plate_exact_match(plate_obj, req):
                    if team == controller.get_team():
                        self.last_order_id = o.get("order_id")
                    return o
            subset_orders = []
            for o in active:
                req = self._required_food_types(o)
                if self._plate_subset_of_order(plate_obj, req):
                    subset_orders.append(o)
            if subset_orders:
                active = subset_orders

        # Compute ready_counts once (used by estimate_time many times)
        _ready_counts_cached: Dict[int, int] = defaultdict(int)
        _plate_rc = self._plate_at_counter_or_bot(controller, map_team)
        for fd in self._plate_food_dicts(_plate_rc):
            fid = fd.get("food_id")
            if fid is not None:
                _ready_counts_cached[fid] += 1
        for pos in self.tile_cache.get(map_team, {}).get("COUNTER", []):
            tile = controller.get_tile(map_team, pos[0], pos[1])
            item = getattr(tile, "item", None) if tile else None
            if isinstance(item, Food):
                _ready_counts_cached[item.food_id] += 1
        ws_rc = self.workstations.get(map_team, {})
        cooker_pos_rc = ws_rc.get("cooker")
        if cooker_pos_rc:
            tile = controller.get_tile(map_team, cooker_pos_rc[0], cooker_pos_rc[1])
            pan = getattr(tile, "item", None) if tile else None
            if isinstance(pan, Pan) and isinstance(getattr(pan, "food", None), Food):
                _ready_counts_cached[pan.food.food_id] += 1
        _bot_positions_order = {}
        for bid in bot_ids:
            bstate = controller.get_bot_state(bid)
            if bstate:
                _bot_positions_order[bid] = (bstate["x"], bstate["y"])
                holding = bstate.get("holding")
                if holding and holding.get("type") == "Food":
                    fid = holding.get("food_id")
                    if fid is not None:
                        _ready_counts_cached[fid] += 1

        def estimate_time(o: Dict) -> float:
            req = self._required_food_types(o)
            counts = dict(_ready_counts_cached)
            missing: List[FoodType] = []
            for ft in req:
                if counts.get(ft.food_id, 0) > 0:
                    counts[ft.food_id] -= 1
                else:
                    missing.append(ft)

            cookables = [ft for ft in missing if ft.can_cook]
            choppable = [ft for ft in missing if ft.can_chop and not ft.can_cook]
            simple = [ft for ft in missing if not ft.can_cook and not ft.can_chop]

            cook_time = GameConstants.COOK_PROGRESS * len(cookables) + 5 * len(cookables)
            chop_time = 3 * len(choppable)
            simple_time = 2 * len(simple)

            need_shop = False
            for ft in missing:
                if not self._box_has_food(controller, map_team, ft):
                    need_shop = True
                    break
            shop_overhead = self.tuning.get("shop_overhead", 6) if need_shop else 0

            ws = self.workstations.get(map_team, {})
            prep = ws.get("prep_counter")
            shop_pos = ws.get("shop")
            submit = ws.get("submit")

            def min_bot_dist_to(target: Optional[Tuple[int, int]]) -> Optional[int]:
                if target is None:
                    return None
                best = None
                for bid in bot_ids:
                    start = _bot_positions_order.get(bid)
                    if start is None:
                        continue
                    d = self._bfs_distance(controller, map_team, start, target)
                    if d is None:
                        d = self._chebyshev(start, target)
                    if best is None or d < best:
                        best = d
                return best

            # Simplified travel estimate
            travel_time = 3  # Base travel time
            
            return cook_time + chop_time + simple_time + shop_overhead + travel_time

        # On small maps, prioritize single-ingredient orders for quick wins
        if profile.get("small_map"):
            single_req = []
            for o in active:
                req = self._required_food_types(o)
                if len(req) == 1:
                    single_req.append(o)
            if single_req:
                def single_score(o: Dict) -> float:
                    time_left = max(1, o["expires_turn"] - turn)
                    est = estimate_time(o)
                    if est > time_left:
                        return -1e9
                    reward = o["reward"]
                    penalty = o.get("penalty", 0)
                    return (reward + 0.5 * penalty) / max(1.0, est)

                def single_tiebreak(o: Dict) -> Tuple:
                    est = estimate_time(o)
                    req_len = len(self._required_food_types(o))
                    return (est, req_len)

                best_single = max(single_req, key=lambda o: (single_score(o), -single_tiebreak(o)[0], -single_tiebreak(o)[1]))
                if team == controller.get_team():
                    self.last_order_id = best_single.get("order_id")
                return best_single

        horizon = self.tuning.get("future_horizon", 90)
        future_orders = [o for o in orders if o.get("created_turn", 0) >= turn and o.get("created_turn", 0) <= turn + horizon]
        future_freq: Dict[int, int] = defaultdict(int)
        for o in future_orders:
            for name in o.get("required", []):
                ft = FOOD_BY_NAME.get(name)
                if ft is not None:
                    future_freq[ft.food_id] += 1
        max_future = max(future_freq.values()) if future_freq else 0

        def score(o: Dict) -> float:
            time_left = max(1, o["expires_turn"] - turn)
            req = self._required_food_types(o)
            est = estimate_time(o)
            feasible = est <= (time_left - 2)

            claimed_by = o.get("claimed_by")
            if claimed_by is None:
                claim_bonus = self.tuning.get("claim_bonus_none", 1000)
            elif claimed_by in bot_ids:
                claim_bonus = self.tuning.get("claim_bonus_ours", 500)
            else:
                claim_bonus = 0

            value_penalty_weight = self.tuning.get("value_penalty_weight", 0.6)
            value = o["reward"] + (value_penalty_weight * o["penalty"])
            base = value / max(1.0, est)
            if not feasible:
                base *= self.tuning.get("infeasible_multiplier", 0.1)
            if est > time_left:
                base *= 0.05

            if plate_has_food:
                if self._plate_exact_match(plate_obj, req):
                    base += self.tuning.get("plate_exact_bonus", 5000)
                elif self._plate_subset_of_order(plate_obj, req):
                    base += self.tuning.get("plate_subset_bonus", 2000)
                else:
                    base -= self.tuning.get("plate_mismatch_penalty", 2000)

            if max_future > 0:
                future_score = 0
                for ft in req:
                    future_score += future_freq.get(ft.food_id, 0)
                base += (future_score / max_future) * self.tuning.get("future_bonus", 250)

            if profile.get("small_map"):
                base -= self.tuning.get("small_map_simplicity_penalty", 180) * max(0, (len(req) - 1))

            if plate_has_food and self._plate_subset_of_order(plate_obj, req) and time_left > 30:
                base += 250
            urgency_weight = self.tuning.get("urgency_weight", 0.0)
            if urgency_weight > 0:
                base *= (1.0 + (urgency_weight / max(1.0, time_left)))
            if time_left < 80:
                base *= 1.0 / (1.0 + 0.5 * len(req))
            if o.get("order_id") == self.last_order_id:
                base += self.tuning.get("last_order_bonus", 200)
            return base + claim_bonus
        def tiebreak(o: Dict) -> Tuple:
            est = estimate_time(o)
            req_len = len(self._required_food_types(o))
            return (est, req_len)
        
        def is_feasible(o: Dict) -> bool:
            time_left = max(1, o["expires_turn"] - turn)
            req = self._required_food_types(o)
            # Orders requiring cookables with short deadlines are impossible
            cookables = [ft for ft in req if ft.can_cook]
            if cookables and time_left < GameConstants.COOK_PROGRESS + 5:  # cook turns + buffer
                return False
            est = estimate_time(o)
            # Be more conservative - need some margin
            return est <= time_left

        # Prefer feasible orders; if none, still pick best by score to avoid stalling (e.g. simple_map after turn 258).
        feasible_orders = [o for o in active if is_feasible(o)]
        if feasible_orders:
            best = max(feasible_orders, key=lambda o: (score(o), -tiebreak(o)[0], -tiebreak(o)[1]))
        else:
            best = max(active, key=lambda o: (score(o), -tiebreak(o)[0], -tiebreak(o)[1]))
        if team == controller.get_team():
            self.last_order_id = best.get("order_id")
        return best

    def _select_enemy_order_for_sabotage(self, controller: RobotController, team: Team) -> Optional[Dict]:
        if team == controller.get_enemy_team() and self._cached_orders_turn == controller.get_turn() and self._cached_orders_enemy is not None:
            orders = self._cached_orders_enemy
        else:
            orders = controller.get_orders(team)
        if not orders:
            return None
        turn = controller.get_turn()
        active = [o for o in orders if o.get("is_active")]
        if not active:
            return None

        def sabotage_score(o: Dict) -> float:
            time_left = max(1, o["expires_turn"] - turn)
            req = self._required_food_types(o)
            cook_weight = sum(1 for ft in req if ft.can_cook or ft.can_chop)
            value = o["reward"] + o["penalty"]
            penalty = o.get("penalty", 0)
            penalty_boost = 0
            if penalty >= self.tuning.get("high_penalty_threshold", 2500):
                penalty_boost = 4000 + (penalty * 0.6)
            return (value / time_left) + (cook_weight * 150) + penalty_boost

        return max(active, key=sabotage_score)

    def _enemy_high_penalty_order(self, controller: RobotController, team: Team) -> Optional[Dict]:
        if team == controller.get_enemy_team() and self._cached_orders_turn == controller.get_turn() and self._cached_orders_enemy is not None:
            orders = self._cached_orders_enemy
        else:
            orders = controller.get_orders(team)
        if not orders:
            return None
        turn = controller.get_turn()
        threshold = self.tuning.get("high_penalty_threshold", 2500)
        urgent_turns = self.tuning.get("high_penalty_urgent_turns", 55)
        high = []
        for o in orders:
            if not o.get("is_active"):
                continue
            if o.get("penalty", 0) >= threshold and (o.get("expires_turn", 0) - turn) <= urgent_turns:
                high.append(o)
        if not high:
            return None
        return max(high, key=lambda o: o.get("penalty", 0))

    def _low_pressure_on_own(self, controller: RobotController) -> bool:
        own_team = controller.get_team()
        self._ensure_tile_cache(controller, own_team)
        if self._cached_orders_turn == controller.get_turn() and self._cached_orders_own is not None:
            orders = self._cached_orders_own
        else:
            orders = controller.get_orders(own_team)
        active = [o for o in orders if o.get("is_active")]
        if not active:
            return True
        turn = controller.get_turn()
        best = self._select_active_order(controller, own_team)
        if not best:
            return True
        time_left = max(1, best.get("expires_turn", 0) - turn)
        est = 0
        req = self._required_food_types(best)
        if req:
            plate_obj = self._plate_at_counter_or_bot(controller, own_team)
            missing = self._missing_for_plate(req, plate_obj)
            cookables = [ft for ft in missing if ft.can_cook]
            choppables = [ft for ft in missing if ft.can_chop and not ft.can_cook]
            simples = [ft for ft in missing if not ft.can_cook and not ft.can_chop]
            est = (GameConstants.COOK_PROGRESS * len(cookables)) + (3 * len(choppables)) + (2 * len(simples)) + 6
        if time_left <= GameConstants.COOK_PROGRESS + 5:
            return False
        profile = self._get_map_profile(controller, own_team)
        margin = self.tuning.get("low_pressure_margin_small", 0.35) if profile.get("small_map") else self.tuning.get("low_pressure_margin", 0.45)
        penalty = best.get("penalty", 0)
        penalty_ok = penalty < self.tuning.get("high_penalty_threshold", 2500)
        return penalty_ok and (est <= (time_left * margin))

    def _required_food_types(self, order: Dict) -> List[FoodType]:
        req = []
        for name in order.get("required", []):
            ft = FOOD_BY_NAME.get(name)
            if ft is not None:
                req.append(ft)
        return req

    def _food_satisfies(self, food: Food, ft: FoodType) -> bool:
        if food.food_id != ft.food_id:
            return False
        if ft.can_chop and not food.chopped:
            return False
        if ft.can_cook and food.cooked_stage != 1:
            return False
        if not ft.can_cook and food.cooked_stage != 0:
            return False
        return True

    def _food_dict_satisfies(self, food: Dict, ft: FoodType) -> bool:
        if food.get("food_id") != ft.food_id:
            return False
        if ft.can_chop and not food.get("chopped", False):
            return False
        if ft.can_cook and food.get("cooked_stage", 0) != 1:
            return False
        if not ft.can_cook and food.get("cooked_stage", 0) != 0:
            return False
        return True

    def _plate_food_dicts(self, plate) -> List[Dict]:
        if plate is None:
            return []
        if isinstance(plate, Plate):
            foods = []
            for f in plate.food:
                if isinstance(f, Food):
                    foods.append({
                        "food_id": f.food_id,
                        "chopped": f.chopped,
                        "cooked_stage": f.cooked_stage,
                    })
            return foods
        if isinstance(plate, dict):
            return plate.get("food", [])
        return []

    def _order_signature(self, required: List[FoodType]) -> List[Tuple[int, bool, int]]:
        sig = [(ft.food_id, ft.can_chop, 1 if ft.can_cook else 0) for ft in required]
        sig.sort()
        return sig

    def _plate_signature(self, plate) -> List[Tuple[int, bool, int]]:
        sig = []
        for f in self._plate_food_dicts(plate):
            sig.append((f.get("food_id"), f.get("chopped", False), f.get("cooked_stage", 0)))
        sig.sort()
        return sig

    def _sig_counts(self, sig: List[Tuple[int, bool, int]]) -> Dict[Tuple[int, bool, int], int]:
        counts: Dict[Tuple[int, bool, int], int] = defaultdict(int)
        for t in sig:
            counts[t] += 1
        return counts

    def _plate_subset_of_order(self, plate, required: List[FoodType]) -> bool:
        plate_sig = self._plate_signature(plate)
        order_sig = self._order_signature(required)
        plate_counts = self._sig_counts(plate_sig)
        order_counts = self._sig_counts(order_sig)
        for k, v in plate_counts.items():
            if order_counts.get(k, 0) < v:
                return False
        return True

    def _plate_exact_match(self, plate, required: List[FoodType]) -> bool:
        return self._plate_signature(plate) == self._order_signature(required)

    def _missing_for_plate(self, required: List[FoodType], plate) -> List[FoodType]:
        remaining = list(required)
        foods = self._plate_food_dicts(plate)
        for f in foods:
            for i, ft in enumerate(remaining):
                if self._food_dict_satisfies(f, ft):
                    remaining.pop(i)
                    break
        return remaining

    def _plate_has_uncooked_cookable(self, plate) -> bool:
        for f in self._plate_food_dicts(plate):
            fid = f.get("food_id")
            ft = FOOD_BY_ID.get(fid)
            if ft is None:
                continue
            if ft.can_cook and f.get("cooked_stage", 0) == 0:
                return True
        return False

    # ----------------------------
    # Item scanning helpers
    # ----------------------------
    def _tile_item(self, controller: RobotController, map_team: Team, pos: Optional[Tuple[int, int]]):
        if pos is None:
            return None
        tile = controller.get_tile(map_team, pos[0], pos[1])
        if tile is None:
            return None
        return tile.item

    def _find_ready_food_on_counters(self, controller: RobotController, map_team: Team, ft: FoodType) -> Optional[Tuple[int, int]]:
        counters = self.tile_cache.get(map_team, {}).get("COUNTER", [])
        for pos in counters:
            tile = controller.get_tile(map_team, pos[0], pos[1])
            if tile is None:
                continue
            item = getattr(tile, "item", None)
            if isinstance(item, Food) and self._food_satisfies(item, ft):
                return pos
        return None

    def _find_plate_on_counters(self, controller: RobotController, map_team: Team, near: Optional[Tuple[int, int]] = None) -> Optional[Tuple[int, int]]:
        counters = self.tile_cache.get(map_team, {}).get("COUNTER", [])
        if not counters:
            return None
        best = None
        best_dist = 10**9
        for pos in counters:
            tile = controller.get_tile(map_team, pos[0], pos[1])
            item = getattr(tile, "item", None) if tile else None
            if isinstance(item, Plate) and not item.dirty:
                dist = self._chebyshev(pos, near) if near else 0
                if dist < best_dist:
                    best_dist = dist
                    best = pos
        return best

    def _find_any_food_on_counters(self, controller: RobotController, map_team: Team) -> Optional[Tuple[int, int]]:
        counters = self.tile_cache.get(map_team, {}).get("COUNTER", [])
        for pos in counters:
            tile = controller.get_tile(map_team, pos[0], pos[1])
            if tile is None:
                continue
            if isinstance(getattr(tile, "item", None), Food):
                return pos
        return None

    def _has_ready_food(self, controller: RobotController, map_team: Team, ft: FoodType) -> bool:
        if self._find_ready_food_on_counters(controller, map_team, ft) is not None:
            return True
        for pos in self.tile_cache.get(map_team, {}).get("COOKER", []):
            tile = controller.get_tile(map_team, pos[0], pos[1])
            pan = getattr(tile, "item", None) if tile else None
            if isinstance(pan, Pan) and isinstance(getattr(pan, "food", None), Food):
                if self._food_satisfies(pan.food, ft):
                    return True
        return False

    def _has_food_id_on_counters(self, controller: RobotController, map_team: Team, ft: FoodType) -> bool:
        counters = self.tile_cache.get(map_team, {}).get("COUNTER", [])
        for pos in counters:
            tile = controller.get_tile(map_team, pos[0], pos[1])
            item = getattr(tile, "item", None) if tile else None
            if isinstance(item, Food) and item.food_id == ft.food_id:
                return True
        return False

    def _find_empty_counter_near(self, controller: RobotController, map_team: Team, near: Optional[Tuple[int, int]]) -> Optional[Tuple[int, int]]:
        counters = self.tile_cache.get(map_team, {}).get("COUNTER", [])
        if not counters:
            return None
        if near is None:
            near = counters[0]
        best = None
        best_dist = 10**9
        for pos in counters:
            tile = controller.get_tile(map_team, pos[0], pos[1])
            if tile is None:
                continue
            if getattr(tile, "item", None) is not None:
                continue
            dist = self._chebyshev(pos, near)
            if dist < best_dist:
                best_dist = dist
                best = pos
        return best

    def _find_empty_counter_reachable(self, controller: RobotController, map_team: Team, origin: Tuple[int, int]) -> Optional[Tuple[int, int]]:
        counters = self.tile_cache.get(map_team, {}).get("COUNTER", [])
        if not counters:
            return None
        best = None
        best_dist = 10**9
        for pos in counters:
            tile = controller.get_tile(map_team, pos[0], pos[1])
            if tile is None:
                continue
            if getattr(tile, "item", None) is not None:
                continue
            d = None
            for dx, dy in DIRECTIONS:
                ax, ay = pos[0] + dx, pos[1] + dy
                if not (0 <= ax < self.map_width and 0 <= ay < self.map_height):
                    continue
                adj = controller.get_tile(map_team, ax, ay)
                if adj is None or not adj.is_walkable:
                    continue
                dist = self._bfs_distance(controller, map_team, origin, (ax, ay))
                if dist is None:
                    continue
                if d is None or dist < d:
                    d = dist
            if d is None:
                continue
            if d < best_dist:
                best_dist = d
                best = pos
        return best

    def _find_counter_reachable(self, controller: RobotController, map_team: Team, origin: Tuple[int, int], near: Optional[Tuple[int, int]] = None) -> Optional[Tuple[int, int]]:
        counters = self.tile_cache.get(map_team, {}).get("COUNTER", [])
        if not counters:
            return None
        ref = near or origin
        best = None
        best_score = 10**9
        for pos in counters:
            d = None
            for dx, dy in DIRECTIONS:
                ax, ay = pos[0] + dx, pos[1] + dy
                if not (0 <= ax < self.map_width and 0 <= ay < self.map_height):
                    continue
                adj = controller.get_tile(map_team, ax, ay)
                if adj is None or not adj.is_walkable:
                    continue
                dist = self._bfs_distance(controller, map_team, origin, (ax, ay))
                if dist is None:
                    continue
                if d is None or dist < d:
                    d = dist
            if d is None:
                continue
            score = d + self._chebyshev(pos, ref)
            if score < best_score:
                best_score = score
                best = pos
        return best

    def _find_nearest_reachable(self, controller: RobotController, map_team: Team, origin: Tuple[int, int], positions: List[Tuple[int, int]]) -> Optional[Tuple[int, int]]:
        """Return the position in positions that is reachable from origin with minimum BFS distance. For multi-region maps."""
        if not positions:
            return None
        best = None
        best_dist = 10**9
        for pos in positions:
            d = self._bfs_distance(controller, map_team, origin, pos)
            if d is not None and d < best_dist:
                best_dist = d
                best = pos
        return best

    def _can_reach_cooker(self, controller: RobotController, map_team: Team, origin: Tuple[int, int], cooker_pos: Tuple[int, int]) -> bool:
        """COOKER tiles are non-walkable; check if origin can reach any walkable cell adjacent to cooker (split/donut)."""
        walkable = self._walkable.get(map_team, set())
        cx, cy = cooker_pos
        for dx, dy in DIRECTIONS:
            ax, ay = cx + dx, cy + dy
            if (ax, ay) in walkable and self._bfs_distance(controller, map_team, origin, (ax, ay)) is not None:
                return True
        return False

    def _find_nearest_reachable_cooker(self, controller: RobotController, map_team: Team, origin: Tuple[int, int], cooker_list: List[Tuple[int, int]]) -> Optional[Tuple[int, int]]:
        """Return the cooker in cooker_list that is reachable from origin (via adjacent tile) with minimum BFS distance."""
        if not cooker_list:
            return None
        best_cooker = None
        best_dist = 10**9
        walkable = self._walkable.get(map_team, set())
        for cooker_pos in cooker_list:
            cx, cy = cooker_pos
            cook_dist = 10**9
            for dx, dy in DIRECTIONS:
                ax, ay = cx + dx, cy + dy
                if (ax, ay) in walkable:
                    d = self._bfs_distance(controller, map_team, origin, (ax, ay))
                    if d is not None and d < cook_dist:
                        cook_dist = d
            if cook_dist < best_dist:
                best_dist = cook_dist
                best_cooker = cooker_pos
        return best_cooker

    def _get_reachable_workstations(self, controller: RobotController, map_team: Team, origin: Tuple[int, int]) -> Dict[str, Optional[Tuple[int, int]]]:
        """Resolve workstation positions reachable from origin. For maps with multiple kitchens (e.g. split)."""
        cache_key = (map_team, origin)
        if cache_key in self._reachable_ws_cache:
            return self._reachable_ws_cache[cache_key]
        tc = self.tile_cache.get(map_team, {})
        submit_list = tc.get("SUBMIT", [])
        cooker_list = tc.get("COOKER", [])
        shop_list = tc.get("SHOP", [])
        sink_list = tc.get("SINK", [])
        sinktable_list = tc.get("SINKTABLE", [])
        trash_list = tc.get("TRASH", [])
        canonical = self.workstations.get(map_team, {})
        canonical_cooker = canonical.get("cooker")
        canonical_submit = canonical.get("submit")
        canonical_prep = canonical.get("prep_counter")
        canonical_plate = canonical.get("plate_counter")
        # Use canonical cooker/submit/prep/plate when reachable so all bots work at the same kitchen (single-kitchen / split).
        # COOKER is non-walkable; use _can_reach_cooker (reachable to adjacent tile) for cooker.
        if canonical_cooker is not None and self._can_reach_cooker(controller, map_team, origin, canonical_cooker):
            cooker_pos = canonical_cooker
        else:
            cooker_pos = self._find_nearest_reachable_cooker(controller, map_team, origin, cooker_list) if cooker_list else None
        if canonical_submit is not None and self._bfs_distance(controller, map_team, origin, canonical_submit) is not None:
            submit_pos = canonical_submit
        else:
            submit_pos = self._find_nearest_reachable(controller, map_team, origin, submit_list) if submit_list else None
        canonical_shop = canonical.get("shop")
        canonical_sink = canonical.get("sink")
        canonical_sinktable = canonical.get("sinktable")
        canonical_trash = canonical.get("trash")
        if canonical_shop is not None and self._bfs_distance(controller, map_team, origin, canonical_shop) is not None:
            shop_pos = canonical_shop
        else:
            shop_pos = self._find_nearest_reachable(controller, map_team, origin, shop_list) if shop_list else None
        if canonical_sink is not None and self._bfs_distance(controller, map_team, origin, canonical_sink) is not None:
            sink_pos = canonical_sink
        else:
            sink_pos = self._find_nearest_reachable(controller, map_team, origin, sink_list) if sink_list else None
        if canonical_sinktable is not None and self._bfs_distance(controller, map_team, origin, canonical_sinktable) is not None:
            sinktable_pos = canonical_sinktable
        else:
            sinktable_pos = self._find_nearest_reachable(controller, map_team, origin, sinktable_list) if sinktable_list else None
        if canonical_trash is not None and self._bfs_distance(controller, map_team, origin, canonical_trash) is not None:
            trash_pos = canonical_trash
        else:
            trash_pos = self._find_nearest_reachable(controller, map_team, origin, trash_list) if trash_list else None
        # Fallback: use canonical submit/trash when unreachable so bot has a target (desperation move can make progress; avoids messy stall).
        if submit_pos is None and canonical_submit is not None:
            submit_pos = canonical_submit
        if trash_pos is None and canonical.get("trash") is not None:
            trash_pos = canonical.get("trash")
        if canonical_prep is not None and self._bfs_distance(controller, map_team, origin, canonical_prep) is not None:
            prep_counter = canonical_prep
        else:
            prep_counter = self._find_counter_reachable(controller, map_team, origin, cooker_pos) if cooker_pos else None
            if prep_counter is None:
                prep_counter = self._find_empty_counter_reachable(controller, map_team, origin) or self._find_empty_counter_near(controller, map_team, cooker_pos)
        if canonical_plate is not None and self._bfs_distance(controller, map_team, origin, canonical_plate) is not None:
            plate_counter = canonical_plate
        else:
            plate_counter = self._find_counter_reachable(controller, map_team, origin, submit_pos) if submit_pos else None
            if plate_counter is None and submit_pos:
                plate_counter = self._find_counter_reachable(controller, map_team, origin, submit_pos)
        result = {
            "submit": submit_pos,
            "cooker": cooker_pos,
            "shop": shop_pos,
            "sink": sink_pos,
            "sinktable": sinktable_pos,
            "trash": trash_pos,
            "prep_counter": prep_counter,
            "plate_counter": plate_counter,
        }
        self._reachable_ws_cache[cache_key] = result
        return result

    def _is_single_counter_map(self, map_team: Team) -> bool:
        return len(self.tile_cache.get(map_team, {}).get("COUNTER", [])) <= 1

    def _find_empty_cooker(self, controller: RobotController, map_team: Team, origin: Tuple[int, int]) -> Optional[Tuple[int, int]]:
        cookers = self.tile_cache.get(map_team, {}).get("COOKER", [])
        best = None
        best_dist = 10**9
        for pos in cookers:
            tile = controller.get_tile(map_team, pos[0], pos[1])
            pan = getattr(tile, "item", None) if tile else None
            if isinstance(pan, Pan) and pan.food is None:
                dist = self._chebyshev(origin, pos)
                if dist < best_dist:
                    best_dist = dist
                    best = pos
        return best

    def _find_box_with_food(self, controller: RobotController, map_team: Team, ft: FoodType, near: Optional[Tuple[int, int]] = None) -> Optional[Tuple[int, int]]:
        """Find a box containing food that matches the required FoodType (same id, chopped/cooked state)."""
        boxes = self.tile_cache.get(map_team, {}).get("BOX", [])
        if not boxes:
            return None
        ref = near or (boxes[0] if boxes else None)
        best = None
        best_dist = 10**9
        for pos in boxes:
            tile = controller.get_tile(map_team, pos[0], pos[1])
            if tile is None or not isinstance(tile, Box):
                continue
            if getattr(tile, "count", 0) <= 0:
                continue
            item = getattr(tile, "item", None)
            if not isinstance(item, Food):
                continue
            if not self._food_satisfies(item, ft):
                continue
            dist = self._chebyshev(pos, ref) if ref else 0
            if dist < best_dist:
                best_dist = dist
                best = pos
        return best

    def _find_box_with_food_type(self, controller: RobotController, map_team: Team, ft: FoodType, near: Optional[Tuple[int, int]] = None) -> Optional[Tuple[int, int]]:
        """Find a box containing this food type (same food_id), any chopped/cooked state (for cook to process)."""
        boxes = self.tile_cache.get(map_team, {}).get("BOX", [])
        if not boxes:
            return None
        ref = near or (boxes[0] if boxes else None)
        best = None
        best_dist = 10**9
        for pos in boxes:
            tile = controller.get_tile(map_team, pos[0], pos[1])
            if tile is None or not isinstance(tile, Box):
                continue
            if getattr(tile, "count", 0) <= 0:
                continue
            item = getattr(tile, "item", None)
            if not isinstance(item, Food) or item.food_id != ft.food_id:
                continue
            dist = self._chebyshev(pos, ref) if ref else 0
            if dist < best_dist:
                best_dist = dist
                best = pos
        return best

    def _find_box_for_store(self, controller: RobotController, map_team: Team, ft: FoodType) -> Optional[Tuple[int, int]]:
        boxes = self.tile_cache.get(map_team, {}).get("BOX", [])
        for pos in boxes:
            tile = controller.get_tile(map_team, pos[0], pos[1])
            if tile is None or not isinstance(tile, Box):
                continue
            if getattr(tile, "count", 0) == 0:
                return pos
            item = getattr(tile, "item", None)
            if isinstance(item, Food) and item.food_id == ft.food_id:
                return pos
        return None

    def _box_has_food(self, controller: RobotController, map_team: Team, ft: FoodType) -> bool:
        boxes = self.tile_cache.get(map_team, {}).get("BOX", [])
        for pos in boxes:
            tile = controller.get_tile(map_team, pos[0], pos[1])
            if tile is None or not isinstance(tile, Box):
                continue
            item = getattr(tile, "item", None)
            if getattr(tile, "count", 0) > 0 and isinstance(item, Food) and item.food_id == ft.food_id:
                return True
        return False

    def _shop_for_items(
        self,
        controller: RobotController,
        bot_id: int,
        map_team: Team,
        items: List,
    ) -> bool:
        """Move to shop and buy first affordable item from the list. Returns True if we moved or bought."""
        if not items:
            return False
        bot_state = controller.get_bot_state(bot_id)
        if bot_state is None or bot_state.get("holding"):
            return False
        ws = self.workstations.get(map_team, {})
        shop_pos = ws.get("shop")
        if not shop_pos:
            return False
        if self._move_towards(controller, bot_id, map_team, shop_pos):
            for item in items:
                if controller.get_team_money(controller.get_team()) >= item.buy_cost:
                    controller.buy(bot_id, item, shop_pos[0], shop_pos[1])
                    return True
            return True
        return True

    # ----------------------------
    # Role assignment
    # ----------------------------
    def _assign_roles(self, controller: RobotController, bot_ids: List[int], map_team: Team) -> None:
        if self.bot_roles and all(bid in self.bot_roles for bid in bot_ids):
            return

        self.bot_roles = {}
        if len(bot_ids) == 1:
            self.bot_roles[bot_ids[0]] = "solo"
            return

        ws = self.workstations.get(map_team, {})
        cooker_pos = ws.get("cooker")
        submit_pos = ws.get("submit")
        cooker_list = self.tile_cache.get(map_team, {}).get("COOKER", [])
        submit_list = self.tile_cache.get(map_team, {}).get("SUBMIT", [])
        shop_adj = self.shop_adjacent_cache.get(map_team, [])

        _bot_positions = {}
        for bid in bot_ids:
            b = controller.get_bot_state(bid)
            _bot_positions[bid] = (b["x"], b["y"]) if b else (0, 0)

        def bot_pos(bid: int) -> Tuple[int, int]:
            return _bot_positions.get(bid, (0, 0))

        def dist_to(target: Optional[Tuple[int, int]], bid: int) -> int:
            if target is None:
                return 10**9
            d = self._bfs_distance(controller, map_team, bot_pos(bid), target)
            if d is None:
                d = self._chebyshev(bot_pos(bid), target)
            return d

        def dist_to_cooker(bid: int) -> int:
            # COOKER tiles are non-walkable; use walkable tiles adjacent to cooker for BFS (donut).
            # When we have a canonical cooker, use only its adjacent tiles so both bots are assigned to the same kitchen (chess).
            walkable = self._walkable.get(map_team, set())
            if cooker_pos is not None:
                cx, cy = cooker_pos
                best = 10**9
                for dx, dy in DIRECTIONS:
                    ax, ay = cx + dx, cy + dy
                    if (ax, ay) in walkable:
                        d = self._bfs_distance(controller, map_team, bot_pos(bid), (ax, ay))
                        if d is not None and d < best:
                            best = d
                return best if best != 10**9 else self._chebyshev(bot_pos(bid), cooker_pos)
            cooker_adj = self.cooker_adjacent_cache.get(map_team, [])
            if cooker_adj:
                best = 10**9
                for t in cooker_adj:
                    d = self._bfs_distance(controller, map_team, bot_pos(bid), t)
                    if d is not None and d < best:
                        best = d
                return best if best != 10**9 else 10**9
            return dist_to(cooker_pos, bid)

        def dist_to_nearest_reachable(bid: int, positions: List[Tuple[int, int]]) -> int:
            if not positions:
                return 10**9
            origin = bot_pos(bid)
            best = 10**9
            for pos in positions:
                d = self._bfs_distance(controller, map_team, origin, pos)
                if d is not None and d < best:
                    best = d
            return best if best != 10**9 else self._chebyshev(bot_pos(bid), positions[0])

        def dist_to_shop(bid: int) -> int:
            if not shop_adj:
                return 10**9
            best = 10**9
            for t in shop_adj:
                d = self._bfs_distance(controller, map_team, bot_pos(bid), t)
                if d is None:
                    continue
                if d < best:
                    best = d
            return best

        # Use canonical cooker/submit for role assignment when set (single-kitchen / multi-station maps) so we match original behavior.
        # Use dist_to_cooker (cooker_adjacent) so donut-like maps pathfind to walkable tiles.
        if cooker_pos is not None and submit_pos is not None:
            cook_bot = min(bot_ids, key=lambda bid: dist_to_cooker(bid) + dist_to_shop(bid))
            remaining = [bid for bid in bot_ids if bid != cook_bot]
            plate_bot = min(remaining, key=lambda bid: dist_to(submit_pos, bid)) if remaining else cook_bot
        else:
            cook_bot = min(bot_ids, key=lambda bid: dist_to_nearest_reachable(bid, cooker_list) + dist_to_shop(bid))
            remaining = [bid for bid in bot_ids if bid != cook_bot]
            plate_bot = min(remaining, key=lambda bid: dist_to_nearest_reachable(bid, submit_list)) if remaining else cook_bot
        self.bot_roles[cook_bot] = "cook"
        if plate_bot not in self.bot_roles:
            self.bot_roles[plate_bot] = "plate"

        for bid in bot_ids:
            if bid not in self.bot_roles:
                self.bot_roles[bid] = "support"

    # ----------------------------
    # Sabotage logic (enemy map)
    # ----------------------------
    def _play_sabotage(self, controller: RobotController, bot_id: int, map_team: Team) -> None:
        self._ensure_tile_cache(controller, map_team)
        bot_state = controller.get_bot_state(bot_id)
        if bot_state is None:
            return
        turn = controller.get_turn()
        holding = bot_state.get("holding")
        bx, by = bot_state["x"], bot_state["y"]
        ws = self._get_reachable_workstations(controller, map_team, (bx, by))
        trash_pos = ws.get("trash")
        high_penalty_order = self._enemy_high_penalty_order(controller, map_team)
        if high_penalty_order is not None:
            self._sabotage_mode = "high_penalty"
        enemy_order = high_penalty_order or self._select_enemy_order_for_sabotage(controller, map_team)
        enemy_required = self._required_food_types(enemy_order) if enemy_order else []
        enemy_food_ids = {ft.food_id for ft in enemy_required}
        if self._sabotage_mode == "quick":
            cooldown = self.tuning.get("quick_sabotage_cooldown", 80)
            if turn - self._last_quick_sabotage_turn > cooldown:
                self._quick_sabotage_count = 0
            limit = self.tuning.get("quick_sabotage_limit", 2)
            if self._quick_sabotage_count >= limit and holding is None:
                cooker_pos = ws.get("cooker")
                if cooker_pos:
                    self._move_towards(controller, bot_id, map_team, cooker_pos)
                return

        # If holding something, handle it first
        if holding:
            if self._sabotage_mode == "quick" and holding.get("type") == "Food" and trash_pos:
                if self._move_towards(controller, bot_id, map_team, trash_pos):
                    controller.trash(bot_id, trash_pos[0], trash_pos[1])
                    self._quick_sabotage_count += 1
                    self._last_quick_sabotage_turn = turn
                return
            if holding.get("type") == "Plate" and holding.get("dirty"):
                sink_pos = ws.get("sink")
                if sink_pos and self._move_towards(controller, bot_id, map_team, sink_pos):
                    controller.put_dirty_plate_in_sink(bot_id, sink_pos[0], sink_pos[1])
                return
            if holding.get("type") == "Pan":
                pan_food = holding.get("food")
                if pan_food is not None and trash_pos and self._move_towards(controller, bot_id, map_team, trash_pos):
                    controller.trash(bot_id, trash_pos[0], trash_pos[1])
                    return
                # Hide pan on a far counter to deny easy access
                counters = self.tile_cache.get(map_team, {}).get("COUNTER", [])
                cookers = self.tile_cache.get(map_team, {}).get("COOKER", [])
                best_counter = None
                best_score = -1
                for pos in counters:
                    tile = controller.get_tile(map_team, pos[0], pos[1])
                    if tile is None or getattr(tile, "item", None) is not None:
                        continue
                    score = 0
                    for cpos in cookers:
                        score += self._chebyshev(pos, cpos)
                    if score > best_score:
                        best_score = score
                        best_counter = pos
                if best_counter and self._move_towards(controller, bot_id, map_team, best_counter):
                    controller.place(bot_id, best_counter[0], best_counter[1])
                    return
                cooker_pos = ws.get("cooker")
                if cooker_pos:
                    self._move_towards(controller, bot_id, map_team, cooker_pos)
                return
            if trash_pos and self._move_towards(controller, bot_id, map_team, trash_pos):
                controller.trash(bot_id, trash_pos[0], trash_pos[1])
            return

        # Quick sabotage: steal 1-2 items to offset enemy timing
        if self._sabotage_mode == "quick" and holding is None:
            for ft in enemy_required:
                food_pos = self._find_ready_food_on_counters(controller, map_team, ft)
                if food_pos:
                    if self._move_towards(controller, bot_id, map_team, food_pos):
                        controller.pickup(bot_id, food_pos[0], food_pos[1])
                        self._last_quick_sabotage_turn = turn
                    return
            any_food_pos = self._find_any_food_on_counters(controller, map_team)
            if any_food_pos:
                if self._move_towards(controller, bot_id, map_team, any_food_pos):
                    controller.pickup(bot_id, any_food_pos[0], any_food_pos[1])
                    self._last_quick_sabotage_turn = turn
                return
            if enemy_food_ids:
                boxes = self.tile_cache.get(map_team, {}).get("BOX", [])
                for pos in boxes:
                    tile = controller.get_tile(map_team, pos[0], pos[1])
                    if tile is None or not isinstance(tile, Box):
                        continue
                    item = getattr(tile, "item", None)
                    if getattr(tile, "count", 0) > 0 and isinstance(item, Food) and item.food_id in enemy_food_ids:
                        if self._move_towards(controller, bot_id, map_team, pos):
                            controller.pickup(bot_id, pos[0], pos[1])
                            self._last_quick_sabotage_turn = turn
                        return

        # Take completed plates first (most immediate threat)
        counters = self.tile_cache.get(map_team, {}).get("COUNTER", [])
        best_plate_pos = None
        for pos in counters:
            tile = controller.get_tile(map_team, pos[0], pos[1])
            if tile is None:
                continue
            item = getattr(tile, "item", None)
            if isinstance(item, Plate) and not item.dirty and item.food:
                if enemy_required and self._plate_exact_match(item, enemy_required):
                    best_plate_pos = pos
                    break
                if best_plate_pos is None:
                    best_plate_pos = pos
        if best_plate_pos:
            if self._move_towards(controller, bot_id, map_team, best_plate_pos):
                controller.pickup(bot_id, best_plate_pos[0], best_plate_pos[1])
                if self._sabotage_mode == "quick":
                    self._last_quick_sabotage_turn = turn
            return

        # Steal a pan from cooker (with cooked food preferred) to deny cooking
        cookers = self.tile_cache.get(map_team, {}).get("COOKER", [])
        best_cooker = None
        best_score = -1
        for pos in cookers:
            tile = controller.get_tile(map_team, pos[0], pos[1])
            pan = getattr(tile, "item", None) if tile else None
            if not isinstance(pan, Pan):
                continue
            cooked = 1 if (pan.food is not None and getattr(pan.food, "cooked_stage", 0) >= 1) else 0
            score = cooked * 1000 - self._chebyshev((bot_state["x"], bot_state["y"]), pos)
            if score > best_score:
                best_score = score
                best_cooker = pos
        if best_cooker:
            if self._move_towards(controller, bot_id, map_team, best_cooker):
                controller.pickup(bot_id, best_cooker[0], best_cooker[1])
            return

        # Take high-priority required food from counters and trash it
        if enemy_required:
            for ft in enemy_required:
                food_pos = self._find_ready_food_on_counters(controller, map_team, ft)
                if food_pos:
                    if self._move_towards(controller, bot_id, map_team, food_pos):
                        controller.pickup(bot_id, food_pos[0], food_pos[1])
                        if self._sabotage_mode == "quick":
                            self._last_quick_sabotage_turn = turn
                    return

        # Take any food from counters and trash it
        any_food_pos = self._find_any_food_on_counters(controller, map_team)
        if any_food_pos:
            if self._move_towards(controller, bot_id, map_team, any_food_pos):
                controller.pickup(bot_id, any_food_pos[0], any_food_pos[1])
                if self._sabotage_mode == "quick":
                    self._last_quick_sabotage_turn = turn
            return

        # Take clean plates from sink table and trash them (deprive enemy of plates)
        sinktable_pos = ws.get("sinktable")
        if sinktable_pos:
            tile = controller.get_tile(map_team, sinktable_pos[0], sinktable_pos[1])
            if tile is not None and getattr(tile, "num_clean_plates", 0) > 0:
                if self._move_towards(controller, bot_id, map_team, sinktable_pos):
                    controller.take_clean_plate(bot_id, sinktable_pos[0], sinktable_pos[1])
                return

        # Take raw ingredients from boxes to slow enemy production (focus on required items)
        boxes = self.tile_cache.get(map_team, {}).get("BOX", [])
        if enemy_food_ids:
            for pos in boxes:
                tile = controller.get_tile(map_team, pos[0], pos[1])
                if tile is None or not isinstance(tile, Box):
                    continue
                item = getattr(tile, "item", None)
                if getattr(tile, "count", 0) > 0 and isinstance(item, Food) and item.food_id in enemy_food_ids:
                    if self._move_towards(controller, bot_id, map_team, pos):
                        controller.pickup(bot_id, pos[0], pos[1])
                        if self._sabotage_mode == "quick":
                            self._last_quick_sabotage_turn = turn
                    return
        for pos in boxes:
            tile = controller.get_tile(map_team, pos[0], pos[1])
            if tile is None or not isinstance(tile, Box):
                continue
            if getattr(tile, "count", 0) > 0:
                if self._move_towards(controller, bot_id, map_team, pos):
                    controller.pickup(bot_id, pos[0], pos[1])
                return

        # Otherwise move toward cooker to interfere with cooking flow
        cooker_pos = ws.get("cooker")
        if cooker_pos:
            self._move_towards(controller, bot_id, map_team, cooker_pos)

    # ----------------------------
    # Main turn
    # ----------------------------
    def play_turn(self, controller: RobotController):
        self._reset_turn_state(controller.get_turn())
        own_team = controller.get_team()
        bot_ids = controller.get_team_bot_ids(own_team)
        self._team_bot_ids = bot_ids
        turn = controller.get_turn()
        self._cached_orders_turn = turn
        self._cached_orders_own = controller.get_orders(own_team)
        self._cached_orders_enemy = controller.get_orders(controller.get_enemy_team())
        if not bot_ids:
            return

        first_state = controller.get_bot_state(bot_ids[0])
        if first_state is None:
            return
        map_team = Team[first_state["map_team"]]
        self._ensure_tile_cache(controller, map_team)
        self._shop_accessible = self._has_shop_access(controller, map_team)
        self._assign_roles(controller, bot_ids, map_team)
        if map_team == own_team:
            self._sabotage_mode = None
        if self.enable_switch and controller.can_switch_maps():
            if self._should_switch_for_sabotage(controller):
                controller.switch_maps()
                first_state = controller.get_bot_state(bot_ids[0])
                if first_state is None:
                    return
                map_team = Team[first_state["map_team"]]
                self._ensure_tile_cache(controller, map_team)

        order = self._select_active_order(controller, own_team)
        if not self.disable_sabotage and map_team != controller.get_team():
            for bid in bot_ids:
                self._play_sabotage(controller, bid, map_team)
            return

        if order is None:
            return

        required = self._required_food_types(order)
        orders_own = self._cached_orders_own if (self._cached_orders_turn == turn) else controller.get_orders(own_team)
        active_orders = [o for o in (orders_own or []) if o.get("is_active")]

        for bid in bot_ids:
            bstate = controller.get_bot_state(bid)
            holding = bstate.get("holding") if bstate else None
            if holding and holding.get("type") == "Plate":
                if holding.get("food"):
                    # If plate exactly matches any active order, submit it (avoids stalling with completed-but-unsubmitted plate).
                    plate_matches_any_order = False
                    for o in active_orders:
                        req_o = self._required_food_types(o)
                        if self._plate_exact_match(holding, req_o):
                            plate_matches_any_order = True
                            origin = (bstate["x"], bstate["y"])
                            ws = self._get_reachable_workstations(controller, map_team, origin)
                            submit_pos = ws.get("submit")
                            if submit_pos and self._move_towards(controller, bid, map_team, submit_pos):
                                controller.submit(bid, submit_pos[0], submit_pos[1])
                                continue
                            break
                    if not plate_matches_any_order and (not self._plate_subset_of_order(holding, required) or self._plate_has_uncooked_cookable(holding)):
                        origin = (bstate["x"], bstate["y"])
                        ws = self._get_reachable_workstations(controller, map_team, origin)
                        trash_pos = ws.get("trash")
                        plate_counter = ws.get("plate_counter")
                        if trash_pos and self._move_towards(controller, bid, map_team, trash_pos):
                            controller.trash(bid, trash_pos[0], trash_pos[1])
                        elif plate_counter and self._move_towards(controller, bid, map_team, plate_counter):
                            controller.place(bid, plate_counter[0], plate_counter[1])
                        continue
            role = self.bot_roles.get(bid, "support")
            if role == "solo":
                self._play_solo(controller, bid, map_team, required)
            elif role == "cook":
                self._play_cook(controller, bid, map_team, required)
            elif role == "plate":
                self._play_plate(controller, bid, map_team, required)
            else:
                self._play_support(controller, bid, map_team, required)

    # ----------------------------
    # Role behaviors
    # ----------------------------
    def _play_cook(self, controller: RobotController, bot_id: int, map_team: Team, required: List[FoodType]) -> None:
        bot_state = controller.get_bot_state(bot_id)
        if bot_state is None:
            return
        holding = bot_state.get("holding")
        bx, by = bot_state["x"], bot_state["y"]

        ws = self._get_reachable_workstations(controller, map_team, (bx, by))
        profile = self._get_map_profile(controller, map_team)
        prep_counter = ws.get("prep_counter")
        plate_counter = ws.get("plate_counter")
        cookers = self.tile_cache.get(map_team, {}).get("COOKER", [])
        # Use canonical cooker from ws first (chess: same kitchen for both bots); cookers are non-walkable so _find_nearest_reachable(cookers) would return None.
        cooker_pos = ws.get("cooker") or (self._find_nearest_reachable_cooker(controller, map_team, (bx, by), cookers) if cookers else None)
        shop_pos = ws.get("shop")
        trash_pos = ws.get("trash")
        sink_pos = ws.get("sink")
        missing = self._missing_for_plate(required, self._plate_at_counter_or_bot(controller, map_team, (bx, by)))
        single_counter = self._is_single_counter_map(map_team)
        pan_food = None
        ws_cooker = ws.get("cooker")
        if cookers:
            nearest_with_food = None
            nearest_dist = 10**9
            for pos in cookers:
                tile = controller.get_tile(map_team, pos[0], pos[1])
                pan = getattr(tile, "item", None) if tile else None
                if isinstance(pan, Pan) and isinstance(getattr(pan, "food", None), Food):
                    dist = self._chebyshev((bx, by), pos)
                    if dist < nearest_dist:
                        nearest_dist = dist
                        nearest_with_food = pos
            if nearest_with_food:
                tile = controller.get_tile(map_team, nearest_with_food[0], nearest_with_food[1])
                pan = getattr(tile, "item", None) if tile else None
                if isinstance(pan, Pan) and isinstance(getattr(pan, "food", None), Food):
                    pan_food = pan.food
                # Only switch to this cooker if same kitchen (chess) or single cooker
                if nearest_with_food == ws_cooker or ws_cooker is None or len(cookers) == 1:
                    cooker_pos = nearest_with_food

        # If holding a dirty plate (e.g. picked up by mistake), put it in the sink
        if holding and holding.get("type") == "Plate" and holding.get("dirty"):
            if sink_pos and self._move_towards(controller, bot_id, map_team, sink_pos):
                controller.put_dirty_plate_in_sink(bot_id, sink_pos[0], sink_pos[1])
            return

        if cooker_pos:
            tile = controller.get_tile(map_team, cooker_pos[0], cooker_pos[1])
            if tile is not None and not isinstance(getattr(tile, "item", None), Pan):
                if shop_pos and self._move_towards(controller, bot_id, map_team, shop_pos):
                    if controller.get_team_money(controller.get_team()) >= ShopCosts.PAN.buy_cost:
                        controller.buy(bot_id, ShopCosts.PAN, shop_pos[0], shop_pos[1])
                return

        # if holding pan, place on cooker
        if holding and holding.get("type") == "Pan":
            if cooker_pos and self._move_towards(controller, bot_id, map_team, cooker_pos):
                controller.place(bot_id, cooker_pos[0], cooker_pos[1])
            return

        # if holding plate by mistake, drop it near plate counter
        if holding and holding.get("type") == "Plate":
            if plate_counter and self._move_towards(controller, bot_id, map_team, plate_counter):
                controller.place(bot_id, plate_counter[0], plate_counter[1])
            return

        # if pan has cooked food ready and we're holding something, free hands to avoid burning
        if pan_food is not None and holding and holding.get("type") == "Food":
            if pan_food.cooked_stage == 1:
                if prep_counter:
                    tile = controller.get_tile(map_team, prep_counter[0], prep_counter[1])
                    if tile is not None and getattr(tile, "item", None) is None:
                        if self._move_towards(controller, bot_id, map_team, prep_counter):
                            controller.place(bot_id, prep_counter[0], prep_counter[1])
                        return
                ft = FOOD_BY_NAME.get(holding.get("food_name"))
                if ft is not None:
                    box_pos = self._find_box_for_store(controller, map_team, ft)
                    if box_pos and self._move_towards(controller, bot_id, map_team, box_pos):
                        controller.place(bot_id, box_pos[0], box_pos[1])
                return

        # batch shopping for cookable/choppable items to minimize trips
        if holding is None:
            allow_shop = True
            if single_counter and prep_counter:
                tile = controller.get_tile(map_team, prep_counter[0], prep_counter[1])
                if tile is not None and getattr(tile, "item", None) is not None:
                    allow_shop = False

            if allow_shop:
                shop_cook = []
                for ft in missing:
                    if ft.can_cook or ft.can_chop:
                        if pan_food is not None and ft.can_cook:
                            continue
                        if self._has_food_id_on_counters(controller, map_team, ft):
                            continue
                        if cooker_pos:
                            tile = controller.get_tile(map_team, cooker_pos[0], cooker_pos[1])
                            pan = getattr(tile, "item", None) if tile else None
                            if isinstance(pan, Pan) and isinstance(getattr(pan, "food", None), Food) and pan.food.food_id == ft.food_id:
                                continue
                        if not self._box_has_food(controller, map_team, ft):
                            shop_cook.append(ft)
                if shop_cook and self._shop_for_items(controller, bot_id, map_team, shop_cook):
                    return

        # handle holding food
        if holding and holding.get("type") == "Food":
            ft = FOOD_BY_NAME.get(holding.get("food_name"))
            if ft is None:
                return
            # chop if needed — place on empty prep counter or find another empty counter
            if ft.can_chop and not holding.get("chopped", False):
                if prep_counter:
                    tile = controller.get_tile(map_team, prep_counter[0], prep_counter[1])
                    if tile is not None:
                        item = getattr(tile, "item", None)
                        if item is None:
                            # Counter is empty, place meat
                            if self._move_towards(controller, bot_id, map_team, prep_counter):
                                controller.place(bot_id, prep_counter[0], prep_counter[1])
                                return
                        # Counter not empty - find ANY empty counter on map
                        all_counters = self.tile_cache.get(map_team, {}).get("COUNTER", [])
                        for cpos in all_counters:
                            ctile = controller.get_tile(map_team, cpos[0], cpos[1])
                            if ctile and getattr(ctile, "item", None) is None:
                                if self._move_towards(controller, bot_id, map_team, cpos):
                                    controller.place(bot_id, cpos[0], cpos[1])
                                return
                        # No empty counter - wait near prep counter (don't put back in box!)
                        self._move_towards(controller, bot_id, map_team, prep_counter)
                        return
                return
            # cook if needed (only when pan is empty AND food is already chopped if it needs chopping)
            if ft.can_cook and holding.get("cooked_stage", 0) == 0:
                # Don't try to cook if it still needs chopping first
                if ft.can_chop and not holding.get("chopped", False):
                    # Should have been handled above, but just in case, skip cooking
                    return
                empty_cooker = self._find_empty_cooker(controller, map_team, (bx, by))
                if empty_cooker and self._move_towards(controller, bot_id, map_team, empty_cooker):
                    controller.place(bot_id, empty_cooker[0], empty_cooker[1])
                return
            # ready food: place on an empty counter (so plate bot can pick up; don't overwrite plate)
            empty_for_food = self._find_empty_counter_near(controller, map_team, plate_counter) or plate_counter
            if empty_for_food:
                tile = controller.get_tile(map_team, empty_for_food[0], empty_for_food[1])
                if tile is not None and getattr(tile, "item", None) is None:
                    if self._move_towards(controller, bot_id, map_team, empty_for_food):
                        controller.place(bot_id, empty_for_food[0], empty_for_food[1])
            return

        # no holding
        if cookers:
            best_ready = None
            best_ready_dist = 10**9
            for pos in cookers:
                tile = controller.get_tile(map_team, pos[0], pos[1])
                pan = getattr(tile, "item", None) if tile else None
                if isinstance(pan, Pan) and isinstance(pan.food, Food) and pan.food.cooked_stage >= 1:
                    dist = self._chebyshev((bx, by), pos)
                    if dist < best_ready_dist:
                        best_ready_dist = dist
                        best_ready = pos
            if best_ready:
                if self._move_towards(controller, bot_id, map_team, best_ready):
                    controller.take_from_pan(bot_id, best_ready[0], best_ready[1])
                return

        # if we have burnt food in hand now, trash it
        holding = (controller.get_bot_state(bot_id) or {}).get("holding")
        if holding and holding.get("type") == "Food":
            if holding.get("cooked_stage", 0) == 2 and trash_pos:
                if self._move_towards(controller, bot_id, map_team, trash_pos):
                    controller.trash(bot_id, trash_pos[0], trash_pos[1])
                return

        # chop food on prep counter if present (or any counter)
        counters = self.tile_cache.get(map_team, {}).get("COUNTER", [])
        for counter_pos in counters:
            tile = controller.get_tile(map_team, counter_pos[0], counter_pos[1])
            if tile is not None and isinstance(getattr(tile, "item", None), Food):
                item = tile.item
                if item.can_chop and not item.chopped:
                    if holding is None and self._move_towards(controller, bot_id, map_team, counter_pos):
                        controller.chop(bot_id, counter_pos[0], counter_pos[1])
                    return
                if item.can_cook and item.cooked_stage == 0 and (not item.can_chop or item.chopped):
                    if holding is None and self._move_towards(controller, bot_id, map_team, counter_pos):
                        controller.pickup(bot_id, counter_pos[0], counter_pos[1])
                    return
                # DON'T pick up ready food here - that's the plate bot's job
                # Only skip if there's ready food (return to prevent further actions)
                if (not item.can_chop or item.chopped) and ((item.can_cook and item.cooked_stage == 1) or (not item.can_cook and item.cooked_stage == 0)):
                    # Leave ready food for plate bot - don't pick it up
                    pass

        # decide next cookable needed
        missing_cookables = [ft for ft in missing if ft.can_cook or ft.can_chop]
        if pan_food is not None:
            missing_cookables = [ft for ft in missing_cookables if not ft.can_cook]

        # avoid duplicating if ready food already on counters
        for ft in list(missing_cookables):
            pos = self._find_ready_food_on_counters(controller, map_team, ft)
            if pos is not None:
                missing_cookables.remove(ft)

        # avoid duplicating if food already in pan
        if cooker_pos:
            tile = controller.get_tile(map_team, cooker_pos[0], cooker_pos[1])
            if tile is not None and isinstance(getattr(tile, "item", None), Pan):
                pan = tile.item
                if isinstance(pan.food, Food):
                    for ft in list(missing_cookables):
                        if pan.food.food_id == ft.food_id:
                            missing_cookables.remove(ft)
                            break

        if missing_cookables:
            ft = missing_cookables[0]
            # Prefer ready food from box, then raw same type from box, then buy
            box_pos = self._find_box_with_food(controller, map_team, ft, ws.get("cooker"))
            if box_pos is None:
                box_pos = self._find_box_with_food_type(controller, map_team, ft, ws.get("cooker"))
            if box_pos and self._move_towards(controller, bot_id, map_team, box_pos):
                controller.pickup(bot_id, box_pos[0], box_pos[1])
                return
            if shop_pos and self._move_towards(controller, bot_id, map_team, shop_pos):
                if controller.get_team_money(controller.get_team()) >= ft.buy_cost:
                    controller.buy(bot_id, ft, shop_pos[0], shop_pos[1])
            return

        # Stockpile high-value cooked items (meat/egg) when pan is idle
        if cooker_pos and pan_food is None and not profile.get("small_map"):
            horizon = self.tuning.get("future_horizon", 90)
            if self._cached_orders_turn == controller.get_turn() and self._cached_orders_own is not None:
                orders = self._cached_orders_own
            else:
                orders = controller.get_orders(controller.get_team())
            active_orders = [o for o in orders if o.get("is_active")]
            future_orders = [o for o in orders if o.get("created_turn", 0) >= controller.get_turn() and o.get("created_turn", 0) <= controller.get_turn() + horizon]
            freq: Dict[int, int] = defaultdict(int)
            for o in active_orders + future_orders:
                for name in o.get("required", []):
                    ft = FOOD_BY_NAME.get(name)
                    if ft is not None:
                        freq[ft.food_id] += 1
            for ft in [FoodType.MEAT, FoodType.EGG]:
                if freq.get(ft.food_id, 0) >= 2 and not self._has_ready_food(controller, map_team, ft):
                    box_pos = self._find_box_with_food_type(controller, map_team, ft, ws.get("cooker"))
                    if box_pos and self._move_towards(controller, bot_id, map_team, box_pos):
                        controller.pickup(bot_id, box_pos[0], box_pos[1])
                        return
                    if shop_pos and self._move_towards(controller, bot_id, map_team, shop_pos):
                        if controller.get_team_money(controller.get_team()) >= ft.buy_cost:
                            controller.buy(bot_id, ft, shop_pos[0], shop_pos[1])
                    return

        has_plate_bot = any(role == "plate" for role in self.bot_roles.values())
        missing_simple = [ft for ft in missing if not ft.can_cook and not ft.can_chop]
        for ft in list(missing_simple):
            if self._find_ready_food_on_counters(controller, map_team, ft) is not None:
                missing_simple.remove(ft)

        if missing_simple and (not has_plate_bot or single_counter):
            ft = missing_simple[0]
            box_pos = self._find_box_with_food(controller, map_team, ft, ws.get("plate_counter"))
            if box_pos and self._move_towards(controller, bot_id, map_team, box_pos):
                controller.pickup(bot_id, box_pos[0], box_pos[1])
                return
            if shop_pos and self._move_towards(controller, bot_id, map_team, shop_pos):
                if controller.get_team_money(controller.get_team()) >= ft.buy_cost:
                    controller.buy(bot_id, ft, shop_pos[0], shop_pos[1])
            return

        # idle near cooker
        if cooker_pos:
            self._move_towards(controller, bot_id, map_team, cooker_pos)

    def _play_plate(self, controller: RobotController, bot_id: int, map_team: Team, required: List[FoodType]) -> None:
        bot_state = controller.get_bot_state(bot_id)
        if bot_state is None:
            return
        holding = bot_state.get("holding")
        bx, by = bot_state["x"], bot_state["y"]

        ws = self._get_reachable_workstations(controller, map_team, (bx, by))
        plate_counter = ws.get("plate_counter")
        plate_pos = self._find_plate_on_counters(controller, map_team, (bx, by))
        if plate_pos and plate_pos != plate_counter:
            plate_counter = plate_pos
        if plate_counter is None:
            reachable_plate = self._find_counter_reachable(controller, map_team, (bx, by), ws.get("submit"))
            if reachable_plate:
                plate_counter = reachable_plate
        submit_pos = ws.get("submit")
        shop_pos = ws.get("shop")
        sinktable_pos = ws.get("sinktable")
        sink_pos = ws.get("sink")
        missing = self._missing_for_plate(required, self._plate_at_counter_or_bot(controller, map_team, (bx, by)))
        cookers = self.tile_cache.get(map_team, {}).get("COOKER", [])
        trash_pos = ws.get("trash")

        # If holding a dirty plate, put it in the sink first
        if holding and holding.get("type") == "Plate" and holding.get("dirty"):
            if sink_pos and self._move_towards(controller, bot_id, map_team, sink_pos):
                controller.put_dirty_plate_in_sink(bot_id, sink_pos[0], sink_pos[1])
            return

        # Help chop missing choppable items if they are on counters
        counters = self.tile_cache.get(map_team, {}).get("COUNTER", [])
        for counter_pos in counters:
            tile = controller.get_tile(map_team, counter_pos[0], counter_pos[1])
            if tile is not None and isinstance(getattr(tile, "item", None), Food):
                item = tile.item
                if item.can_chop and not item.chopped:
                    if holding is None and self._move_towards(controller, bot_id, map_team, counter_pos):
                        controller.chop(bot_id, counter_pos[0], counter_pos[1])
                    return

        # If holding cookable food, place it into an empty cooker
        if holding and holding.get("type") == "Food":
            ft = FOOD_BY_NAME.get(holding.get("food_name"))
            if ft and ft.can_cook and holding.get("cooked_stage", 0) == 0 and (not ft.can_chop or holding.get("chopped", False)):
                empty_cooker = self._find_empty_cooker(controller, map_team, (bot_state["x"], bot_state["y"]))
                if empty_cooker and self._move_towards(controller, bot_id, map_team, empty_cooker):
                    controller.place(bot_id, empty_cooker[0], empty_cooker[1])
                return

        # if plate counter is blocked, move to another empty counter
        if plate_counter:
            tile = controller.get_tile(map_team, plate_counter[0], plate_counter[1])
            if tile is not None and getattr(tile, "item", None) is not None and not isinstance(tile.item, Plate):
                new_counter = self._find_empty_counter_near(controller, map_team, ws.get("submit"))
                if new_counter is not None:
                    ws["plate_counter"] = new_counter
                    plate_counter = new_counter

        single_counter = self._is_single_counter_map(map_team)

        # On single-counter maps, keep plate in hand to avoid blocking prep
        if single_counter and holding is None and plate_counter:
            tile = controller.get_tile(map_team, plate_counter[0], plate_counter[1])
            if tile is not None and isinstance(getattr(tile, "item", None), Plate):
                if self._move_towards(controller, bot_id, map_team, plate_counter):
                    controller.pickup(bot_id, plate_counter[0], plate_counter[1])
                return

        # if holding a plate, try submit when complete or add food from counters
        if holding and holding.get("type") == "Plate":
            if holding.get("food") and (not self._plate_subset_of_order(holding, required) or self._plate_has_uncooked_cookable(holding)):
                if trash_pos and self._move_towards(controller, bot_id, map_team, trash_pos):
                    controller.trash(bot_id, trash_pos[0], trash_pos[1])
                    return
                empty_counter = self._find_empty_counter_near(controller, map_team, ws.get("submit")) or plate_counter
                if empty_counter and self._move_towards(controller, bot_id, map_team, empty_counter):
                    controller.place(bot_id, empty_counter[0], empty_counter[1])
                return
            missing = self._missing_for_plate(required, holding)
            if not missing and submit_pos:
                if not self._plate_exact_match(holding, required):
                    if trash_pos and self._move_towards(controller, bot_id, map_team, trash_pos):
                        controller.trash(bot_id, trash_pos[0], trash_pos[1])
                    return
                if self._move_towards(controller, bot_id, map_team, submit_pos):
                    controller.submit(bot_id, submit_pos[0], submit_pos[1])
                return
            if missing and not holding.get("dirty"):
                for ft in missing:
                    pos = self._find_ready_food_on_counters(controller, map_team, ft)
                    if pos and self._move_towards(controller, bot_id, map_team, pos):
                        controller.add_food_to_plate(bot_id, pos[0], pos[1])
                        return
            if not single_counter:
                empty_counter = self._find_empty_counter_near(controller, map_team, submit_pos) or plate_counter
                if empty_counter:
                    tile = controller.get_tile(map_team, empty_counter[0], empty_counter[1])
                    if tile is not None and getattr(tile, "item", None) is None:
                        if self._move_towards(controller, bot_id, map_team, empty_counter):
                            controller.place(bot_id, empty_counter[0], empty_counter[1])
                            if empty_counter != plate_counter:
                                plate_counter = empty_counter
            return

        # ensure we have a plate (in hand or on counter); on single-counter maps keep plate in hand
        if plate_counter:
            tile = controller.get_tile(map_team, plate_counter[0], plate_counter[1])
            tile_item = getattr(tile, "item", None) if tile else None
            if tile is None or not isinstance(tile_item, Plate):
                # place plate on counter only when we have two counters (so cook can use the other)
                if holding and holding.get("type") == "Plate" and not single_counter:
                    place_counter = self._find_empty_counter_near(controller, map_team, ws.get("submit")) or plate_counter
                    if place_counter:
                        ptile = controller.get_tile(map_team, place_counter[0], place_counter[1])
                        if ptile is not None and getattr(ptile, "item", None) is None:
                            if self._move_towards(controller, bot_id, map_team, place_counter):
                                controller.place(bot_id, place_counter[0], place_counter[1])
                                if place_counter != plate_counter:
                                    ws["plate_counter"] = place_counter
                    return
                # otherwise acquire a plate (keep in hand on single-counter so we can add from counters)
                if holding is None:
                    # prefer sinktable if it has plates (from washing)
                    if sinktable_pos:
                        sink_tile = controller.get_tile(map_team, sinktable_pos[0], sinktable_pos[1])
                        if sink_tile is not None and getattr(sink_tile, "num_clean_plates", 0) > 0:
                            if self._move_towards(controller, bot_id, map_team, sinktable_pos):
                                controller.take_clean_plate(bot_id, sinktable_pos[0], sinktable_pos[1])
                            return
                    if self._shop_for_items(controller, bot_id, map_team, [ShopCosts.PLATE]):
                        return

        # batch shopping for simple ingredients when needed
        if holding is None and missing and not single_counter:
            shop_simple = []
            for ft in missing:
                if not ft.can_cook and not ft.can_chop:
                    if self._find_ready_food_on_counters(controller, map_team, ft) is None and not self._box_has_food(controller, map_team, ft):
                        shop_simple.append(ft)
            if shop_simple and self._shop_for_items(controller, bot_id, map_team, shop_simple):
                return

        # If choppable items are missing, bring raw to prep counter
        if holding is None and missing:
            prep_counter = ws.get("prep_counter")
            for ft in missing:
                if ft.can_chop and not ft.can_cook:
                    box_pos = self._find_box_with_food_type(controller, map_team, ft, prep_counter)
                    if box_pos and self._move_towards(controller, bot_id, map_team, box_pos):
                        controller.pickup(bot_id, box_pos[0], box_pos[1])
                        return

        # Help cook if there are cookable/choppable items missing and a pan is empty
        if holding is None and missing:
            empty_cooker = self._find_empty_cooker(controller, map_team, (bot_state["x"], bot_state["y"]))
            if empty_cooker:
                for ft in missing:
                    if ft.can_cook:
                        pos = self._find_ready_food_on_counters(controller, map_team, ft)
                        if pos and self._move_towards(controller, bot_id, map_team, pos):
                            controller.pickup(bot_id, pos[0], pos[1])
                            return
                        box_pos = self._find_box_with_food_type(controller, map_team, ft, empty_cooker)
                        if box_pos and self._move_towards(controller, bot_id, map_team, box_pos):
                            controller.pickup(bot_id, box_pos[0], box_pos[1])
                            return

        # check plate on counter for completion (only if we're not already holding a plate)
        if not (holding and holding.get("type") == "Plate"):
            plate_obj = self._plate_at_counter_or_bot(controller, map_team, (bx, by))
            if plate_obj is not None:
                missing = self._missing_for_plate(required, plate_obj)
                if not missing:
                    if not self._plate_exact_match(plate_obj, required):
                        if isinstance(plate_obj, Plate) and plate_counter:
                            if self._move_towards(controller, bot_id, map_team, plate_counter):
                                controller.pickup(bot_id, plate_counter[0], plate_counter[1])
                                trash_pos = ws.get("trash")
                                if trash_pos and self._move_towards(controller, bot_id, map_team, trash_pos):
                                    controller.trash(bot_id, trash_pos[0], trash_pos[1])
                        return
                    # Plate is complete - pick it up if on counter (will submit next turn)
                    if isinstance(plate_obj, Plate) and plate_counter:
                        # Plate is on counter, pick it up
                        if self._move_towards(controller, bot_id, map_team, plate_counter):
                            controller.pickup(bot_id, plate_counter[0], plate_counter[1])
                    return
            else:
                missing = self._missing_for_plate(required, None)
        else:
            # We're holding a plate, already handled above
            missing = self._missing_for_plate(required, holding)

        # if holding food, add to plate
        if holding and holding.get("type") == "Food":
            ft = FOOD_BY_NAME.get(holding.get("food_name"))
            if ft is not None and ft.can_chop and not holding.get("chopped", False):
                prep_counter = ws.get("prep_counter")
                if prep_counter and self._move_towards(controller, bot_id, map_team, prep_counter):
                    controller.place(bot_id, prep_counter[0], prep_counter[1])
                return
            ft = FOOD_BY_NAME.get(holding.get("food_name"))
            if ft and ft.can_cook and holding.get("cooked_stage", 0) == 0 and (not ft.can_chop or holding.get("chopped", False)):
                empty_cooker = self._find_empty_cooker(controller, map_team, (bot_state["x"], bot_state["y"]))
                if empty_cooker and self._move_towards(controller, bot_id, map_team, empty_cooker):
                    controller.place(bot_id, empty_cooker[0], empty_cooker[1])
                    return
            # First check if we're holding a plate (can add food directly)
            plate_obj = self._plate_at_counter_or_bot(controller, map_team)
            if plate_obj and isinstance(plate_obj, dict) and plate_obj.get("type") == "Plate" and not plate_obj.get("dirty"):
                # We're holding a plate - add food to it by placing food on counter then adding
                # Actually, we can't add food to plate in hand directly - need plate on counter
                # So place food on counter first, then next turn we can add it
                if plate_counter:
                    tile = controller.get_tile(map_team, plate_counter[0], plate_counter[1])
                    if tile is not None and getattr(tile, "item", None) is None:
                        if self._move_towards(controller, bot_id, map_team, plate_counter):
                            controller.place(bot_id, plate_counter[0], plate_counter[1])
                    return
            # Otherwise, check if there's a plate on counter
            if plate_counter:
                tile = controller.get_tile(map_team, plate_counter[0], plate_counter[1])
                tile_item = getattr(tile, "item", None) if tile else None
                if isinstance(tile_item, Plate) and not tile_item.dirty:
                    if self._move_towards(controller, bot_id, map_team, plate_counter):
                        controller.add_food_to_plate(bot_id, plate_counter[0], plate_counter[1])
                    return
            # No plate available - place food on counter for later
            empty_counter = self._find_empty_counter_near(controller, map_team, plate_counter) or plate_counter
            if empty_counter:
                tile = controller.get_tile(map_team, empty_counter[0], empty_counter[1])
                if tile is not None and getattr(tile, "item", None) is None:
                    if self._move_towards(controller, bot_id, map_team, empty_counter):
                        controller.place(bot_id, empty_counter[0], empty_counter[1])
            return

        # fetch next missing ingredient (skip when single_counter — we only add from counters with plate in hand)
        if missing and not single_counter:
            # prefer ready food on counters
            for ft in missing:
                pos = self._find_ready_food_on_counters(controller, map_team, ft)
                if pos:
                    if self._move_towards(controller, bot_id, map_team, pos):
                        controller.pickup(bot_id, pos[0], pos[1])
                    return

            # then try boxes (free ingredients)
            for ft in missing:
                box_pos = self._find_box_with_food(controller, map_team, ft, ws.get("plate_counter"))
                if box_pos and self._move_towards(controller, bot_id, map_team, box_pos):
                    controller.pickup(bot_id, box_pos[0], box_pos[1])
                    return

            # buy simple ingredients
            for ft in missing:
                if not ft.can_chop and not ft.can_cook:
                    if shop_pos and self._move_towards(controller, bot_id, map_team, shop_pos):
                        if controller.get_team_money(controller.get_team()) >= ft.buy_cost:
                            controller.buy(bot_id, ft, shop_pos[0], shop_pos[1])
                    return

        # If idle and no missing ingredients, help with plate washing to keep station clean
        if not missing and sink_pos:
            sink_tile = controller.get_tile(map_team, sink_pos[0], sink_pos[1])
            if sink_tile is not None and getattr(sink_tile, "num_dirty_plates", 0) > 0:
                if self._move_towards(controller, bot_id, map_team, sink_pos):
                    controller.wash_sink(bot_id, sink_pos[0], sink_pos[1])
                return

        # idle near plate counter
        if plate_counter:
            self._move_towards(controller, bot_id, map_team, plate_counter)

    def _play_solo(self, controller: RobotController, bot_id: int, map_team: Team, required: List[FoodType]) -> None:
        bot_state = controller.get_bot_state(bot_id)
        if bot_state is None:
            return
        holding = bot_state.get("holding")
        bx, by = bot_state["x"], bot_state["y"]

        plate_obj = self._plate_at_counter_or_bot(controller, map_team, (bx, by))
        missing = self._missing_for_plate(required, plate_obj)

        ready = False
        for ft in missing:
            if self._find_ready_food_on_counters(controller, map_team, ft) is not None:
                ready = True
                break

        ws = self._get_reachable_workstations(controller, map_team, (bx, by))
        cooker_pos = ws.get("cooker")
        cooked_in_pan = False
        if cooker_pos:
            tile = controller.get_tile(map_team, cooker_pos[0], cooker_pos[1])
            pan = getattr(tile, "item", None) if tile else None
            if isinstance(pan, Pan) and isinstance(getattr(pan, "food", None), Food):
                if pan.food.cooked_stage == 1:
                    cooked_in_pan = True
                    for ft in missing:
                        if pan.food.food_id == ft.food_id:
                            ready = True
                            break

        if plate_obj is not None and cooked_in_pan and holding is None:
            self._play_cook(controller, bot_id, map_team, required)
            return

        if holding and holding.get("type") == "Plate":
            self._play_plate(controller, bot_id, map_team, required)
            return
        if holding and holding.get("type") == "Food" and plate_obj is not None:
            self._play_plate(controller, bot_id, map_team, required)
            return

        if plate_obj is not None:
            if not ready and any(ft.can_cook or ft.can_chop for ft in missing):
                self._play_cook(controller, bot_id, map_team, required)
                return
            if ready or len(missing) < len(required):
                self._play_plate(controller, bot_id, map_team, required)
                return

        if plate_obj is None and ready:
            self._play_plate(controller, bot_id, map_team, required)
            return

        self._play_cook(controller, bot_id, map_team, required)

    def _play_support(self, controller: RobotController, bot_id: int, map_team: Team, required: List[FoodType]) -> None:
        bot_state = controller.get_bot_state(bot_id)
        if bot_state is None:
            return
        holding = bot_state.get("holding")

        ws = self.workstations[map_team]
        prep_counter = self._find_empty_counter_reachable(controller, map_team, (bot_state["x"], bot_state["y"])) or self._find_empty_counter_near(controller, map_team, ws.get("prep_counter")) or ws.get("prep_counter")
        shop_pos = ws.get("shop")
        sink_pos = ws.get("sink")

        # If holding a dirty plate, put it in the sink
        if holding and holding.get("type") == "Plate" and holding.get("dirty"):
            if sink_pos and self._move_towards(controller, bot_id, map_team, sink_pos):
                controller.put_dirty_plate_in_sink(bot_id, sink_pos[0], sink_pos[1])
            return

        # Help harvest cooked food from any cooker and stage it on counters (only reachable cookers on multi-kitchen maps)
        cookers = self.tile_cache.get(map_team, {}).get("COOKER", [])
        plate_counter = ws.get("plate_counter")
        if holding is None and cookers:
            best_cooker = None
            best_dist = 10**9
            for pos in cookers:
                if self._bfs_distance(controller, map_team, (bx, by), pos) is None:
                    continue
                tile = controller.get_tile(map_team, pos[0], pos[1])
                pan = getattr(tile, "item", None) if tile else None
                if isinstance(pan, Pan) and isinstance(getattr(pan, "food", None), Food):
                    if pan.food.cooked_stage >= 1:
                        dist = self._bfs_distance(controller, map_team, (bx, by), pos) or 10**9
                        if dist < best_dist:
                            best_dist = dist
                            best_cooker = pos
            if best_cooker and self._move_towards(controller, bot_id, map_team, best_cooker):
                controller.take_from_pan(bot_id, best_cooker[0], best_cooker[1])
                return

        if holding and holding.get("type") == "Food":
            ft = FOOD_BY_NAME.get(holding.get("food_name"))
            if ft is not None and (not ft.can_cook or holding.get("cooked_stage", 0) >= 1) and (not ft.can_chop or holding.get("chopped", False)):
                place_pos = self._find_empty_counter_near(controller, map_team, plate_counter) or plate_counter
                if place_pos:
                    tile = controller.get_tile(map_team, place_pos[0], place_pos[1])
                    if tile is not None and getattr(tile, "item", None) is None:
                        if self._move_towards(controller, bot_id, map_team, place_pos):
                            controller.place(bot_id, place_pos[0], place_pos[1])
                        return
                box_pos = self._find_box_for_store(controller, map_team, ft)
                if box_pos and self._move_towards(controller, bot_id, map_team, box_pos):
                    controller.place(bot_id, box_pos[0], box_pos[1])
                return

        if holding and holding.get("type") == "Food":
            ft = FOOD_BY_NAME.get(holding.get("food_name"))
            if ft is None:
                return
            if ft.can_chop and not holding.get("chopped", False):
                if prep_counter and self._move_towards(controller, bot_id, map_team, prep_counter):
                    controller.place(bot_id, prep_counter[0], prep_counter[1])
                return

        # chop anything on prep counter
        if prep_counter:
            tile = controller.get_tile(map_team, prep_counter[0], prep_counter[1])
            if tile is not None and isinstance(getattr(tile, "item", None), Food):
                item = tile.item
                if item.can_chop and not item.chopped:
                    if holding is None and self._move_towards(controller, bot_id, map_team, prep_counter):
                        controller.chop(bot_id, prep_counter[0], prep_counter[1])
                    return

        # buy missing choppable-only ingredient to speed prep (try box first)
        missing = self._missing_for_plate(required, self._plate_at_counter_or_bot(controller, map_team, (bx, by)))
        for ft in missing:
            if ft.can_chop and not ft.can_cook:
                box_pos = self._find_box_with_food(controller, map_team, ft, prep_counter)
                if box_pos and self._move_towards(controller, bot_id, map_team, box_pos):
                    controller.pickup(bot_id, box_pos[0], box_pos[1])
                    return
                if shop_pos and self._move_towards(controller, bot_id, map_team, shop_pos):
                    if controller.get_team_money(controller.get_team()) >= ft.buy_cost:
                        controller.buy(bot_id, ft, shop_pos[0], shop_pos[1])
                return

        # If a cookable item is missing and a cooker is idle, bring raw to cook
        empty_cooker = self._find_empty_cooker(controller, map_team, (bot_state["x"], bot_state["y"]))
        if holding is None and empty_cooker:
            for ft in missing:
                if ft.can_cook:
                    box_pos = self._find_box_with_food_type(controller, map_team, ft, empty_cooker)
                    if box_pos and self._move_towards(controller, bot_id, map_team, box_pos):
                        controller.pickup(bot_id, box_pos[0], box_pos[1])
                        return
                    if shop_pos and self._move_towards(controller, bot_id, map_team, shop_pos):
                        if controller.get_team_money(controller.get_team()) >= ft.buy_cost:
                            controller.buy(bot_id, ft, shop_pos[0], shop_pos[1])
                    return

        # Prioritize washing dirty plates to keep station clean (guide emphasizes cleanliness)
        # Check sink for dirty plates
        if sink_pos:
            sink_tile = controller.get_tile(map_team, sink_pos[0], sink_pos[1])
            if sink_tile is not None and getattr(sink_tile, "num_dirty_plates", 0) > 0:
                # If holding dirty plate, put it in sink first
                if holding and holding.get("type") == "Plate" and holding.get("dirty"):
                    if self._move_towards(controller, bot_id, map_team, sink_pos):
                        controller.put_dirty_plate_in_sink(bot_id, sink_pos[0], sink_pos[1])
                    return
                # Otherwise, wash plates if not holding anything
                elif holding is None:
                    if self._move_towards(controller, bot_id, map_team, sink_pos):
                        controller.wash_sink(bot_id, sink_pos[0], sink_pos[1])
                    return

        if prep_counter:
            self._move_towards(controller, bot_id, map_team, prep_counter)

    # ----------------------------
    # Shared helpers
    # ----------------------------
    def _plate_at_counter_or_bot(self, controller: RobotController, map_team: Team, origin: Optional[Tuple[int, int]] = None):
        if origin is not None:
            ws = self._get_reachable_workstations(controller, map_team, origin)
            plate_counter = ws.get("plate_counter")
            if plate_counter:
                tile = controller.get_tile(map_team, plate_counter[0], plate_counter[1])
                if tile is not None and isinstance(getattr(tile, "item", None), Plate):
                    return tile.item
        else:
            ws = self.workstations.get(map_team, {})
            plate_counter = ws.get("plate_counter")
            if plate_counter:
                tile = controller.get_tile(map_team, plate_counter[0], plate_counter[1])
                if tile is not None and isinstance(getattr(tile, "item", None), Plate):
                    return tile.item

        plate_pos = self._find_plate_on_counters(controller, map_team)
        if plate_pos:
            tile = controller.get_tile(map_team, plate_pos[0], plate_pos[1])
            if tile is not None and isinstance(getattr(tile, "item", None), Plate):
                if origin is None and map_team in self.workstations:
                    self.workstations[map_team]["plate_counter"] = plate_pos
                return tile.item

        # fallback to any bot holding a plate
        bot_ids = self._team_bot_ids if self._team_bot_ids is not None else controller.get_team_bot_ids(controller.get_team())
        for bid in bot_ids:
            bstate = controller.get_bot_state(bid)
            holding = (bstate or {}).get("holding") if bstate else None
            if holding and holding.get("type") == "Plate":
                return holding
        return None

    def _should_switch_for_sabotage(self, controller: RobotController) -> bool:
        info = controller.get_switch_info()
        if not info.get("window_active") or info.get("my_team_switched"):
            return False
        own_team = controller.get_team()
        enemy_team = controller.get_enemy_team()
        own_money = controller.get_team_money(own_team)
        enemy_money = controller.get_team_money(enemy_team)
        turn = controller.get_turn()
        if own_money < 500 and turn < 80:
            return False
        lead = own_money - enemy_money
        profile = self._get_map_profile(controller, own_team)
        remaining_enemy_reward = 0
        enemy_orders = self._cached_orders_enemy if self._cached_orders_turn == turn and self._cached_orders_enemy is not None else controller.get_orders(enemy_team)
        for o in enemy_orders:
            if o.get("completed_turn") is None and o.get("expires_turn", 0) >= turn:
                remaining_enemy_reward += o.get("reward", 0)

        high_penalty = self._enemy_high_penalty_order(controller, enemy_team)
        if high_penalty is not None:
            self._sabotage_mode = "high_penalty"
            return True

        if profile.get("small_map"):
            return False

        cooldown = self.tuning.get("quick_sabotage_cooldown", 80)
        if remaining_enemy_reward >= 12000 and lead >= 2000 and own_money >= 2000 and self._low_pressure_on_own(controller) and (turn - self._last_quick_sabotage_turn) >= cooldown:
            self._sabotage_mode = "quick"
            return True

        if remaining_enemy_reward >= 40000:
            return True

        if not profile.get("small_map") and lead < 0 and remaining_enemy_reward >= 8000:
            return True

        turns_left = info.get("window_end_turn", turn) - turn
        if turns_left <= 20:
            lead_threshold = max(-1000, int(0.2 * remaining_enemy_reward))
        else:
            lead_threshold = max(-2000, int(0.3 * remaining_enemy_reward))
        if remaining_enemy_reward >= 30000:
            lead_threshold -= 2000
        return lead >= lead_threshold
