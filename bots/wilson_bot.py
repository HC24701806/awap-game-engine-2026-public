from collections import deque, defaultdict
from typing import Tuple, Optional, List, Dict
import json
import os

from game_constants import Team, FoodType, ShopCosts, GameConstants
from robot_controller import RobotController
from item import Pan, Plate, Food
from tiles import Box

FOOD_BY_NAME = {ft.food_name: ft for ft in FoodType}
FOOD_BY_ID = {ft.food_id: ft for ft in FoodType}

DIRECTIONS = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
]

DEFAULT_TUNING = {
    "claim_bonus_none": 1000,
    "claim_bonus_ours": 500,
    "plate_exact_bonus": 5000,
    "plate_subset_bonus": 2000,
    "plate_mismatch_penalty": 2000,
    "infeasible_multiplier": 0.1,
    "value_penalty_weight": 0.5,
    "shop_overhead": 6,
    "travel_weight": 1.0,
}


class BotPlayer:
    def __init__(self, map_copy):
        self.map = map_copy
        self.map_width = map_copy.width
        self.map_height = map_copy.height

        self.tile_cache: Dict[Team, Dict[str, List[Tuple[int, int]]]] = {}
        self.workstations: Dict[Team, Dict[str, Tuple[int, int]]] = {}
        self.bot_roles: Dict[int, str] = {}
        self.last_order_id: Optional[int] = None
        self.has_switched = False
        self._turn_id: Optional[int] = None
        self._moved_bots: set = set()
        self.disable_sabotage = False
        self.enable_switch = True
        self.tuning = self._load_tuning()

    def _load_tuning(self) -> Dict[str, float]:
        tuning = dict(DEFAULT_TUNING)
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        path = os.path.join(base_dir, "tools", "wilson_tuning.json")
        if os.path.exists(path):
            try:
                with open(path, "r") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    tuning.update(data)
            except Exception:
                pass
        return tuning

    def _reset_turn_state(self, turn: int) -> None:
        if self._turn_id != turn:
            self._turn_id = turn
            self._moved_bots = set()

    def _get_map_dims(self, controller: RobotController, map_team: Team) -> Tuple[int, int]:
        m = controller.get_map(map_team)
        self.map_width = m.width
        self.map_height = m.height
        return self.map_width, self.map_height

    # ----------------------------
    # Map helpers
    # ----------------------------
    def _chebyshev(self, a: Tuple[int, int], b: Tuple[int, int]) -> int:
        return max(abs(a[0] - b[0]), abs(a[1] - b[1]))

    def _ensure_tile_cache(self, controller: RobotController, map_team: Team) -> None:
        if map_team in self.tile_cache:
            return

        positions: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
        w, h = self._get_map_dims(controller, map_team)
        for x in range(w):
            for y in range(h):
                tile = controller.get_tile(map_team, x, y)
                if tile is None:
                    continue
                positions[tile.tile_name].append((x, y))

        self.tile_cache[map_team] = positions

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

    def _find_nearest(self, origin: Tuple[int, int], positions: List[Tuple[int, int]]) -> Optional[Tuple[int, int]]:
        if not positions:
            return None
        return min(positions, key=lambda p: self._chebyshev(origin, p))

    # ----------------------------
    # Pathfinding
    # ----------------------------
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

        while queue:
            cx, cy = queue.popleft()
            if target_predicate(cx, cy):
                break

            for dx, dy in DIRECTIONS:
                nx, ny = cx + dx, cy + dy
                if (nx, ny) in came_from:
                    continue
                if (nx, ny) in blocked:
                    continue
                if not (0 <= nx < w and 0 <= ny < h):
                    continue
                tile = controller.get_tile(map_team, nx, ny)
                if tile is None or not tile.is_walkable:
                    continue
                came_from[(nx, ny)] = (cx, cy)
                queue.append((nx, ny))

        # find any goal found
        goals = [pos for pos in came_from if target_predicate(pos[0], pos[1])]
        if not goals:
            return None

        goal = goals[0]
        # backtrack to get first step
        curr = goal
        while came_from[curr] != start and came_from[curr] is not None:
            curr = came_from[curr]
        if came_from[curr] is None:
            return None
        return (curr[0] - start[0], curr[1] - start[1])

    def _bfs_distance(self, controller: RobotController, map_team: Team, start: Tuple[int, int], goal: Tuple[int, int]) -> Optional[int]:
        """Return shortest steps from start to goal using walkable tiles, or None if unreachable."""
        if start == goal:
            return 0
        queue = deque([(start[0], start[1], 0)])
        seen = {start}
        w, h = self._get_map_dims(controller, map_team)
        while queue:
            cx, cy, dist = queue.popleft()
            for dx, dy in DIRECTIONS:
                nx, ny = cx + dx, cy + dy
                if not (0 <= nx < w and 0 <= ny < h):
                    continue
                if (nx, ny) in seen:
                    continue
                tile = controller.get_tile(map_team, nx, ny)
                if tile is None or not tile.is_walkable:
                    continue
                if (nx, ny) == goal:
                    return dist + 1
                seen.add((nx, ny))
                queue.append((nx, ny, dist + 1))
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
        for other_id in controller.get_team_bot_ids(controller.get_team()):
            if other_id == bot_id:
                continue
            other = controller.get_bot_state(other_id)
            if other and other.get("map_team") == map_team.name:
                blocked.add((other["x"], other["y"]))

        def is_adjacent(x: int, y: int) -> bool:
            return self._chebyshev((x, y), target) <= 1

        step = self._get_bfs_step(controller, map_team, (bx, by), is_adjacent, blocked=blocked)
        if step is None:
            return False
        if controller.can_move(bot_id, step[0], step[1]):
            controller.move(bot_id, step[0], step[1])
            self._moved_bots.add(bot_id)
            # After moving, we may now be adjacent — allow same-turn action
            after = controller.get_bot_state(bot_id)
            if after and self._chebyshev((after["x"], after["y"]), target) <= 1:
                return True
        return False

    # ----------------------------
    # Order helpers
    # ----------------------------
    def _select_active_order(self, controller: RobotController) -> Optional[Dict]:
        orders = controller.get_orders(controller.get_team())
        if not orders:
            return None

        turn = controller.get_turn()
        bot_ids = controller.get_team_bot_ids(controller.get_team())
        def is_active_order(o: Dict) -> bool:
            if "is_active" in o:
                return bool(o.get("is_active"))
            if o.get("completed_turn") is not None:
                return False
            created = o.get("created_turn")
            expires = o.get("expires_turn")
            if isinstance(created, int) and created > turn:
                return False
            if isinstance(expires, int) and expires < turn:
                return False
            return True

        active = [o for o in orders if is_active_order(o)]
        if not active:
            return None

        map_team = controller.get_team()
        plate_obj = self._plate_at_counter_or_bot(controller, map_team)
        plate_has_food = bool(self._plate_food_dicts(plate_obj))

        def ready_counts(map_team: Team) -> Dict[int, int]:
            counts: Dict[int, int] = defaultdict(int)
            plate = self._plate_at_counter_or_bot(controller, map_team)
            for fd in self._plate_food_dicts(plate):
                fid = fd.get("food_id")
                if fid is not None:
                    counts[fid] += 1
            for pos in self.tile_cache.get(map_team, {}).get("COUNTER", []):
                tile = controller.get_tile(map_team, pos[0], pos[1])
                item = getattr(tile, "item", None) if tile else None
                if isinstance(item, Food):
                    counts[item.food_id] += 1
            ws = self.workstations.get(map_team, {})
            cooker_pos = ws.get("cooker")
            if cooker_pos:
                tile = controller.get_tile(map_team, cooker_pos[0], cooker_pos[1])
                pan = getattr(tile, "item", None) if tile else None
                if isinstance(pan, Pan) and isinstance(pan.food, Food):
                    counts[pan.food.food_id] += 1
            for bid in controller.get_team_bot_ids(controller.get_team()):
                bstate = controller.get_bot_state(bid)
                holding = bstate.get("holding") if bstate else None
                if holding and holding.get("type") == "Food":
                    fid = holding.get("food_id")
                    if fid is not None:
                        counts[fid] += 1
            return counts

        def estimate_time(o: Dict) -> float:
            req = self._required_food_types(o)
            counts = ready_counts(map_team)
            missing: List[FoodType] = []
            for ft in req:
                if counts.get(ft.food_id, 0) > 0:
                    counts[ft.food_id] -= 1
                else:
                    missing.append(ft)

            cookables = [ft for ft in missing if ft.can_cook]
            choppable = [ft for ft in missing if ft.can_chop and not ft.can_cook]
            simple = [ft for ft in missing if not ft.can_cook and not ft.can_chop]

            cook_time = 20 * len(cookables) + 5 * len(cookables)
            chop_time = 3 * len(choppable)
            simple_time = 2 * len(simple)

            need_shop = False
            for ft in missing:
                if not self._box_has_food(controller, map_team, ft):
                    need_shop = True
                    break
            shop_overhead = self.tuning.get("shop_overhead", 6) if need_shop else 0

            # Travel-time estimates using BFS distances between key workstations
            ws = self.workstations.get(map_team, {})
            prep = ws.get("prep_counter")
            shop_pos = ws.get("shop")
            submit = ws.get("submit")

            def min_bot_dist_to(target: Optional[Tuple[int, int]]) -> Optional[int]:
                if target is None:
                    return None
                best = None
                for bid in controller.get_team_bot_ids(controller.get_team()):
                    b = controller.get_bot_state(bid)
                    if not b:
                        continue
                    start = (b["x"], b["y"])
                    d = self._bfs_distance(controller, map_team, start, target)
                    if d is None:
                        d = self._chebyshev(start, target)
                    if best is None or d < best:
                        best = d
                return best

            travel_time = 0
            bot_to_prep = min_bot_dist_to(prep)
            if bot_to_prep is not None:
                travel_time += bot_to_prep

            if need_shop and shop_pos:
                bot_to_shop = min_bot_dist_to(shop_pos)
                if bot_to_shop is not None and prep is not None:
                    shop_to_prep = self._bfs_distance(controller, map_team, shop_pos, prep)
                    if shop_to_prep is None:
                        shop_to_prep = self._chebyshev(shop_pos, prep)
                    travel_time = min(travel_time, bot_to_shop + shop_to_prep) if travel_time else bot_to_shop + shop_to_prep

            if prep and submit:
                p_to_s = self._bfs_distance(controller, map_team, prep, submit)
                if p_to_s is None:
                    p_to_s = self._chebyshev(prep, submit)
                travel_time += max(0, (p_to_s // 2))

            travel_weight = self.tuning.get("travel_weight", 1.0)
            travel_time *= travel_weight

            return cook_time + chop_time + simple_time + shop_overhead + travel_time + 2

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

            value_penalty_weight = self.tuning.get("value_penalty_weight", 0.5)
            value = o["reward"] + (value_penalty_weight * o["penalty"])
            base = value / max(1.0, est)
            if not feasible:
                base *= self.tuning.get("infeasible_multiplier", 0.1)

            if plate_has_food:
                if self._plate_exact_match(plate_obj, req):
                    base += self.tuning.get("plate_exact_bonus", 5000)
                elif self._plate_subset_of_order(plate_obj, req):
                    base += self.tuning.get("plate_subset_bonus", 2000)
                else:
                    base -= self.tuning.get("plate_mismatch_penalty", 2000)

            return base + claim_bonus

        return max(active, key=score)

    def _select_enemy_order_for_sabotage(self, controller: RobotController, team: Team) -> Optional[Dict]:
        orders = controller.get_orders(team)
        if not orders:
            return None
        turn = controller.get_turn()
        active = [o for o in orders if o.get("is_active")]
        if not active:
            return None

        def sabotage_score(o: Dict) -> float:
            time_left = max(1, o.get("expires_turn", turn) - turn)
            req = self._required_food_types(o)
            cook_weight = sum(1 for ft in req if ft.can_cook or ft.can_chop)
            value = o.get("reward", 0) + o.get("penalty", 0)
            return (value / time_left) + (cook_weight * 150)

        return max(active, key=sabotage_score)

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
        counters = self.tile_cache[map_team].get("COUNTER", [])
        for pos in counters:
            tile = controller.get_tile(map_team, pos[0], pos[1])
            if tile is None:
                continue
            item = getattr(tile, "item", None)
            if isinstance(item, Food) and self._food_satisfies(item, ft):
                return pos
        return None

    def _find_any_food_on_counters(self, controller: RobotController, map_team: Team) -> Optional[Tuple[int, int]]:
        counters = self.tile_cache[map_team].get("COUNTER", [])
        for pos in counters:
            tile = controller.get_tile(map_team, pos[0], pos[1])
            if tile is None:
                continue
            if isinstance(getattr(tile, "item", None), Food):
                return pos
        return None

    def _has_food_id_on_counters(self, controller: RobotController, map_team: Team, ft: FoodType) -> bool:
        counters = self.tile_cache[map_team].get("COUNTER", [])
        for pos in counters:
            tile = controller.get_tile(map_team, pos[0], pos[1])
            item = getattr(tile, "item", None) if tile else None
            if isinstance(item, Food) and item.food_id == ft.food_id:
                return True
        return False

    def _find_empty_counter_near(self, controller: RobotController, map_team: Team, near: Optional[Tuple[int, int]]) -> Optional[Tuple[int, int]]:
        counters = self.tile_cache[map_team].get("COUNTER", [])
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
    def _assign_roles(self, bot_ids: List[int]) -> None:
        if self.bot_roles and all(bid in self.bot_roles for bid in bot_ids):
            return

        self.bot_roles = {}
        for i, bid in enumerate(sorted(bot_ids)):
            if len(bot_ids) == 1:
                self.bot_roles[bid] = "solo"
            elif i == 0:
                self.bot_roles[bid] = "cook"
            elif i == 1:
                self.bot_roles[bid] = "plate"
            else:
                self.bot_roles[bid] = "support"

    # ----------------------------
    # Sabotage logic (enemy map)
    # ----------------------------
    def _play_sabotage(self, controller: RobotController, bot_id: int, map_team: Team) -> None:
        self._ensure_tile_cache(controller, map_team)
        bot_state = controller.get_bot_state(bot_id)
        if bot_state is None:
            return
        holding = bot_state.get("holding")
        ws = self.workstations.get(map_team, {})
        trash_pos = ws.get("trash")

        counters = self.tile_cache.get(map_team, {}).get("COUNTER", [])
        bot_pos = (bot_state.get("x"), bot_state.get("y"))

        turn = controller.get_turn()
        switch_start = GameConstants.MIDGAME_SWITCH_TURN
        switch_end = GameConstants.MIDGAME_SWITCH_TURN + GameConstants.MIDGAME_SWITCH_DURATION
        turns_left = max(0, switch_end - turn)

        def dist(a: Tuple[int, int], b: Tuple[int, int]) -> int:
            return self._chebyshev(a, b)

        # If holding something, trash it first (or drop on counter to clutter)
        if holding:
            if trash_pos and self._move_towards(controller, bot_id, map_team, trash_pos):
                controller.trash(bot_id, trash_pos[0], trash_pos[1])
                return
            empty_counter = self._find_empty_counter_near(controller, map_team, ws.get("submit"))
            if empty_counter and self._move_towards(controller, bot_id, map_team, empty_counter):
                controller.place(bot_id, empty_counter[0], empty_counter[1])
            return

        # Determine best target to sabotage based on time left in switch window
        targets: List[Tuple[int, Tuple[int, int], str]] = []

        # Plate with food on counters (highest value)
        for pos in counters:
            tile = controller.get_tile(map_team, pos[0], pos[1])
            if tile is None:
                continue
            item = getattr(tile, "item", None)
            if isinstance(item, Plate) and not item.dirty and item.food:
                score = 1000 - dist(bot_pos, pos)
                targets.append((score, pos, "plate_food"))

        # Food in pan (prioritize interrupting cooking to toss it)
        cooker_pos = ws.get("cooker")
        if cooker_pos:
            tile = controller.get_tile(map_team, cooker_pos[0], cooker_pos[1])
            if tile is not None and isinstance(getattr(tile, "item", None), Pan):
                pan = tile.item
                if pan.food is not None:
                    cooking = getattr(pan.food, "cooked_stage", 0) == 0
                    # higher score if it's actively cooking (to deny progress)
                    score = (900 if cooking else 700) - dist(bot_pos, cooker_pos)
                    targets.append((score, cooker_pos, "pan_food"))
                    if cooking:
                        # hard-prioritize cooking pans so we can toss the food
                        if self._move_towards(controller, bot_id, map_team, cooker_pos):
                            controller.take_from_pan(bot_id, cooker_pos[0], cooker_pos[1])
                        return

        # Clean plates from sink table
        sinktable_pos = ws.get("sinktable")
        if sinktable_pos:
            tile = controller.get_tile(map_team, sinktable_pos[0], sinktable_pos[1])
            if tile is not None and getattr(tile, "num_clean_plates", 0) > 0:
                score = 400 - dist(bot_pos, sinktable_pos)
                targets.append((score, sinktable_pos, "clean_plate"))

        # Food from counters
        food_pos = self._find_any_food_on_counters(controller, map_team)
        if food_pos:
            score = 500 - dist(bot_pos, food_pos)
            targets.append((score, food_pos, "food_counter"))

        # Raw ingredients from boxes
        boxes = self.tile_cache.get(map_team, {}).get("BOX", [])
        for pos in boxes:
            tile = controller.get_tile(map_team, pos[0], pos[1])
            if tile is None or not isinstance(tile, Box):
                continue
            if getattr(tile, "count", 0) > 0:
                score = 200 - dist(bot_pos, pos)
                targets.append((score, pos, "box"))

        # If switch is about to end, prioritize blocking submit/cooker
        submit_pos = ws.get("submit")
        if turns_left <= 15 and submit_pos:
            if self._move_towards(controller, bot_id, map_team, submit_pos):
                return
            return

        if targets:
            targets.sort(reverse=True, key=lambda t: t[0])
            _, pos, kind = targets[0]
            if kind == "plate_food":
                if self._move_towards(controller, bot_id, map_team, pos):
                    controller.pickup(bot_id, pos[0], pos[1])
                return
            if kind == "pan_food":
                if self._move_towards(controller, bot_id, map_team, pos):
                    controller.take_from_pan(bot_id, pos[0], pos[1])
                return
            if kind == "clean_plate":
                if self._move_towards(controller, bot_id, map_team, pos):
                    controller.take_clean_plate(bot_id, pos[0], pos[1])
                return
            if kind == "food_counter" or kind == "box":
                if self._move_towards(controller, bot_id, map_team, pos):
                    controller.pickup(bot_id, pos[0], pos[1])
                return

        # Otherwise move toward submit or cooker to obstruct enemy submissions
        targets = [p for p in [submit_pos, cooker_pos] if p is not None]
        if targets:
            target = min(targets, key=lambda p: self._chebyshev(bot_pos, p))
            self._move_towards(controller, bot_id, map_team, target)

    # ----------------------------
    # Main turn
    # ----------------------------
    def play_turn(self, controller: RobotController):
        self._reset_turn_state(controller.get_turn())
        bot_ids = controller.get_team_bot_ids(controller.get_team())
        if not bot_ids:
            return

        self._assign_roles(bot_ids)

        # Decide when to switch for sabotage (best of Ethan/Wilson)
        if self.enable_switch and controller.can_switch_maps() and not self.has_switched:
            if self._should_switch_for_sabotage(controller):
                if controller.switch_maps():
                    self.has_switched = True

        first_state = controller.get_bot_state(bot_ids[0])
        if first_state is None:
            return
        home_team = controller.get_team()
        home_map_team = Team[first_state["map_team"]]
        self._ensure_tile_cache(controller, home_map_team)

        order = self._select_active_order(controller)

        if order is None:
            # still allow sabotage for any bot on enemy map
            for bid in bot_ids:
                bstate = controller.get_bot_state(bid)
                if not bstate:
                    continue
                b_map_team = Team[bstate["map_team"]]
                if not self.disable_sabotage and b_map_team != home_team:
                    self._ensure_tile_cache(controller, b_map_team)
                    self._play_sabotage(controller, bid, b_map_team)
            return

        required = self._required_food_types(order)

        for bid in bot_ids:
            bstate = controller.get_bot_state(bid)
            if not bstate:
                continue
            b_map_team = Team[bstate["map_team"]]
            if not self.disable_sabotage and b_map_team != home_team:
                self._ensure_tile_cache(controller, b_map_team)
                self._play_sabotage(controller, bid, b_map_team)
                continue

            role = self.bot_roles.get(bid, "support")
            if role == "solo":
                self._play_solo(controller, bid, b_map_team, required)
            elif role == "cook":
                self._play_cook(controller, bid, b_map_team, required)
            elif role == "plate":
                self._play_plate(controller, bid, b_map_team, required)
            else:
                self._play_support(controller, bid, b_map_team, required)

    # ----------------------------
    # Role behaviors
    # ----------------------------
    def _play_cook(self, controller: RobotController, bot_id: int, map_team: Team, required: List[FoodType]) -> None:
        bot_state = controller.get_bot_state(bot_id)
        if bot_state is None:
            return
        holding = bot_state.get("holding")
        bx, by = bot_state["x"], bot_state["y"]

        ws = self.workstations[map_team]
        prep_counter = self._find_empty_counter_near(controller, map_team, ws.get("prep_counter")) or ws.get("prep_counter")
        plate_counter = ws.get("plate_counter")
        cooker_pos = ws.get("cooker")
        shop_pos = ws.get("shop")
        trash_pos = ws.get("trash")
        sink_pos = ws.get("sink")
        missing = self._missing_for_plate(required, self._plate_at_counter_or_bot(controller, map_team))
        single_counter = prep_counter and plate_counter and prep_counter == plate_counter
        pan_food = None
        if cooker_pos:
            tile = controller.get_tile(map_team, cooker_pos[0], cooker_pos[1])
            pan = getattr(tile, "item", None) if tile else None
            if isinstance(pan, Pan) and isinstance(getattr(pan, "food", None), Food):
                pan_food = pan.food

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
                        # Counter not empty - find another empty counter
                        empty_counter = self._find_empty_counter_near(controller, map_team, prep_counter)
                        if empty_counter and empty_counter != prep_counter:
                            if self._move_towards(controller, bot_id, map_team, empty_counter):
                                controller.place(bot_id, empty_counter[0], empty_counter[1])
                                return
                        # try to place into box if counter is blocked
                        box_pos = self._find_box_for_store(controller, map_team, ft)
                        if box_pos and self._move_towards(controller, bot_id, map_team, box_pos):
                            controller.place(bot_id, box_pos[0], box_pos[1])
                            return
                        # No other counter - wait near prep counter (it should clear soon)
                        # On single-counter maps, plate bot should keep plate in hand
                        self._move_towards(controller, bot_id, map_team, prep_counter)
                        return
                return
            # cook if needed (only when pan is empty AND food is already chopped if it needs chopping)
            if ft.can_cook and holding.get("cooked_stage", 0) == 0:
                # Don't try to cook if it still needs chopping first
                if ft.can_chop and not holding.get("chopped", False):
                    # Should have been handled above, but just in case, skip cooking
                    return
                if cooker_pos:
                    tile = controller.get_tile(map_team, cooker_pos[0], cooker_pos[1])
                    pan = getattr(tile, "item", None) if tile else None
                    if isinstance(pan, Pan) and pan.food is None and self._move_towards(controller, bot_id, map_team, cooker_pos):
                        controller.place(bot_id, cooker_pos[0], cooker_pos[1])
                return
            # ready food: place on an empty counter (so plate bot can pick up; don't overwrite plate)
            empty_for_food = self._find_empty_counter_near(controller, map_team, plate_counter) or plate_counter
            if empty_for_food:
                tile = controller.get_tile(map_team, empty_for_food[0], empty_for_food[1])
                if tile is not None and getattr(tile, "item", None) is None:
                    if self._move_towards(controller, bot_id, map_team, empty_for_food):
                        controller.place(bot_id, empty_for_food[0], empty_for_food[1])
                        return
            # If no empty counter, store in a box to avoid getting stuck holding cooked food
            box_pos = self._find_box_for_store(controller, map_team, ft)
            if box_pos and self._move_towards(controller, bot_id, map_team, box_pos):
                controller.place(bot_id, box_pos[0], box_pos[1])
                return
            # Otherwise wait near plate counter
            if plate_counter:
                self._move_towards(controller, bot_id, map_team, plate_counter)
            return

        # no holding
        if cooker_pos:
            tile = controller.get_tile(map_team, cooker_pos[0], cooker_pos[1])
            if tile is not None and isinstance(getattr(tile, "item", None), Pan):
                pan = tile.item
                if pan.food is not None:
                    if pan.food.cooked_stage == 1:
                        if self._move_towards(controller, bot_id, map_team, cooker_pos):
                            controller.take_from_pan(bot_id, cooker_pos[0], cooker_pos[1])
                        return
                    if pan.food.cooked_stage == 2:
                        if self._move_towards(controller, bot_id, map_team, cooker_pos):
                            controller.take_from_pan(bot_id, cooker_pos[0], cooker_pos[1])
                        return

        # if we have burnt food in hand now, trash it
        holding = controller.get_bot_state(bot_id).get("holding")
        if holding and holding.get("type") == "Food":
            if holding.get("cooked_stage", 0) == 2 and trash_pos:
                if self._move_towards(controller, bot_id, map_team, trash_pos):
                    controller.trash(bot_id, trash_pos[0], trash_pos[1])
                return

        # chop food on prep counter if present (or any counter)
        counters = self.tile_cache[map_team].get("COUNTER", [])
        for counter_pos in counters:
            tile = controller.get_tile(map_team, counter_pos[0], counter_pos[1])
            if tile is not None and isinstance(getattr(tile, "item", None), Food):
                item = tile.item
                if item.can_chop and not item.chopped:
                    if self._move_towards(controller, bot_id, map_team, counter_pos):
                        controller.chop(bot_id, counter_pos[0], counter_pos[1])
                    return
                if item.can_cook and item.cooked_stage == 0 and (not item.can_chop or item.chopped):
                    if self._move_towards(controller, bot_id, map_team, counter_pos):
                        controller.pickup(bot_id, counter_pos[0], counter_pos[1])
                    return
                if (not item.can_chop or item.chopped) and ((item.can_cook and item.cooked_stage == 1) or (not item.can_cook and item.cooked_stage == 0)):
                    if self._move_towards(controller, bot_id, map_team, counter_pos):
                        controller.pickup(bot_id, counter_pos[0], counter_pos[1])
                    return

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

        ws = self.workstations[map_team]
        plate_counter = ws.get("plate_counter")
        submit_pos = ws.get("submit")
        shop_pos = ws.get("shop")
        sinktable_pos = ws.get("sinktable")
        sink_pos = ws.get("sink")
        missing = self._missing_for_plate(required, self._plate_at_counter_or_bot(controller, map_team))

        # If holding a dirty plate, put it in the sink first
        if holding and holding.get("type") == "Plate" and holding.get("dirty"):
            if sink_pos and self._move_towards(controller, bot_id, map_team, sink_pos):
                controller.put_dirty_plate_in_sink(bot_id, sink_pos[0], sink_pos[1])
            return

        # if plate counter is blocked, move to another empty counter
        if plate_counter:
            tile = controller.get_tile(map_team, plate_counter[0], plate_counter[1])
            if tile is not None and getattr(tile, "item", None) is not None and not isinstance(tile.item, Plate):
                new_counter = self._find_empty_counter_near(controller, map_team, ws.get("submit"))
                if new_counter is not None:
                    ws["plate_counter"] = new_counter
                    plate_counter = new_counter

        prep_counter_ref = ws.get("prep_counter")
        single_counter = prep_counter_ref and plate_counter and prep_counter_ref == plate_counter

        # if holding a plate, try submit when complete or add food from counters
        if holding and holding.get("type") == "Plate":
            missing = self._missing_for_plate(required, holding)
            if not missing and submit_pos:
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
                empty_counter = self._find_empty_counter_near(controller, map_team, ws.get("submit")) or plate_counter
                if empty_counter:
                    tile = controller.get_tile(map_team, empty_counter[0], empty_counter[1])
                    if tile is not None and getattr(tile, "item", None) is None:
                        if self._move_towards(controller, bot_id, map_team, empty_counter):
                            controller.place(bot_id, empty_counter[0], empty_counter[1])
                            if empty_counter != plate_counter:
                                ws["plate_counter"] = empty_counter
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

        # check plate on counter for completion (only if we're not already holding a plate)
        if not (holding and holding.get("type") == "Plate"):
            plate_obj = self._plate_at_counter_or_bot(controller, map_team)
            if plate_obj is not None:
                missing = self._missing_for_plate(required, plate_obj)
                if not missing:
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

        plate_obj = self._plate_at_counter_or_bot(controller, map_team)
        missing = self._missing_for_plate(required, plate_obj)

        ready = False
        for ft in missing:
            if self._find_ready_food_on_counters(controller, map_team, ft) is not None:
                ready = True
                break

        ws = self.workstations.get(map_team, {})
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
        prep_counter = self._find_empty_counter_near(controller, map_team, ws.get("prep_counter")) or ws.get("prep_counter")
        shop_pos = ws.get("shop")
        sink_pos = ws.get("sink")

        # If holding a dirty plate, put it in the sink
        if holding and holding.get("type") == "Plate" and holding.get("dirty"):
            if sink_pos and self._move_towards(controller, bot_id, map_team, sink_pos):
                controller.put_dirty_plate_in_sink(bot_id, sink_pos[0], sink_pos[1])
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
                    if self._move_towards(controller, bot_id, map_team, prep_counter):
                        controller.chop(bot_id, prep_counter[0], prep_counter[1])
                    return

        # buy missing choppable-only ingredient to speed prep (try box first)
        missing = self._missing_for_plate(required, self._plate_at_counter_or_bot(controller, map_team))
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
    def _plate_at_counter_or_bot(self, controller: RobotController, map_team: Team):
        ws = self.workstations[map_team]
        plate_counter = ws.get("plate_counter")
        if plate_counter:
            tile = controller.get_tile(map_team, plate_counter[0], plate_counter[1])
            if tile is not None and isinstance(getattr(tile, "item", None), Plate):
                return tile.item

        # fallback to any bot holding a plate
        for bid in controller.get_team_bot_ids(controller.get_team()):
            bstate = controller.get_bot_state(bid)
            holding = bstate.get("holding") if bstate else None
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
        lead = own_money - enemy_money

        turn = controller.get_turn()
        remaining_enemy_reward = 0
        for o in controller.get_orders(enemy_team):
            if o.get("completed_turn") is None and o.get("expires_turn", 0) >= turn:
                remaining_enemy_reward += o.get("reward", 0)

        if remaining_enemy_reward >= 40000:
            return True

        turns_left = info.get("window_end_turn", turn) - turn
        if turns_left <= 20:
            lead_threshold = max(-1000, int(0.2 * remaining_enemy_reward))
        else:
            lead_threshold = max(-2000, int(0.3 * remaining_enemy_reward))
        if remaining_enemy_reward >= 30000:
            lead_threshold -= 2000
        return lead >= lead_threshold
