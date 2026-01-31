from collections import deque, defaultdict
from typing import Tuple, Optional, List, Dict

from game_constants import Team, FoodType, ShopCosts
from robot_controller import RobotController
from item import Pan, Plate, Food
from tiles import Box

FOOD_BY_NAME = {ft.food_name: ft for ft in FoodType}

DIRECTIONS = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
]


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

    # ----------------------------
    # Map helpers
    # ----------------------------
    def _chebyshev(self, a: Tuple[int, int], b: Tuple[int, int]) -> int:
        return max(abs(a[0] - b[0]), abs(a[1] - b[1]))

    def _ensure_tile_cache(self, controller: RobotController, map_team: Team) -> None:
        if map_team in self.tile_cache:
            return

        positions: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
        for x in range(self.map_width):
            for y in range(self.map_height):
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
                if not (0 <= nx < self.map_width and 0 <= ny < self.map_height):
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

    def _move_towards(self, controller: RobotController, bot_id: int, map_team: Team, target: Tuple[int, int]) -> bool:
        """Move one step toward target. Returns True if bot is now adjacent (can act on target this turn)."""
        bot_state = controller.get_bot_state(bot_id)
        if bot_state is None:
            return False
        bx, by = bot_state["x"], bot_state["y"]
        if self._chebyshev((bx, by), target) <= 1:
            return True

        # Block tiles occupied by other bots on same team (same map) so we don't try to move into them
        blocked = set()
        for other_id in controller.get_team_bot_ids():
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
            # After moving, we may now be adjacent — allow same-turn action
            after = controller.get_bot_state(bot_id)
            if after and self._chebyshev((after["x"], after["y"]), target) <= 1:
                return True
        return False

    # ----------------------------
    # Order helpers
    # ----------------------------
    def _select_active_order(self, controller: RobotController) -> Optional[Dict]:
        orders = controller.get_orders()
        if not orders:
            return None

        turn = controller.get_turn()
        bot_ids = controller.get_team_bot_ids()
        active = [o for o in orders if o.get("is_active")]
        if not active:
            return None

        def score(o: Dict) -> float:
            time_left = max(1, o["expires_turn"] - turn)
            claimed_by = o.get("claimed_by")
            
            # Prefer unclaimed orders or orders claimed by our own bots
            if claimed_by is None:
                claim_bonus = 1000  # Strong preference for unclaimed
            elif claimed_by in bot_ids:
                claim_bonus = 500  # Also work on orders we're already working on
            else:
                claim_bonus = 0  # Avoid orders claimed by others (if any)
            
            # Prefer high reward, low penalty, and enough time to complete
            value = o["reward"] - 3 * o["penalty"]
            urgency = value / max(1, time_left)
            return urgency + (value * 0.01) + claim_bonus

        return max(active, key=score)

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

    # ----------------------------
    # Role assignment
    # ----------------------------
    def _assign_roles(self, bot_ids: List[int]) -> None:
        if self.bot_roles and all(bid in self.bot_roles for bid in bot_ids):
            return

        self.bot_roles = {}
        for i, bid in enumerate(sorted(bot_ids)):
            if i == 0:
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

        # If holding something, trash it first
        if holding:
            if trash_pos and self._move_towards(controller, bot_id, map_team, trash_pos):
                controller.trash(bot_id, trash_pos[0], trash_pos[1])
            return

        # Take cooked/ready food from pan and trash it
        cooker_pos = ws.get("cooker")
        if cooker_pos:
            tile = controller.get_tile(map_team, cooker_pos[0], cooker_pos[1])
            if tile is not None and isinstance(getattr(tile, "item", None), Pan):
                pan = tile.item
                if pan.food is not None:
                    if self._move_towards(controller, bot_id, map_team, cooker_pos):
                        controller.take_from_pan(bot_id, cooker_pos[0], cooker_pos[1])
                    return

        # Take food from counters and trash it
        food_pos = self._find_any_food_on_counters(controller, map_team)
        if food_pos:
            if self._move_towards(controller, bot_id, map_team, food_pos):
                controller.pickup(bot_id, food_pos[0], food_pos[1])
            return

        # Take clean plates from sink table and trash them (deprive enemy of plates)
        sinktable_pos = ws.get("sinktable")
        if sinktable_pos:
            tile = controller.get_tile(map_team, sinktable_pos[0], sinktable_pos[1])
            if tile is not None and getattr(tile, "num_clean_plates", 0) > 0:
                if self._move_towards(controller, bot_id, map_team, sinktable_pos):
                    controller.take_clean_plate(bot_id, sinktable_pos[0], sinktable_pos[1])
                return

        # Take plate with food from counters and trash it (high priority - these are completed orders)
        counters = self.tile_cache.get(map_team, {}).get("COUNTER", [])
        for pos in counters:
            tile = controller.get_tile(map_team, pos[0], pos[1])
            if tile is None:
                continue
            item = getattr(tile, "item", None)
            if isinstance(item, Plate) and not item.dirty and item.food:
                if self._move_towards(controller, bot_id, map_team, pos):
                    controller.pickup(bot_id, pos[0], pos[1])
                return

        # Take raw ingredients from boxes to slow enemy production
        boxes = self.tile_cache.get(map_team, {}).get("BOX", [])
        for pos in boxes:
            tile = controller.get_tile(map_team, pos[0], pos[1])
            if tile is None or not isinstance(tile, Box):
                continue
            if getattr(tile, "count", 0) > 0:
                if self._move_towards(controller, bot_id, map_team, pos):
                    controller.pickup(bot_id, pos[0], pos[1])
                return

        # Otherwise move toward submit to obstruct enemy submissions
        submit_pos = ws.get("submit")
        if submit_pos:
            self._move_towards(controller, bot_id, map_team, submit_pos)

    # ----------------------------
    # Main turn
    # ----------------------------
    def play_turn(self, controller: RobotController):
        bot_ids = controller.get_team_bot_ids()
        if not bot_ids:
            return

        self._assign_roles(bot_ids)

        first_state = controller.get_bot_state(bot_ids[0])
        if first_state is None:
            return
        map_team = Team[first_state["map_team"]]
        self._ensure_tile_cache(controller, map_team)

        # Strategic switch to enemy map for sabotage
        # Switch when: no active orders, OR we have good money lead, OR orders are about to expire
        switch_info = controller.get_switch_info()
        order = self._select_active_order(controller)
        turn = controller.get_turn()
        can_switch = switch_info.get("window_active", False) and not switch_info.get("my_team_switched", True)
        
        # Check if we should switch strategically
        should_switch = False
        if can_switch and not self.has_switched:
            if order is None:
                # No active orders - good time to sabotage
                should_switch = True
            elif order:
                # Check if order is about to expire soon (within 20 turns)
                time_left = order.get("expires_turn", turn + 100) - turn
                if time_left < 20 and order.get("claimed_by") in controller.get_team_bot_ids():
                    # Order we're working on is about to expire - switch to sabotage enemy
                    should_switch = True
        
        if should_switch and controller.can_switch_maps():
            if controller.switch_maps():
                self.has_switched = True
                return

        # if on enemy map, sabotage
        if map_team != controller.get_team():
            for bid in bot_ids:
                self._play_sabotage(controller, bid, map_team)
            return

        if order is None:
            return

        required = self._required_food_types(order)

        for bid in bot_ids:
            role = self.bot_roles.get(bid, "support")
            if role == "cook":
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

        ws = self.workstations[map_team]
        prep_counter = self._find_empty_counter_near(controller, map_team, ws.get("prep_counter")) or ws.get("prep_counter")
        plate_counter = ws.get("plate_counter")
        cooker_pos = ws.get("cooker")
        shop_pos = ws.get("shop")
        trash_pos = ws.get("trash")
        sink_pos = ws.get("sink")

        # If holding a dirty plate (e.g. picked up by mistake), put it in the sink
        if holding and holding.get("type") == "Plate" and holding.get("dirty"):
            if sink_pos and self._move_towards(controller, bot_id, map_team, sink_pos):
                controller.put_dirty_plate_in_sink(bot_id, sink_pos[0], sink_pos[1])
            return

        if cooker_pos:
            tile = controller.get_tile(map_team, cooker_pos[0], cooker_pos[1])
            if tile is not None and not isinstance(getattr(tile, "item", None), Pan):
                if shop_pos and self._move_towards(controller, bot_id, map_team, shop_pos):
                    if controller.get_team_money() >= ShopCosts.PAN.buy_cost:
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
        missing = self._missing_for_plate(required, self._plate_at_counter_or_bot(controller, map_team))
        missing_cookables = [ft for ft in missing if ft.can_cook or ft.can_chop]

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
                if controller.get_team_money() >= ft.buy_cost:
                    controller.buy(bot_id, ft, shop_pos[0], shop_pos[1])
            return

        missing_simple = [ft for ft in missing if not ft.can_cook and not ft.can_chop]
        for ft in list(missing_simple):
            if self._find_ready_food_on_counters(controller, map_team, ft) is not None:
                missing_simple.remove(ft)

        if missing_simple:
            ft = missing_simple[0]
            box_pos = self._find_box_with_food(controller, map_team, ft, ws.get("plate_counter"))
            if box_pos and self._move_towards(controller, bot_id, map_team, box_pos):
                controller.pickup(bot_id, box_pos[0], box_pos[1])
                return
            if shop_pos and self._move_towards(controller, bot_id, map_team, shop_pos):
                if controller.get_team_money() >= ft.buy_cost:
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

        # ensure we have a plate (in hand or on counter); on single-counter maps keep plate in hand
        prep_counter_ref = ws.get("prep_counter")
        single_counter = prep_counter_ref and plate_counter and prep_counter_ref == plate_counter
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
                    if shop_pos and self._move_towards(controller, bot_id, map_team, shop_pos):
                        if controller.get_team_money() >= ShopCosts.PLATE.buy_cost:
                            controller.buy(bot_id, ShopCosts.PLATE, shop_pos[0], shop_pos[1])
                    return

        # if holding a plate, try submit when complete or add food from counters
        if holding and holding.get("type") == "Plate":
            missing = self._missing_for_plate(required, holding)
            if not missing and submit_pos:
                # Move towards submit (returns True when adjacent or on it)
                # Can submit from adjacent tiles (Chebyshev distance <= 1)
                if self._move_towards(controller, bot_id, map_team, submit_pos):
                    controller.submit(bot_id, submit_pos[0], submit_pos[1])
                return
            # add food from counters (plate in hand) — works when only one counter
            if missing and not holding.get("dirty"):
                for ft in missing:
                    pos = self._find_ready_food_on_counters(controller, map_team, ft)
                    if pos and self._move_towards(controller, bot_id, map_team, pos):
                        controller.add_food_to_plate(bot_id, pos[0], pos[1])
                        return
            # put plate back on counter only when counter is empty (for multi-counter maps)
            # On single-counter maps, keep plate in hand so cook bot can use counter
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
                        if controller.get_team_money() >= ft.buy_cost:
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
                    if controller.get_team_money() >= ft.buy_cost:
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
        for bid in controller.get_team_bot_ids():
            bstate = controller.get_bot_state(bid)
            holding = bstate.get("holding") if bstate else None
            if holding and holding.get("type") == "Plate":
                return holding
        return None
