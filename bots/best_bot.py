from collections import defaultdict
from typing import Tuple, Optional, List, Dict

import os
import sys

from game_constants import Team, FoodType
from item import Food, Pan

sys.path.append(os.path.dirname(__file__))
from wilson_bot import BotPlayer as WilsonBot


class BotPlayer(WilsonBot):
    """Hybrid bot: Wilson core + Alex order scoring + Ethan switch/sabotage triggers."""

    def __init__(self, map_copy):
        super().__init__(map_copy)
        self.disable_sabotage = False
        self.enable_switch = True
        self.last_order_id: Optional[int] = None

    def play_turn(self, controller):
        self._reset_turn_state(controller.get_turn())
        bot_ids = controller.get_team_bot_ids(controller.get_team())
        if not bot_ids:
            return

        self._assign_roles(bot_ids)

        first_state = controller.get_bot_state(bot_ids[0])
        if first_state is None:
            return
        map_team = Team[first_state["map_team"]]
        self._ensure_tile_cache(controller, map_team)

        if self.enable_switch and controller.can_switch_maps():
            if self._should_switch_for_sabotage(controller):
                controller.switch_maps()
                first_state = controller.get_bot_state(bot_ids[0])
                if first_state is None:
                    return
                map_team = Team[first_state["map_team"]]
                self._ensure_tile_cache(controller, map_team)

        order = self._select_active_order(controller)
        if not self.disable_sabotage and map_team != controller.get_team():
            for bid in bot_ids:
                self._play_sabotage(controller, bid, map_team)
            return

        if order is None:
            return

        required = self._required_food_types(order)

        for bid in bot_ids:
            role = self.bot_roles.get(bid, "support")
            if role == "solo":
                self._play_solo(controller, bid, map_team, required)
            elif role == "cook":
                self._play_cook(controller, bid, map_team, required)
            elif role == "plate":
                self._play_plate(controller, bid, map_team, required)
            else:
                self._play_support(controller, bid, map_team, required)

    def _select_active_order(self, controller) -> Optional[Dict]:
        orders = controller.get_orders(controller.get_team())
        if not orders:
            return None

        turn = controller.get_turn()
        bot_ids = controller.get_team_bot_ids(controller.get_team())
        active = [o for o in orders if o.get("is_active")]
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
            shop_overhead = 6 if need_shop else 0

            return cook_time + chop_time + simple_time + shop_overhead + 2

        def missing_after_ready(o: Dict) -> List[FoodType]:
            req = self._required_food_types(o)
            counts = ready_counts(map_team)
            missing: List[FoodType] = []
            for ft in req:
                if counts.get(ft.food_id, 0) > 0:
                    counts[ft.food_id] -= 1
                else:
                    missing.append(ft)
            return missing

        def wilson_key(o: Dict) -> Tuple[int, int, int, float, float, float, float]:
            req = self._required_food_types(o)
            missing = missing_after_ready(o)
            time_left = max(1, o["expires_turn"] - turn)
            est = estimate_time(o)
            value = o["reward"] + (0.5 * o["penalty"])

            shop_missing = 0
            total_cost = 0
            for ft in missing:
                if not self._box_has_food(controller, map_team, ft):
                    shop_missing += 1
                    total_cost += ft.buy_cost

            if plate_has_food and self._plate_exact_match(plate_obj, req):
                plate_priority = 0
            elif plate_has_food and self._plate_subset_of_order(plate_obj, req):
                plate_priority = 1
            else:
                plate_priority = 2

            return (plate_priority, shop_missing, total_cost, time_left, len(missing), est, -value)

        def alex_score(o: Dict) -> float:
            time_left = max(1, o["expires_turn"] - turn)
            req = self._required_food_types(o)
            est = estimate_time(o)
            feasible = est <= (time_left - 2)

            claimed_by = o.get("claimed_by")
            if claimed_by is None:
                claim_bonus = 1000
            elif claimed_by in bot_ids:
                claim_bonus = 500
            else:
                claim_bonus = 0

            value = o["reward"] + (0.5 * o["penalty"])
            base = value / max(1.0, est)
            if not feasible:
                base *= 0.1

            if plate_has_food:
                if self._plate_exact_match(plate_obj, req):
                    base += 5000
                elif self._plate_subset_of_order(plate_obj, req):
                    base += 2000
                else:
                    base -= 2000

            if o.get("order_id") == self.last_order_id:
                base += 250

            return base + claim_bonus

        best = min(active, key=lambda o: (wilson_key(o), -alex_score(o)))
        self.last_order_id = best.get("order_id")
        return best

    def _should_switch_for_sabotage(self, controller) -> bool:
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
