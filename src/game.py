# game.py

'''python src/game.py --red bots/sample_bot.py --blue bots/sample_bot.py --map maps/tiny_map.txt --render'''

import argparse
import copy
import importlib.util
import json
import os
import sys
import time
import traceback
from threading import Thread
from typing import Optional, Any, Dict, List, Tuple

from game_constants import Team, GameConstants
from game_state import GameState
from robot_controller import RobotController

from map_processor import load_two_team_maps_and_orders
from render import Renderer


def import_file(module_name: str, file_path: str):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {file_path}")
    module = importlib.util.module_from_spec(spec)

    sys.modules[module_name] = module

    spec.loader.exec_module(module)
    return module



def find_default_floor_spawn(m, prefer_center=True) -> Tuple[int, int]:
    '''if map has no red, blue spawn markers, find the centermost walkable spawn'''
    if prefer_center:
        cx, cy = m.width // 2, m.height // 2
        for r in range(min(m.width, m.height)):
            for dx in range(-r, r + 1):
                for dy in range(-r, r + 1):
                    x, y = cx + dx, cy + dy
                    if m.in_bounds(x, y) and getattr(m.tiles[x][y], "is_walkable", False):
                        return (x, y)
    for y in range(m.height):
        for x in range(m.width):
            if getattr(m.tiles[x][y], "is_walkable", False):
                return (x, y)
    return (0, 0)


class Game:
    def __init__(
        self,
        red_bot_path: str,
        blue_bot_path: str,
        map_path: str,
        replay_path: Optional[str] = None,
        render: bool = False,
        turn_limit: int = GameConstants.TOTAL_TURNS,
        per_turn_timeout_s: float = 10.0,
        fps_cap: int = 30,
    ):
        self.render_enabled = render
        self.turn_limit = turn_limit
        self.per_turn_timeout_s = per_turn_timeout_s
        self.fps_cap = fps_cap

        self.replay_path = replay_path
        if replay_path is not None:
            os.makedirs(os.path.dirname(replay_path) or ".", exist_ok=True)

        #load the maps
        map_red, map_blue, orders_red, orders_blue, parsed = load_two_team_maps_and_orders(map_path)

        #create game state
        self.game_state = GameState(red_map=map_red, blue_map=map_blue)


        #get midgame switch window from map
        self.game_state.switch_turn = getattr(parsed, "switch_turn", GameConstants.MIDGAME_SWITCH_TURN)
        self.game_state.switch_duration = getattr(parsed, "switch_duration", GameConstants.MIDGAME_SWITCH_DURATION)


        #load orders into the game state
        self.game_state.orders[Team.RED] = orders_red
        self.game_state.orders[Team.BLUE] = orders_blue

        #make next_order_id to avoid collisions if spawn_order() is useed later
        max_id = 0
        for o in orders_red:
            max_id = max(max_id, o.order_id)
        self.game_state.next_order_id = max_id + 1

        #import bots, need the play turn mechanic
        self.red_failed_init = False
        self.blue_failed_init = False

        #try to import
        try:
            red_name = os.path.basename(red_bot_path).rsplit(".", 1)[0]
            self.red_player = import_file(red_name, red_bot_path).BotPlayer(copy.deepcopy(self.game_state.red_map))
        except Exception as e:
            self.red_failed_init = True
            print(f"[INIT] Red bot failed: {e}")
            traceback.print_exc()

        try:
            blue_name = os.path.basename(blue_bot_path).rsplit(".", 1)[0]
            self.blue_player = import_file(blue_name, blue_bot_path).BotPlayer(copy.deepcopy(self.game_state.blue_map))
        except Exception as e:
            self.blue_failed_init = True
            print(f"[INIT] Blue bot failed: {e}")
            traceback.print_exc()

        #generate the controllers
        self.red_controller = RobotController(Team.RED, self.game_state)
        self.blue_controller = RobotController(Team.BLUE, self.game_state)

        #put the bots in the parsed map
        if parsed.spawns_red:
            for (x, y) in parsed.spawns_red:
                self.game_state.add_bot(Team.RED, x, y)
        else:
            x, y = find_default_floor_spawn(self.game_state.red_map)
            self.game_state.add_bot(Team.RED, x, y)

        if parsed.spawns_blue:
            for (x, y) in parsed.spawns_blue:
                self.game_state.add_bot(Team.BLUE, x, y)
        else:
            x, y = find_default_floor_spawn(self.game_state.blue_map)
            self.game_state.add_bot(Team.BLUE, x, y)

        #replay
        self.replay: List[Dict[str, Any]] = []

        #renderer if available
        self.renderer = Renderer(self.game_state) if self.render_enabled else None

    def call_player(self, team: Team) -> bool:
        '''calls the player run code'''
        if team == Team.RED:
            if self.red_failed_init:
                return False
            player = self.red_player
            controller = self.red_controller
        else:
            if self.blue_failed_init:
                return False
            player = self.blue_player
            controller = self.blue_controller

        ok = True
        exc: Optional[BaseException] = None

        def runner():
            nonlocal ok, exc
            #try it
            try:
                player.play_turn(controller)
            except BaseException as e:
                ok = False
                exc = e

        t0 = time.time()
        th = Thread(target=runner, daemon=True) #run in a separate thread
        th.start()
        th.join(self.per_turn_timeout_s)
        dt = time.time() - t0

        if th.is_alive():
            print(f"[TURN RUNNER] {team.name} timed out ({dt:.3f}s > {self.per_turn_timeout_s:.3f}s)")
            return False
        if not ok:
            print(f"[TURN REUNNER] {team.name} crashed: {exc}")
            traceback.print_exc()
            return False
        return True

    def record_turn(self):
        self.replay.append(self.game_state.to_dict()) #for the replay rile

    def render(self) -> bool:
        '''render ONLY IF we want to render'''
        if not self.render_enabled or self.renderer is None:
            return True
        return self.renderer.render_once(fps_cap=self.fps_cap)

    def run_game(self) -> Optional[Team]:
        '''run the game and return a winner'''

        #needs init
        if self.red_failed_init and self.blue_failed_init:
            print("[GAME] Both bots failed to initialize.")
            return None

        #render init
        if not self.render():
            return None

        for _ in range(self.turn_limit):
            #start turn (money + environment + expirations)
            self.game_state.start_turn()

            #call blue then red
            blue_ok = self.call_player(Team.BLUE)
            red_ok = self.call_player(Team.RED)

            #record and render
            self.record_turn()
            if not self.render():
                break

            #if one side crashes, then the other side wins by default
            if not blue_ok and red_ok:
                print("[GAME] BLUE failed, RED wins")
                winner = Team.RED
                self.export_replay(winner)
                return winner
            if not red_ok and blue_ok:
                print("[GAME] RED failed, BLUE wins")
                winner = Team.BLUE
                self.export_replay(winner)
                return winner
            if not red_ok and not blue_ok:
                print("[GAME] Both failed, no winner")
                self.export_replay(None)
                return None

        red_money = self.game_state.get_team_money(Team.RED)
        blue_money = self.game_state.get_team_money(Team.BLUE)
        
        print(f"[GAME OVER] money scores: RED=${red_money}, BLUE=${blue_money}")

        if red_money > blue_money:
            print(f"[RESULT] RED WINS by ${red_money - blue_money}!")
            winner = Team.RED
        elif blue_money > red_money:
            print(f"[RESULT] BLUE WINS by ${blue_money - red_money}!")
            winner = Team.BLUE
        else:
            print("[RESULT] DRAW")
            winner = None

        self.export_replay(winner)
        return None

    def export_replay(self, winner: Optional[Team]):
        '''json dump'''
        if self.replay_path is None:
            return
        payload = {
            "winner": None if winner is None else winner.name,
            "turns": len(self.replay),
            "switch_turn_start": self.game_state.switch_turn,
            "switch_turn_end": self.game_state.switch_turn + self.game_state.switch_duration, 
            "replay": self.replay,
        }
        with open(self.replay_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"[REPLAY] wrote {self.replay_path}")

    def close(self):
        if self.renderer is not None:
            self.renderer.close()


def main():
    '''parse and run'''
    ap = argparse.ArgumentParser()
    ap.add_argument("--red", required=True, help="path to red bot python file (defines BotPlayer)")
    ap.add_argument("--blue", required=True, help="path to blue bot python file (defines BotPlayer)")
    ap.add_argument("--map", required=True, help="path to map text file (layout + optional ORDERS:)")
    ap.add_argument("--replay", default=None, help="optional output replay json path")
    ap.add_argument("--render", action="store_true", help="enable pygame rendering")
    ap.add_argument("--turns", type=int, default=GameConstants.TOTAL_TURNS, help="turn limit")
    ap.add_argument("--timeout", type=float, default=10.0, help="per-turn timeout seconds per bot")
    ap.add_argument("--fps", type=int, default=30, help="fps cap when rendering")
    args = ap.parse_args()

    g = Game(
        red_bot_path=args.red,
        blue_bot_path=args.blue,
        map_path=args.map,
        replay_path=args.replay,
        render=args.render,
        turn_limit=args.turns,
        per_turn_timeout_s=args.timeout,
        fps_cap=args.fps,
    )
    try:
        g.run_game()
    finally:
        g.close()


if __name__ == "__main__":
    main()
