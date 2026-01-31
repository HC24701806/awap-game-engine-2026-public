"""
Game state that keeps track of teams, bots, maps, team money, bot items, and orders.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

from game_constants import Team, TileType, FoodType, GameConstants
from map import Map
from tiles import Tile, Floor, Wall, Counter, Sink, SinkTable, Cooker, Trash, Submit, Shop, Box
from item import Item, Food, Plate, Pan


# -----------------------
# Exceptions
# -----------------------

class GameStateException(Exception):
    pass


# -----------------------
# Orders
# -----------------------

@dataclass
class Order:
    '''Order class that is based on order type in game constants'''
    order_id: int
    required: List[FoodType]
    created_turn: int
    expires_turn: int
    reward: int
    penalty: int
    claimed_by: Optional[int] = None
    completed_turn: Optional[int] = None
    penalized: bool = False 

    def is_expired(self, turn: int) -> bool:
        return turn > self.expires_turn

    def is_active(self, turn: int) -> bool:
        return self.created_turn <= turn <= self.expires_turn and self.completed_turn is None


def plate_food_signature(plate: Plate) -> List[Tuple[int, bool, int]]:
    '''Helper that basically creates a unique signature for each user plated food'''

    sig: List[Tuple[int, bool, int]] = [] #unique plate signature exists as a list of tuples

    for f in plate.food:
        if isinstance(f, Food):
            sig.append((f.food_id, bool(getattr(f, "chopped", False)), int(getattr(f, "cooked_stage", 0))))
        elif isinstance(f, FoodType):
            sig.append((f.food_id, False, 0))
        else:
            sig.append((-1, False, 0))

    sig.sort()
    return sig


def order_signature(req: List[FoodType]) -> List[Tuple[int, bool, int]]:
    '''Helper that creates the order required plate signature'''
    
    sig = [(ft.food_id, ft.can_chop, 1 if ft.can_cook else 0) for ft in req] #basically force chopped and cooked if the food can
    sig.sort()
    return sig


def plate_matches_order(plate: Plate, order: Order) -> bool:
    '''Sees if the plate matches the order'''
    return plate_food_signature(plate) == order_signature(order.required)


# -----------------------
# Bots
# -----------------------

@dataclass
class BotState:
    '''For each bot, they have their bot state to keep track of'''
    bot_id: int
    team: Team #original non-switched team
    x: int
    y: int
    holding: Optional[Item] = None #can only hold 1 item at a time

    map_team: Team = Team.RED  #add_bot() will set this correctly, default is RED for now

    def pos(self) -> Tuple[int, int]:
        '''Helper that gets their position'''
        return (self.x, self.y)


# -----------------------
# Tile factory and map normalization
# -----------------------

def tile_factory(tile_type: TileType) -> Tile:
    '''Converst tile type to a tile class'''
    if tile_type == TileType.FLOOR:
        return Floor()
    if tile_type == TileType.WALL:
        return Wall()
    if tile_type == TileType.COUNTER:
        return Counter()
    if tile_type == TileType.SINK:
        return Sink()
    if tile_type == TileType.SINKTABLE:
        return SinkTable()
    if tile_type == TileType.COOKER:
        return Cooker()
    if tile_type == TileType.TRASH:
        return Trash()
    if tile_type == TileType.SUBMIT:
        return Submit()
    if tile_type == TileType.SHOP:
        return Shop()
    return Tile(tile_type)


def normalize_map_tiles(m: Map) -> None:
    '''It converts map tiles from tile type to actual tiles that are interactable IF NEEDED (at the beginning especially)'''
    if m.tiles is None:
        m.tiles = [[tile_factory(TileType.FLOOR) for _ in range(m.height)] for _ in range(m.width)]
        return

    sample = m.tiles[0][0] #assume tiles is either all tile type or tiles
    if isinstance(sample, TileType):
        m.tiles = [[tile_factory(cell) for cell in col] for col in m.tiles]  # m.tiles is [x][y]

    #do nothing
    elif isinstance(sample, Tile):
        return
    
    #error 
    else:
        raise GameStateException(f"cannot recognize map tile type: {type(sample)}")


# -----------------------
# GameState
# -----------------------

class GameState:
    '''Game state class that keeps track of the state at each turn'''
    def __init__(self, red_map: Map, blue_map: Map):
        self.red_map = red_map
        self.blue_map = blue_map

        self.turn = 0
        self.bots: Dict[int, BotState] = {}
        
        #shared team money
        self.team_money: Dict[Team, int] = {Team.RED: 150, Team.BLUE: 150}
        
        #each team has its own independent order list
        #this is filled in in game.py after processing the map
        self.orders: Dict[Team, List[Order]] = {Team.RED: [], Team.BLUE: []}
        
        self.next_order_id = 1

        #switching states
        self.switch_turn = GameConstants.MIDGAME_SWITCH_TURN
        self.switch_duration = GameConstants.MIDGAME_SWITCH_DURATION
        self.switched = {Team.RED: False, Team.BLUE: False}

        #init map tiles
        normalize_map_tiles(self.red_map)
        normalize_map_tiles(self.blue_map)

        #occ maps
        self.occupancy = {
            Team.RED: [[None for _ in range(self.red_map.height)] for _ in range(self.red_map.width)],
            Team.BLUE: [[None for _ in range(self.blue_map.height)] for _ in range(self.blue_map.width)],
        }


    # -------------
    # Map helpers
    # --------------

    def get_map(self, team: Team) -> Map:
        return self.red_map if team == Team.RED else self.blue_map

    def get_tile(self, team: Team, x: int, y: int) -> Tile:
        m = self.get_map(team)
        if not m.in_bounds(x, y):
            raise GameStateException(f"out of bounds error: ({x},{y}) for team {team.name}")
        return m.tiles[x][y]

    def is_walkable(self, team: Team, x: int, y: int) -> bool:
        '''helper for movement'''
        t = self.get_tile(team, x, y)
        return bool(getattr(t, "is_walkable", False)) #we will use getattr because it has a default functionality

    # -------------
    # Money helpers
    # -------------

    def get_team_money(self, team: Team) -> int:
        return self.team_money.get(team, 0)

    def add_team_money(self, team: Team, delta: int) -> None:
        self.team_money[team] = self.team_money.get(team, 0) + delta

    # -------------
    # Bot creation
    # -------------

    def add_bot(self, team: Team, x: int, y: int, bot_id: Optional[int] = None) -> int:
        '''Add a bot on (x, y) and return the bot id'''
        if not self.is_walkable(team, x, y):
            raise GameStateException(f"can't place bot on non walkable tile at ({x},{y})")

        occ = self.occupancy[team][x][y]
        if occ is not None:
            raise GameStateException(f"tile ({x},{y}) already occupied by bot {occ}")

        #just make a new bot with new id (just error)
        if bot_id is None:
            bot_id = 0 if len(self.bots) == 0 else (max(self.bots.keys()) + 1)

        #start off at the beginning with current map team
        self.bots[bot_id] = BotState(bot_id=bot_id, team=team, x=x, y=y, holding=None, map_team=team)
        self.occupancy[team][x][y] = bot_id
        return bot_id

    def get_bot(self, bot_id: int) -> BotState:
        '''Get the bot class'''
        if bot_id not in self.bots:
            raise GameStateException(f"Invalid bot_id: {bot_id}")
        return self.bots[bot_id]

    # -------------
    # Turn mechanics
    # -------------

    def start_turn(self) -> None:
        '''Run this at the start of each turn for environmental and passive'''
        self.turn += 1
        
        #passive money
        self.add_team_money(Team.RED, GameConstants.MONEY_PER_TURN)
        self.add_team_money(Team.BLUE, GameConstants.MONEY_PER_TURN)

        #add envirnomental ticks (ie cooks) that do not require player action
        self.tick_environment(Team.RED)
        self.tick_environment(Team.BLUE)

        #order logic
        self.expire_orders()

        #switch back when the time period ends
        if self.switch_window_ended(): #do this everytime in case of error
            self.return_team_home_if_switched(Team.RED)
            self.return_team_home_if_switched(Team.BLUE)

    def add_clean_plate_to_sinktable_near(self, team: Team, x: int, y: int) -> None:
        '''helper to put already washed dishes in the sink table automatically'''
        m = self.get_map(team)

        #first, we check for sinktable near us in the common case
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nx, ny = x + dx, y + dy
            if not m.in_bounds(nx, ny):
                continue
            t = m.tiles[nx][ny]
            if isinstance(t, SinkTable):
                t.num_clean_plates += 1
                return

        #if there is no sink table near us in the common cas , we put the clean plates in the first sink table we see location
        for ix in range(m.width):
            for iy in range(m.height):
                t = m.tiles[ix][iy]
                if isinstance(t, SinkTable):
                    t.num_clean_plates += 1
                    return

    def tick_environment(self, team: Team) -> None:
        '''cooking ticks helper that basically cooks if pan is in the food or wash if the dishes are washing'''
        m = self.get_map(team)

        for x in range(m.width):
            for y in range(m.height):

                #get the tile
                tile = m.tiles[x][y]

                #if the tile is a cooker, then we auto cook it through ticking
                if isinstance(tile, Cooker):
                    pan = tile.item
                    if isinstance(pan, Pan) and isinstance(pan.food, Food):
                        tile.cook_progress += 1
                        if tile.cook_progress == GameConstants.COOK_PROGRESS and pan.food.cooked_stage == 0:
                            pan.food.cooked_stage = 1
                        elif tile.cook_progress >= GameConstants.BURN_PROGRESS:
                            pan.food.cooked_stage = 2

                #if the tile is a sink, then if we are washing, then we clean it
                if isinstance(tile, Sink):

                    if tile.using and tile.num_dirty_plates > 0:
                        tile.curr_dirty_plate_progress += 1

                        if tile.curr_dirty_plate_progress >= GameConstants.PLATE_WASH_PROGRESS:
                            tile.curr_dirty_plate_progress = 0
                            tile.num_dirty_plates -= 1
                            self.add_clean_plate_to_sinktable_near(team, x, y)

                    # reset the tile each turn so the user needs ot keep washing
                    tile.using = False

    def expire_orders(self) -> None:
        '''
        If an order expires without being completed then penalize that TEAM.
        Keeps all orders in the history, only marks them as penalized.
        '''
        for team in [Team.RED, Team.BLUE]:
            
            # We iterate through the existing list without creating a filtered copy
            for o in self.orders.get(team, []):

                # Check if order is expired, not completed, and hasn't been penalized yet
                if o.completed_turn is None and o.is_expired(self.turn):
                    if not o.penalized:
                        self.add_team_money(team, -o.penalty)
                        o.penalized = True
            

    # -------------
    # Orders
    # -------------

    def spawn_order(self, required: List[FoodType], delta_time: int = 20, reward: int = 5, penalty: int = 2) -> int:
        '''
        creates an order for both teams
        returns the shared order_id but for both teams
        '''

        order_id = self.next_order_id
        self.next_order_id += 1

        def make_order() -> Order:
            '''helper'''
            return Order(
                order_id=order_id,
                required=required,
                created_turn=self.turn,
                expires_turn=self.turn + delta_time,
                reward=reward,
                penalty=penalty,
            )

        self.orders[Team.RED].append(make_order())
        self.orders[Team.BLUE].append(make_order())

        return order_id


    def add_dirty_plate_to_sink_near(self, team: Team, x: int, y: int) -> None:
        '''helper to add dirty plates'''
        m = self.get_map(team)

        # add to near sink in the common sink
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nx, ny = x + dx, y + dy
            if not m.in_bounds(nx, ny):
                continue
            t = m.tiles[nx][ny]
            if isinstance(t, Sink):
                t.num_dirty_plates += 1
                return

        # the first sink anywhere
        for ix in range(m.width):
            for iy in range(m.height):
                t = m.tiles[ix][iy]
                if isinstance(t, Sink):
                    t.num_dirty_plates += 1
                    return

    def submit_plate(self, bot_id: int, target_x: int, target_y: int) -> bool:
        '''logic to submit the plate, will go to MAP team not the team that submitted'''

        bot = self.get_bot(bot_id)
        tile = self.get_tile(bot.map_team, target_x, target_y)

        if not isinstance(tile, Submit):
            return False
        if not isinstance(bot.holding, Plate):
            return False

        order_team = bot.map_team #MAP OWNER, not the submission team
        for o in self.orders.get(order_team, []):
            if o.is_active(self.turn) and plate_matches_order(bot.holding, o):
                o.claimed_by = bot_id
                o.completed_turn = self.turn

                #reward map owner
                self.add_team_money(order_team, o.reward)

                #dirty plate goes into sink on that map specifically
                self.add_dirty_plate_to_sink_near(order_team, target_x, target_y)

                bot.holding = None #lets go of jitem
                return True

        return False


    # -----------------------
    # Movement (interact will be implemented in robot_controller)
    # -----------------------

    def move_bot(self, bot_id: int, dx: int, dy: int) -> bool:
        '''move bot with checks; needs to be wrt current MAP team (ie switched)'''

        bot = self.get_bot(bot_id)
        new_x, new_y = bot.x + dx, bot.y + dy
        m = self.get_map(bot.map_team)

        if not m.in_bounds(new_x, new_y):
            return False
        if not self.is_walkable(bot.map_team, new_x, new_y):
            return False
        if self.occupancy[bot.map_team][new_x][new_y] is not None:
            return False

        self.occupancy[bot.map_team][bot.x][bot.y] = None
        self.occupancy[bot.map_team][new_x][new_y] = bot_id

        bot.x, bot.y = new_x, new_y
        return True

    

    # -----------------------
    # Switch mechanics 
    # -----------------------

    def switch_window_active(self, turn: Optional[int] = None) -> bool:
        '''return true if we are in the switch window'''
        t = self.turn if turn is None else turn
        start = self.switch_turn
        end = self.switch_turn + self.switch_duration - 1
        return start <= t <= end

    def switch_window_ended(self, turn: Optional[int] = None) -> bool:
        '''return true if we are PAST the switch window'''
        t = self.turn if turn is None else turn
        end = self.switch_turn + self.switch_duration - 1
        return t > end

    def other_team(self, team: Team) -> Team:
        '''return the other team'''
        return Team.RED if team == Team.BLUE else Team.BLUE

    def is_walkable_on_map(self, map_team: Team, x: int, y: int) -> bool:
        '''map-based walkability dependent on input team'''
        t = self.get_tile(map_team, x, y)
        return bool(getattr(t, "is_walkable", False))

    def find_free_spawn_near(self, map_team: Team, prefer_x: int, prefer_y: int) -> Tuple[int, int]:
        '''
        find spawn point for the switch where the team specifies
        '''
        m = self.get_map(map_team)

        def can_spawn(x: int, y: int) -> bool:
            '''in bounds and not occupied and walkable'''
            if not m.in_bounds(x, y):
                return False
            if self.occupancy[map_team][x][y] is not None:
                return False
            return self.is_walkable_on_map(map_team, x, y)

        #look around for floor
        for r in range(max(m.width, m.height)):
            for dx in range(-r, r + 1):
                for dy in range(-r, r + 1):
                    x, y = prefer_x + dx, prefer_y + dy
                    if not can_spawn(x, y):
                        continue
                    if getattr(m.tiles[x][y], "tile_name", "") == "FLOOR":
                        return (x, y)

        #look around for walkable
        for r in range(max(m.width, m.height)):
            for dx in range(-r, r + 1):
                for dy in range(-r, r + 1):
                    x, y = prefer_x + dx, prefer_y + dy
                    if can_spawn(x, y):
                        return (x, y)

        #just scan for anything spawnable
        for x in range(m.width):
            for y in range(m.height):
                if can_spawn(x, y):
                    return (x, y)

        #worst case is (0, 0)
        return (0, 0)

    def request_switch(self, team: Team) -> bool:
        '''
        performs the actual switch, once per team
        '''

        #checks for switching
        if not self.switch_window_active():
            return False
        if self.switched.get(team, False):
            return False

        dest_map = self.other_team(team)

        #clear the occupancy first in previous map
        bot_ids = [bid for bid, b in self.bots.items() if b.team == team]
        for bid in bot_ids:
            b = self.bots[bid]
            self.occupancy[b.map_team][b.x][b.y] = None

        #place on destination map with no  collisions between ANY bots
        for bid in bot_ids:
            b = self.bots[bid]
            spawn_x, spawn_y = self.find_free_spawn_near(dest_map, b.x, b.y)
            b.map_team = dest_map
            b.x, b.y = spawn_x, spawn_y
            self.occupancy[dest_map][spawn_x][spawn_y] = bid

        #set state
        self.switched[team] = True
        return True

    def return_team_home_if_switched(self, team: Team) -> None:
        '''Set the reverse of switch for returning to home map'''

        if not self.switched.get(team, False):
            return

        bot_ids = [bid for bid, b in self.bots.items() if b.team == team]

        #clear current occupancy
        for bid in bot_ids:
            b = self.bots[bid]
            self.occupancy[b.map_team][b.x][b.y] = None

        #respawn on home map
        for bid in bot_ids:
            b = self.bots[bid]
            spawn_x, spawn_y = self.find_free_spawn_near(team, b.x, b.y)
            b.map_team = team
            b.x, b.y = spawn_x, spawn_y
            self.occupancy[team][spawn_x][spawn_y] = bid

        self.switched[team] = False


    # -----------------------
    # Serialization
    # -----------------------

    def to_dict(self) -> Dict[str, Any]:
        def item_to_dict(it: Optional[Item]) -> Any:
            if it is None:
                return None
            if isinstance(it, Food):
                return {
                    "type": "Food",
                    "food_name": it.food_name,
                    "food_id": it.food_id,
                    "chopped": it.chopped,
                    "cooked_stage": it.cooked_stage,
                }
            if isinstance(it, Plate):
                return {
                    "type": "Plate",
                    "dirty": it.dirty,
                    "food": [item_to_dict(f if isinstance(f, Food) else Food(f)) for f in it.food],
                }
            if isinstance(it, Pan):
                return {"type": "Pan", "food": item_to_dict(it.food)}
            return {"type": type(it).__name__}

        bots_payload = [
            {
                "bot_id": bot_id,
                "team": b.team.name,
                "x": b.x,
                "y": b.y,
                "holding": item_to_dict(b.holding),
                "map_team": getattr(b, "map_team", b.team).name,
            }
            for bot_id, b in self.bots.items()
        ]

        orders_payload = {
            Team.RED.name: [
                {
                    "order_id": o.order_id,
                    "required": [ft.food_name for ft in o.required],
                    "created_turn": o.created_turn,
                    "expires_turn": o.expires_turn,
                    "reward": o.reward,
                    "penalty": o.penalty,
                    "claimed_by": o.claimed_by,
                    "completed_turn": o.completed_turn,
                }
                for o in self.orders.get(Team.RED, [])
            ],
            Team.BLUE.name: [
                {
                    "order_id": o.order_id,
                    "required": [ft.food_name for ft in o.required],
                    "created_turn": o.created_turn,
                    "expires_turn": o.expires_turn,
                    "reward": o.reward,
                    "penalty": o.penalty,
                    "claimed_by": o.claimed_by,
                    "completed_turn": o.completed_turn,
                }
                for o in self.orders.get(Team.BLUE, [])
            ],
        }


        return {
            "turn": self.turn,
            "team_money": {Team.RED.name: self.get_team_money(Team.RED), Team.BLUE.name: self.get_team_money(Team.BLUE)},
            "bots": bots_payload,
            "orders": orders_payload,
            "red_map": self.red_map.to_2d_list(),
            "blue_map": self.blue_map.to_2d_list(),
        }