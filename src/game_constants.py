'''game_constants.py'''

from enum import Enum
from dataclasses import dataclass


'''
----------------------
General Game Constants
----------------------
'''

class Team(Enum):
  RED = 0
  BLUE = 1

class TileType(Enum):
  '''Tile contains all the tile types of the map'''
  def __init__(self, tile_name: str, tile_id: int, is_walkable: bool, is_dangerous: bool, is_placeable: bool, is_interactable: bool):
    self.__tile_name = tile_name
    self.__tile_id = tile_id
    self.__is_walkable = is_walkable
    self.__is_dangerous = is_dangerous
    self.__is_placeable = is_placeable
    self.__is_interactable = is_interactable

  @property
  def tile_name(self): return self.__tile_name
  @property
  def tile_id(self): return self.__tile_id
  @property
  def is_walkable(self): return self.__is_walkable
  @property
  def is_dangerous(self): return self.__is_dangerous
  @property
  def is_placeable(self): return self.__is_placeable
  @property
  def is_interactable(self): return self.__is_interactable


  FLOOR = ("FLOOR", 0, True, False, False, False)
  WALL = ("WALL", 1, False, False, False, False)
  COUNTER = ("COUNTER", 2, False, False, True, False)
  BOX = ("BOX", 3, False, False, True, True)
  SINK = ("SINK", 4, False, False, True, True)
  SINKTABLE = ("SINKTABLE", 5, False, False, False, True)
  COOKER = ("COOKER", 6, False, False, True, True)
  TRASH = ("TRASH", 9, False, False, True, False)
  SUBMIT = ("SUBMIT", 10, True, False, True, True)
  SHOP = ("SHOP", 11, False, False, False, True)


class FoodType(Enum):
  def __init__(self, food_name:str, food_id: int, can_chop: bool, can_cook: bool, buy_cost: int):
    self.__food_name = food_name
    self.__food_id = food_id
    self.__can_chop = can_chop
    self.__can_cook = can_cook
    self.__buy_cost = buy_cost

  @property
  def food_name(self): return self.__food_name
  @property
  def food_id(self): return self.__food_id
  @property
  def can_chop(self): return self.__can_chop
  @property
  def can_cook(self): return self.__can_cook
  @property
  def buy_cost(self): return self.__buy_cost

  EGG = ("EGG", 0, False, True, 20)
  ONIONS = ("ONIONS", 1, True, False, 30)
  MEAT = ("MEAT", 2, True, True, 80)
  NOODLES = ("NOODLES", 3, False, False, 40)
  SAUCE = ("SAUCE", 4, False, False, 10)


class ShopCosts(Enum):
  def __init__(self, item_name: str, buy_cost: int):
    self.__item_name = item_name
    self.__buy_cost = buy_cost

  @property
  def item_name(self): return self.__item_name
  @property
  def buy_cost(self): return self.__buy_cost

  PLATE = ("PLATE", 2)
  PAN = ("PAN", 4)


class FrozenMeta(type):
  '''cannot edit game constants check'''
  def __setattr__(cls, name, value):
    raise AttributeError(f"Cannot edit constants '{name}' in GameConstants")

class GameConstants(metaclass = FrozenMeta):
  TOTAL_TURNS = 500 #this is default without engine specification

  MONEY_PER_TURN = 1

  #time it takes to cook and burn; [0, 20) uncooked, [20, 40) cooked, [40,..) burnt
  COOK_PROGRESS = 20
  BURN_PROGRESS = 40

  PLATE_WASH_PROGRESS = 2

  #WARNING: this should be specified in the map, but if not, default are these:
  MIDGAME_SWITCH_TURN = 250
  MIDGAME_SWITCH_DURATION = 100