import numpy as np
from whimbox.map.convert import *

# 注意：本模块目前没有任何外部引用（全仓 0 处 import）。
# 两个转换函数都需要 map_name 来查 GAMELOC_TO_PNGMAP_OFFSET_DICT，
# 因此 map_name 是构造所需的必填参数；此前漏传会让调用直接抛 TypeError。
    

class NikkiPosition():
    def __init__(self, position, map_name) -> None:
        if isinstance(position,list):
            self.position=np.array(position)
        else:
            self.position=position
        self.map_name = map_name

        self.px = self._get_px_position()
        self.game = self._get_game_location()

    def _get_px_position(self):
        pass

    def _get_game_location(self):
        pass

class PxPostion(NikkiPosition):
    def __init__(self,position, map_name) -> None:
        super().__init__(position, map_name)

    def _get_px_position(self):
        return self.position

    def _get_game_location(self):
        return convert_PngMapPx_to_GameLoc(self.position, self.map_name)
    
class GameLocation(NikkiPosition):
    def __init__(self,position, map_name) -> None:
        super().__init__(position, map_name)

    def _get_px_position(self):
        return convert_GameLoc_to_PngMapPx(self.position, self.map_name)

    def _get_game_location(self):
        return self.position
