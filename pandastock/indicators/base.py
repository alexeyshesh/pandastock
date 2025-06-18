from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import auto, StrEnum
from typing import TYPE_CHECKING

import pandas as pd

from matplotlib.axes import Axes

if TYPE_CHECKING:
    from candle import CandleFrame


class PlotPosition(StrEnum):
    over = auto()
    under = auto()


@dataclass
class PlotStyle:
    color: str
    position: PlotPosition
    levels: tuple[float] | None = None
    minmax: tuple[float, float] | None = None


class Indicator(ABC):

    plot_position: PlotPosition = NotImplemented

    def __init__(self):
        pass

    @abstractmethod
    def build(self, data: 'CandleFrame') -> 'CandleFrame':
        ...

    @abstractmethod
    def plot(self, data: 'CandleFrame', axes: Axes):
        ...
