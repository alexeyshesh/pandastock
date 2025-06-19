import pandas as pd

from matplotlib.axes import Axes

from .base import Indicator, PlotPosition


class SMA(Indicator):

    plot_position = PlotPosition.over

    def __init__(self, window: int = 15, col: str = 'close'):
        self.window = window
        self.col = col

    def build(self, data: pd.DataFrame) -> pd.DataFrame:
        return (
            data[self.col]
            .rolling(window=self.window, min_periods=self.window)
            .mean()
            .to_frame('sma')
        )

    def plot(self, data: pd.DataFrame, axes: Axes) -> None:
        axes.plot(
            range(len(data['sma'])),
            data,
            color='lightsteelblue',
            alpha=0.6,
            linewidth=2,
            label='SMA',
        )
        axes.legend()
