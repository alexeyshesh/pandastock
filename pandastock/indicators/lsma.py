import numpy as np
import pandas as pd

from matplotlib.axes import Axes

from .base import Indicator, PlotPosition


class LSMA(Indicator):

    plot_position = PlotPosition.over

    def __init__(self, window: int = 15, col: str = 'close'):
        self.window = window
        self.col = col

    def build(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Вычисляет Least Squares Moving Average по колонке 'close' с заданным окном.
        """
        close = data['close']
        idx = np.arange(self.window)
        result = [np.nan] * (self.window - 1)
        for i in range(self.window - 1, len(close)):
            y = close.iloc[i - self.window+1:i + 1].values
            A = np.vstack([idx, np.ones(self.window)]).T
            a, b = np.linalg.lstsq(A, y, rcond=None)[0]
            result.append(a * (self.window - 1) + b)

        return pd.Series(result, index=close.index).to_frame('lsma')

    def plot(self, data: pd.DataFrame, axes: Axes) -> None:
        axes.plot(
            range(len(data['lsma'])),
            data,
            color='lightsteelblue',
            alpha=0.6,
            linewidth=2,
            label='LSMA',
        )
        axes.legend()
