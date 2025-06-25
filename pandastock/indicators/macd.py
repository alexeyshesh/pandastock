import numpy as np
import pandas as pd

from matplotlib.axes import Axes

from .base import Indicator, PlotPosition


class MACD(Indicator):

    plot_position = PlotPosition.under

    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9, col: str = 'close'):
        self.fast = fast
        self.slow = slow
        self.signal = signal
        self.col = col

    def _ema(self, series: pd.Series, window: int) -> pd.Series:
        return series.ewm(span=window, adjust=False).mean()

    def build(self, data: pd.DataFrame) -> pd.DataFrame:
        close = data[self.col]
        fast_ema = self._ema(close, self.fast)
        slow_ema = self._ema(close, self.slow)

        macd_line = fast_ema - slow_ema
        signal_line = self._ema(macd_line, self.signal)
        histogram = macd_line - signal_line

        return pd.DataFrame({
            'macd': macd_line,
            'signal': signal_line,
            'histogram': histogram,
        })

    def plot(self, data: pd.DataFrame, axes: Axes) -> None:
        axes.plot(
            range(len(data['macd'])),
            data['macd'],
            color='blue',
            alpha=0.8,
            linewidth=1,
            label='MACD',
        )
        axes.plot(
            range(len(data['signal'])),
            data['signal'],
            color='orange',
            alpha=0.8,
            linewidth=1,
            label='Signal',
        )
        axes.bar(
            range(len(data['histogram'])),
            data['histogram'],
            color=np.where(data['histogram'] >= 0, 'green', 'red'),
            alpha=0.3,
            width=0.8,
            label='Histogram',
        )
        axes.legend()
