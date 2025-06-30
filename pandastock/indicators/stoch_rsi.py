import numpy as np
import pandas as pd

from matplotlib.axes import Axes

from .base import Indicator, PlotPosition
from .rsi import RSI


class StochasticRSI(Indicator):

    plot_position = PlotPosition.under

    def __init__(self, period: int = 14, k: int = 3, d: int = 3, col: str = 'close'):
        self.period = period
        self.k = k
        self.d = d
        self.col = col
        self._rsi = RSI(period, col)
        self._rsi_values = []
        self._k_values = []

    def next_value(self, candle: pd.Series) -> pd.Series:
        rsi_value = self._rsi.next_value(candle)['rsi']
        self._rsi_values.append(rsi_value)

        if len(self._rsi_values) < self.period:
            return pd.Series({'stoch_rsi': np.nan})

        # Calculate stochastic RSI
        current_rsi = self._rsi_values[-1]
        min_rsi = min(self._rsi_values[-self.period:])
        max_rsi = max(self._rsi_values[-self.period:])

        if max_rsi == min_rsi:
            stoch_rsi = 0.0
        else:
            stoch_rsi = 100 * (current_rsi - min_rsi) / (max_rsi - min_rsi)

        self._k_values.append(stoch_rsi)

        if len(self._k_values) < self.k:
            return pd.Series({'stoch_rsi': np.nan})

        # Calculate K line (simple moving average of stoch_rsi)
        if len(self._k_values) < self.k + self.d - 1:
            return pd.Series({'stoch_rsi': np.nan})

        # Calculate D line (simple moving average of K line)
        d_line = sum(self._k_values[-self.d:]) / self.d

        return pd.Series({'stoch_rsi': d_line})

    def build(self, data: pd.DataFrame) -> pd.DataFrame:
        result = []
        for _, row in data.iterrows():
            result.append(self.next_value(row))
        df = pd.concat(result, axis=1).T
        df.index = data.index
        return df

    @property
    def name(self) -> str:
        return f'StochRSI {self.d} {self.k}'

    def plot(self, data: pd.DataFrame, axes: Axes) -> None:
        axes.plot(
            range(len(data['stoch_rsi'])),
            data['stoch_rsi'],
            label=self.name,
            color='blue',
        )

        axes.axhline(y=100, color='white')
        axes.axhline(y=0, color='white')
        axes.axhline(y=80, color='gray', linestyle='--', linewidth=0.9)
        axes.axhline(y=20, color='gray', linestyle='--', linewidth=0.9)

        axes.set_ylabel(self.name)
        axes.legend()
