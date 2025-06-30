import numpy as np
import pandas as pd

from matplotlib.axes import Axes

from pandastock.indicators.base import Indicator, PlotPosition


class RSI(Indicator):

    plot_position = PlotPosition.under

    def __init__(
        self,
        period: int = 14,
        col: str = 'close',
    ):
        self.period = period
        self.col = col
        self._prev_value = None
        self._avg_gain = None
        self._avg_loss = None

    def next_value(self, candle: pd.Series) -> pd.Series:
        current_value = candle[self.col]

        if self._prev_value is None:
            self._prev_value = current_value
            return pd.Series({'rsi': np.nan})

        delta = current_value - self._prev_value
        self._prev_value = current_value

        gain = max(delta, 0)
        loss = max(-delta, 0)

        if self._avg_gain is None or self._avg_loss is None:
            self._avg_gain = gain
            self._avg_loss = loss
            return pd.Series({'rsi': np.nan})

        # Update averages
        alpha = 1 / self.period
        self._avg_gain = gain * alpha + self._avg_gain * (1 - alpha)
        self._avg_loss = loss * alpha + self._avg_loss * (1 - alpha)

        if self._avg_loss == 0:
            rsi = 100.0
        else:
            rs = self._avg_gain / self._avg_loss
            rsi = 100 - (100 / (1 + rs))

        return pd.Series({'rsi': np.round(rsi, 2)})

    def build(self, data: pd.DataFrame, name: str = 'rsi') -> pd.DataFrame:
        result = []
        for _, row in data.iterrows():
            result.append(self.next_value(row))
        df = pd.concat(result, axis=1).T
        df.index = data.index
        return df

    @property
    def name(self) -> str:
        return f'RSI {self.period}'

    def plot(self, data: pd.DataFrame, axes: Axes) -> None:
        axes.plot(
            range(len(data['rsi'])),
            data['rsi'],
            label=self.name,
            color='purple',
        )

        axes.axhline(y=100, color='white')
        axes.axhline(y=0, color='white')

        axes.set_ylabel(self.name)
        axes.legend()
