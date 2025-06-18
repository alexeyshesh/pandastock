from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd

from matplotlib.patches import Rectangle
from matplotlib.axes import Axes

from indicators.base import Indicator, PlotStyle, PlotPosition


class CandleFrame(pd.DataFrame):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._indicators: dict[str, Indicator] = {}
        self._indicator_results: dict[str, CandleFrame] = {}

    @staticmethod
    def _prepare_dataframe(
        df: pd.DataFrame,
        agg: str | None = None,
        remove_weekend: bool = False,
    ) -> pd.DataFrame:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        if remove_weekend:
            df = df[
                df['timestamp']
                .apply(lambda x: (x.weekday() < 5))
            ]
        df.set_index('timestamp', inplace=True)

        if agg:
            return (
                df
                .resample(agg)
                .agg(
                    {
                        'open': 'first',
                        'high': 'max',
                        'low': 'min',
                        'close': 'last',
                        'volume': 'sum',
                    },
                )
                .dropna()
            )
        return df.dropna()

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame):
        return cls(data=df.values, index=df.index, columns=df.columns)

    @classmethod
    def from_csv(
        cls,
        path: str,
        agg: str | None = None,
        remove_weekend: bool = False,
        **kwargs,
    ):
        df = pd.read_csv(path, **kwargs)
        df = cls._prepare_dataframe(df, agg, remove_weekend)
        return cls(data=df.values, index=df.index, columns=df.columns)

    def add_indicators(self, **kwargs: Indicator) -> None:
        self._indicators.update(kwargs)

        for name, indicator in kwargs.items():
            indicator_values = indicator.build(self)
            self._indicator_results[name] = indicator_values

            for col in indicator_values:
                self[f'{name}__{col}'] = indicator_values[col]

    def _plot_candles(self, left: int, right: int, axis: Axes) -> None:
        for idx, (time, row) in enumerate(self[left:right].iterrows()):
            color = 'g' if row['close'] >= row['open'] else 'r'
            axis.plot([idx, idx], [row['low'], row['high']], color='black', linewidth=1)
            axis.add_patch(
                Rectangle(
                    (idx-0.3, min(row['open'], row['close'])),
                    0.6,
                    abs(row['close'] - row['open']),
                    color=color,
                    alpha=0.7,
                )
            )
        axis.set_ylabel('Price')

    def _plot_volume(self, left: int, right: int, axes: Axes) -> None:
        axes.bar(
            range(right - left),
            self[left:right]['volume'],
            color='skyblue',
            width=0.8,
            alpha=0.7,
            label='Volume',
        )
        axes.set_ylabel('Volume')
        axes.legend()

    def _plot_indicators_over(self, left: int, right: int, axes: Axes) -> None:
        indicators = {
            n: i
            for n, i in self._indicators.items()
            if i.plot_position == PlotPosition.over
        }
        for name, indicator in indicators.items():
            indicator.plot(self._indicator_results[name][left:right], axes)

    def _plot_indicators_under(self, left: int, right: int, axes_list: list[Axes]) -> None:
        indicators = {
            n: i
            for n, i in self._indicators.items()
            if i.plot_position == PlotPosition.under
        }
        for axes, (name, indicator) in zip(axes_list, indicators.items()):
            indicator.plot(self._indicator_results[name][left:right], axes)

    def plot(  # type: ignore
        self,
        center_time: pd.Timestamp | str,
        window: int = 30,
        figsize: tuple[int, int] = (14, 10),
    ) -> None:
        center_loc: int = self.index.get_loc(center_time)  # type: ignore
        left = max(center_loc - window, 0)
        right = min(center_loc + window + 1, len(self))

        subplots_count = (
            2
            + sum(1 for i in self._indicators.values() if i.plot_position == PlotPosition.under)
        )
        _, axes = plt.subplots(
            nrows=subplots_count,  # +1 for volume
            ncols=1,
            sharex=True,
            figsize=figsize,
            gridspec_kw={'height_ratios': [3, 1] + [1 for _ in range(subplots_count - 2)]},
        )

        self._plot_candles(left, right, axes[0])
        self._plot_volume(left, right, axes[1])
        self._plot_indicators_over(left, right, axes[0])
        self._plot_indicators_under(left, right, axes[2:])

        # --- xticks с поворотом ---
        xticks_idx = range(0, (right - left), max(1, (right - left) // 10))
        xticks_labels = [self[left:right].index[i].strftime("%m-%d %H:%M") for i in xticks_idx]
        axes[-1].set_xticks(xticks_idx)
        axes[-1].set_xticklabels(xticks_labels, rotation=45)

        # --- Вертикальная линия по центру ---
        center_rel = center_loc - left
        for subax in axes:
            subax.axvline(center_rel, color='orange', linestyle='--', linewidth=1)

        plt.tight_layout()
        plt.show()
