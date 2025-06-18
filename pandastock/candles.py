import pandas as pd

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
