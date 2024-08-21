from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator, Self

import pandas as pd
import pyarrow
from dataframe_operations.base import BaseDataFrameOperations, DataFrames
from paths import TRAIN_PARQUET_NAME, USERS_PARQUET, USERS_SESSION_PARQUET


class PandasDataFrameOperations(BaseDataFrameOperations[pd.DataFrame]):

    @classmethod
    @contextmanager
    def setup(cls, cpu_count: int) -> Generator[Self, None, None]:
        pyarrow.set_io_thread_count(cpu_count)
        pyarrow.set_cpu_count(cpu_count)

        yield cls()

    def filter(self, users_df: pd.DataFrame) -> Any:
        return users_df[
            ~users_df["is_deleted"]
            & users_df["address"].str.contains("Box")
            & (users_df["balance"] > 10000)
            & (users_df["cards"].str.len() >= 3)
        ]

    def group(self, train_df: pd.DataFrame) -> pd.DataFrame:
        return train_df.groupby("session", as_index=False).agg(
            ts_first=pd.NamedAgg(column="ts", aggfunc="first"),
            aid_mean=pd.NamedAgg(column="aid", aggfunc="mean"),
            aid_sum=pd.NamedAgg(column="aid", aggfunc="sum"),
            aid_count=pd.NamedAgg(column="aid", aggfunc="count"),
            aid_median=pd.NamedAgg(column="aid", aggfunc="median"),
            aid_min=pd.NamedAgg(column="aid", aggfunc="min"),
            aid_max=pd.NamedAgg(column="aid", aggfunc="max"),
        )

    def is_in(self, train_df: Any) -> pd.DataFrame:
        return train_df[train_df["type"].isin({1, 2})]

    def join(self, dataframes: DataFrames[pd.DataFrame]) -> pd.DataFrame:
        train_data, users_session_data, users_data = dataframes
        return (
            train_data.merge(
                users_session_data, left_on="session", right_on="session_id", how="left"
            ).drop(columns="session_id")
        )

    def read_parquet(self, parquet: Path) -> pd.DataFrame:
        return pd.read_parquet(parquet)

    def prepare_datasets(self, dataset_path: Path) -> DataFrames[pd.DataFrame]:
        return (
            pd.read_parquet(dataset_path / TRAIN_PARQUET_NAME),
            pd.read_parquet(USERS_SESSION_PARQUET),
            pd.read_parquet(dataset_path / USERS_PARQUET),
        )

    @staticmethod
    def to_pandas(df: pd.DataFrame) -> pd.DataFrame:
        return df

    @property
    def provider_name(self) -> str:
        return "pandas"
