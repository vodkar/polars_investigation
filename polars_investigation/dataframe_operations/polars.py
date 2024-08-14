import os
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from typing import Generator, Self

import pandas as pd
import polars as pl
from dataframe_operations.base import BaseDataFrameOperations, DataFrames
from paths import TRAIN_PARQUET_NAME, USERS_PARQUET, USERS_SESSION_PARQUET


class PolarsDataFrameOperations(BaseDataFrameOperations[pl.DataFrame]):

    def __init__(self, pl: ModuleType):
        self.pl = pl

    @classmethod
    @contextmanager
    def setup(cls, cpu_count: int) -> Generator[Self, None, None]:
        os.environ["POLARS_MAX_THREADS"] = str(cpu_count)

        yield cls(pl)

    def filter(self, users_df: pl.DataFrame) -> pl.DataFrame:
        return users_df.filter(
            self.pl.col("is_deleted").not_()
            & self.pl.col("address").str.contains("Box")
            & (self.pl.col("balance") > 10000)
            & (self.pl.col("cards").list.len() >= 3)
            # & (
            #     self.pl.col("cards")
            #     .list.eval(self.pl.element().struct.field("provider") == "Mastercard")
            #     .list.any()
            # ),
        )

    def group(self, train_df: pl.DataFrame) -> pl.DataFrame:
        return train_df.groupby("session").agg(
            [
                pl.col("ts").first().alias("ts_first"),
                pl.col("aid").mean().alias("aid_mean"),
                pl.col("aid").sum().alias("aid_sum"),
                pl.col("aid").count().alias("aid_count"),
                pl.col("aid").median().alias("aid_median"),
                pl.col("aid").min().alias("aid_min"),
                pl.col("aid").max().alias("aid_max"),
            ]
        )

    def is_in(self, train_df: pl.DataFrame) -> pl.DataFrame:
        return train_df.filter(
            self.pl.col("type").is_in({1, 2}),
        )

    def join(self, dataframes: DataFrames[pl.DataFrame]) -> pl.DataFrame:
        train_data, users_session_data, users_data = dataframes
        return (
            train_data.join(
                users_session_data,
                left_on="session",
                right_on="session_id",
                how="left",
            )
            # .join(users_data, left_on="user_id", right_on="id")
        )

    def read_parquet(self, parquet: Path) -> pl.DataFrame:
        return self.pl.read_parquet(parquet)

    def prepare_datasets(self, dataset_path: Path) -> DataFrames[pl.DataFrame]:
        return (
            pl.read_parquet(dataset_path / TRAIN_PARQUET_NAME),
            pl.read_parquet(USERS_SESSION_PARQUET),
            pl.read_parquet(dataset_path / USERS_PARQUET),
        )

    @staticmethod
    def to_pandas(df: pl.DataFrame) -> pd.DataFrame:
        return df.to_pandas().convert_dtypes(dtype_backend="pyarrow")

    @property
    def provider_name(self) -> str:
        return "polars"
