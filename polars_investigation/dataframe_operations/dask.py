from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator, Self

import dask.config
import dask.dataframe as dd
import pandas as pd
from dask.distributed import LocalCluster
from dataframe_operations.base import BaseDataFrameOperations, DataFrames
from paths import TRAIN_PARQUET_NAME, USERS_PARQUET, USERS_SESSION_PARQUET


class DaskDataFrameOperations(BaseDataFrameOperations[Any]):

    @classmethod
    @contextmanager
    def setup(cls, cpu_count: int) -> Generator[Self, None, None]:
        with dask.config.set(
            {"distributed.worker.daemon": False, "dataframe.convert-string": False}
        ):
            with LocalCluster(
                threads_per_worker=cpu_count,
                n_workers=1,
                processes=False,
            ) as _:
                yield cls()

    def filter(self, users_df: Any) -> Any:
        return users_df[
            ~users_df["is_deleted"]
            & users_df["address"].str.contains("Box")
            & (users_df["balance"] > 10000)
            & (users_df["cards"].str.len() >= 3)
            # & users_df["cards"].apply(
            #     lambda cards: any(card["provider"] == "Mastercard" for card in cards)
            # )
        ].compute()

    def group(self, train_df: Any) -> pd.DataFrame:
        return (
            train_df.groupby("session")
            .agg(
                ts_first=pd.NamedAgg(column="ts", aggfunc="first"),
                aid_mean=pd.NamedAgg(column="aid", aggfunc="mean"),
                aid_sum=pd.NamedAgg(column="aid", aggfunc="sum"),
                aid_count=pd.NamedAgg(column="aid", aggfunc="count"),
                aid_median=pd.NamedAgg(column="aid", aggfunc="median"),
                aid_min=pd.NamedAgg(column="aid", aggfunc="min"),
                aid_max=pd.NamedAgg(column="aid", aggfunc="max"),
            )
            .reset_index()
            .compute()
        )

    def is_in(self, train_df: Any) -> pd.DataFrame:
        return train_df[train_df["type"].isin({1, 2})].compute()

    def join(self, dataframes: DataFrames[Any]) -> pd.DataFrame:
        train_data, users_session_data, users_data = dataframes
        return (
            train_data.merge(
                users_session_data, left_on="session", right_on="session_id", how="left"
            )
            .drop(columns="session_id")
            # .merge(users_data, left_on="user_id", right_on="id")
            # .drop(columns="id")
            .compute()
        )

    def read_parquet(self, parquet: Path) -> pd.DataFrame:
        return dd.read_parquet(parquet).compute()

    def prepare_datasets(self, dataset_path: Path) -> DataFrames[pd.DataFrame]:
        return (
            dd.read_parquet(dataset_path / TRAIN_PARQUET_NAME).persist(),
            dd.read_parquet(USERS_SESSION_PARQUET).persist(),
            dd.read_parquet(USERS_PARQUET).persist(),
        )

    @staticmethod
    def to_pandas(df: Any) -> pd.DataFrame:
        # Computed Dask df is a pandas df already
        return df.convert_dtypes(dtype_backend="pyarrow")

    @property
    def provider_name(self) -> str:
        return "dask"
