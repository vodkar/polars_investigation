from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from typing import Any, Generator, Self

import memray
import pandas as pd
import ray
from dataframe_operations.base import BaseDataFrameOperations, DataFrames
from paths import (
    MEMRAY_TRACK_FILE,
    TRAIN_PARQUET_NAME,
    USERS_PARQUET,
    USERS_SESSION_PARQUET,
)


class ModinDataFrameOperations(BaseDataFrameOperations[pd.DataFrame]):
    mpd: ModuleType

    def __init__(self, mpd: ModuleType):
        self.mpd = mpd

    @classmethod
    @contextmanager
    def setup(
        cls, cpu_count: int, *, ignore_reinit: bool = False
    ) -> Generator[Self, None, None]:
        ray.init(num_cpus=cpu_count, ignore_reinit_error=ignore_reinit)
        # warming up
        import modin.pandas as mpd

        mpd.read_parquet(str(Path(__file__).parent / "flight_parquet"))
        if MEMRAY_TRACK_FILE.exists():
            MEMRAY_TRACK_FILE.unlink()
        with memray.Tracker(MEMRAY_TRACK_FILE):
            ray.get_runtime_context().get_task_id()
            yield cls(mpd)

        ray.shutdown()

    def filter(self, users_df: pd.DataFrame) -> Any:
        return users_df[
            ~users_df["is_deleted"]
            & users_df["address"].str.contains("Box")
            & (users_df["balance"] > 10000)
            & (users_df["cards"].str.len() >= 3)
            # & users_df["cards"].apply(
            #     lambda cards: any(card["provider"] == "Mastercard" for card in cards)
            # )
        ]

    def group(self, train_df: pd.DataFrame) -> pd.DataFrame:
        return (
            train_df.groupby("session", as_index=False)
            .agg(
                ts_first=pd.NamedAgg(column="ts", aggfunc="first"),
                aid_mean=pd.NamedAgg(column="aid", aggfunc="mean"),
                aid_sum=pd.NamedAgg(column="aid", aggfunc="sum"),
                aid_count=pd.NamedAgg(column="aid", aggfunc="count"),
                aid_median=pd.NamedAgg(column="aid", aggfunc="median"),
                aid_min=pd.NamedAgg(column="aid", aggfunc="min"),
                aid_max=pd.NamedAgg(column="aid", aggfunc="max"),
            )
        )

    def is_in(self, train_df: Any) -> pd.DataFrame:
        return train_df[train_df["type"].isin({1, 2})]

    def join(self, dataframes: DataFrames[pd.DataFrame]) -> pd.DataFrame:
        train_data, users_session_data, users_data = dataframes
        return (
            train_data.merge(
                users_session_data, left_on="session", right_on="session_id", how="left"
            )
            .drop(columns="session_id")
            # .merge(users_data, left_on="user_id", right_on="id")
            # .drop(columns="id")
        )

    def read_parquet(self, parquet: Path) -> pd.DataFrame:
        return self.mpd.read_parquet(parquet)

    def prepare_datasets(self, dataset_path: Path) -> DataFrames[pd.DataFrame]:
        return (
            self.mpd.read_parquet(dataset_path / TRAIN_PARQUET_NAME),
            self.mpd.read_parquet(USERS_SESSION_PARQUET),
            self.mpd.read_parquet(USERS_PARQUET),
        )

    @staticmethod
    def to_pandas(df: Any) -> pd.DataFrame:
        return df.modin.to_pandas().convert_dtypes(dtype_backend="pyarrow")

    @property
    def provider_name(self) -> str:
        return "modin"
