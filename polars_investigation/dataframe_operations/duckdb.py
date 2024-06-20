from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator, Self

import duckdb
import pandas as pd
from dataframe_operations.base import BaseDataFrameOperations, DataFrames
from paths import TRAIN_PARQUET_NAME, USERS_PARQUET, USERS_SESSION_PARQUET


class DuckDBDataFrameOperations(BaseDataFrameOperations[duckdb.DuckDBPyRelation]):
    connection: duckdb.DuckDBPyConnection

    def __init__(self, connection: duckdb.DuckDBPyConnection):
        self.connection = connection

    @classmethod
    @contextmanager
    def setup(cls, cpu_count: int) -> Generator[Self, None, None]:
        con = duckdb.connect(database=":memory:")
        con.sql(f"SET threads TO {cpu_count};")
        # Warming up
        con.read_parquet("datasets/users.parquet").execute()

        yield cls(con)

        con.close()

    def filter(self, users_df: duckdb.DuckDBPyRelation) -> duckdb.DuckDBPyRelation:
        return users_df.filter(
            "is_deleted = false AND address LIKE '%Box%' AND balance > 10000 AND len(cards) >= 3"
        ).execute()

    def group(self, train_df: duckdb.DuckDBPyRelation) -> duckdb.DuckDBPyRelation:
        return train_df.aggregate(
            "session, first(ts) as ts_first, mean(aid) as aid_mean, sum(aid) as aid_sum, count(aid) as aid_count, "
            "median(aid) as aid_median, min(aid) as aid_min, max(aid) as aid_max"
        ).execute()

    def is_in(self, train_df: duckdb.DuckDBPyRelation) -> duckdb.DuckDBPyRelation:
        return train_df.filter("type IN (1, 2)").execute()

    def join(
        self, dataframes: DataFrames[duckdb.DuckDBPyRelation]
    ) -> duckdb.DuckDBPyRelation:
        train_data, users_session_data, users_data = dataframes
        train_data = train_data.set_alias("train_data")
        users_session_data = users_session_data.set_alias("users_session_data")
        # users_data = users_data.set_alias("users_data")
        columns_to_select = set([*train_data.columns, *users_session_data.columns]) - {
            "session_id",
            "id",
        }
        return (
            train_data.join(
                users_session_data,
                "train_data.session = users_session_data.session_id",
                how="left",
            )
            # .join(users_data, "users_session_data.user_id = users_data.id")
            .select(*columns_to_select).execute()
        )

    def read_parquet(self, parquet: Path) -> list[Any]:
        return self.connection.read_parquet(str(parquet)).execute()

    def prepare_datasets(
        self, dataset_path: Path
    ) -> DataFrames[duckdb.DuckDBPyRelation]:
        return (
            self.connection.read_parquet(
                str(dataset_path / TRAIN_PARQUET_NAME)
            ).execute(),
            self.connection.read_parquet(str(USERS_SESSION_PARQUET)).execute(),
            self.connection.read_parquet(str(dataset_path / USERS_PARQUET)).execute(),
        )

    @staticmethod
    def to_pandas(df: duckdb.DuckDBPyRelation) -> pd.DataFrame:
        return df.df().convert_dtypes(dtype_backend="pyarrow")

    @property
    def provider_name(self) -> str:
        return "duckdb"
