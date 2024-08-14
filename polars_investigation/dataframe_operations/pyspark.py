from contextlib import contextmanager
from pathlib import Path
from typing import Generator, Self

import pandas as pd
import pyspark.sql.functions as f
from dataframe_operations.base import BaseDataFrameOperations, DataFrames
from paths import TRAIN_PARQUET_NAME, USERS_PARQUET, USERS_SESSION_PARQUET
from pyspark.sql import DataFrame, SparkSession


class PysparkDataFrameOperations(BaseDataFrameOperations[DataFrame]):
    _memory_consumption: int

    def __init__(self, spark: SparkSession):
        self.spark = spark

    @classmethod
    @contextmanager
    def setup(cls, cpu_count: int) -> Generator[Self, None, None]:
        spark = (
            SparkSession.builder.master(f"local[{cpu_count}]")
            .appName("TestSparkPerformance")
            .config("spark.driver.memory", "80g")
            .config("spark.driver.maxResultSize", "80g")
            .config("spark.executor.memory", "80g")
            .config("spark.executor.cores", cpu_count)
            .config("spark.python.profile.memory", True)
            .getOrCreate()
        )
        # Warming up the JVM
        spark.read.parquet("datasets/users.parquet").count()

        yield cls(spark)

        spark.stop()

    def filter(self, users_df: DataFrame) -> DataFrame:
        df = users_df.filter(
            ~f.col("is_deleted")
            & f.col("address").contains("Box")
            & (f.col("balance") > 10000)
            & (f.size("cards") >= 3)
            # & f.expr("exists(cards, x -> x.provider == 'Mastercard')")
        )
        # Trigger the computation
        df.count()

        self._update_memory_consumption(df)

        return df

    def group(self, train_df: DataFrame) -> DataFrame:
        df = train_df.groupby("session").agg(
            f.first("ts").alias("ts_first"),
            f.mean("aid").alias("aid_mean"),
            f.sum("aid").alias("aid_sum"),
            f.count("aid").alias("aid_count"),
            f.expr("percentile_approx(aid, 0.5)").alias("aid_median"),
            f.min("aid").alias("aid_min"),
            f.max("aid").alias("aid_max"),
        )
        df.count()

        self._update_memory_consumption(df)

        return df

    def is_in(self, train_df: DataFrame) -> DataFrame:
        result = train_df[train_df["type"].isin({1, 2})]
        result.count()

        self._update_memory_consumption(result)

        return result

    def join(self, dataframes: DataFrames[DataFrame]) -> DataFrame:
        train_data, users_session_data, users_data = dataframes
        result = (
            train_data.join(
                users_session_data,
                train_data.session == users_session_data.session_id,
                how="left",
            ).drop("session_id")
            # .join(users_data, users_session_data.user_id == users_data.id)
            # .drop("id")
        )
        result.count()

        self._update_memory_consumption(result)

        return result

    def read_parquet(self, parquet: Path) -> pd.DataFrame:
        df = self.spark.read.parquet(str(parquet))
        df.count()
        self._update_memory_consumption(df)
        return df

    def prepare_datasets(self, dataset_path: Path) -> DataFrames[DataFrame]:
        dfs = (
            self.spark.read.parquet(str(dataset_path / TRAIN_PARQUET_NAME)),
            self.spark.read.parquet(str(USERS_SESSION_PARQUET)),
            self.spark.read.parquet(str(dataset_path / USERS_PARQUET)),
        )
        dfs[0].count()
        dfs[1].count()
        dfs[2].count()

        return dfs

    @staticmethod
    def to_pandas(df: DataFrame) -> pd.DataFrame:
        return df.toPandas().convert_dtypes(dtype_backend="pyarrow")

    @property
    def provider_name(self) -> str:
        return "pyspark"

    def memory_consumption(self) -> int:
        return self._memory_consumption

    def _update_memory_consumption(self, df: DataFrame) -> None:
        self._memory_consumption = (
            self.spark.sparkContext._jvm.org.apache.spark.util.SizeEstimator.estimate(
                df._jdf
            )
        )
