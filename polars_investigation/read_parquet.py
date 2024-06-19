import os
from pathlib import Path
from time import monotonic, sleep

import dask.dataframe as dd
import duckdb
import modin.pandas as mpd
import pandas as pd
import pyarrow
import ray
from pyspark.sql import SparkSession

DATASETS_PATH = Path("../datasets")
PARQUETS_PATH = DATASETS_PATH / "train.parquet"


def pl_read_parquet(cpu_count: int) -> float:
    os.environ["POLARS_MAX_THREADS"] = str(cpu_count)
    import polars as pl

    start = monotonic()
    pl.read_parquet(PARQUETS_PATH)
    return monotonic() - start


def pd_read_parquet(cpu_count: int) -> float:
    pyarrow.set_io_thread_count(cpu_count)
    pyarrow.set_cpu_count(cpu_count)
    start = monotonic()
    pd.read_parquet(PARQUETS_PATH)
    return monotonic() - start


def spark_read_parquet(cpu_count: int) -> float:
    spark = (
        SparkSession.builder.master(f"local[{cpu_count}]")
        .appName("TestSparkPerformance")
        .config("spark.driver.memory", "16g")
        .config("spark.driver.maxResultSize", "8g")
        .config("spark.executor.cores", cpu_count)
        .getOrCreate()
    )
    spark.read.parquet("../datasets/flight_parquet").count()
    start = monotonic()
    spark.read.parquet(str(PARQUETS_PATH)).count()
    runtime = monotonic() - start
    spark.stop()
    return runtime


def dask_read_parquet() -> float:
    dd.read_parquet(../datasets/flight_parquet).compute()
    start = monotonic()
    dd.read_parquet(PARQUETS_PATH).compute()
    return monotonic() - start


def modin_read_parquet(cpu_count: int) -> float:
    ray.init(num_cpus=cpu_count)
    mpd.read_parquet("../datasets/flight_parquet")
    start = monotonic()
    mpd.read_parquet(PARQUETS_PATH)
    runtime = monotonic() - start
    ray.shutdown()
    return runtime


def duckdb_read_parquet(cpu_count: int) -> float:
    con = duckdb.connect(database=":memory:")
    con.sql(f"SET threads TO {cpu_count};")
    start = monotonic()
    con.read_parquet(PARQUETS_PATH.as_posix()).fetchall()
    runtime = monotonic() - start
    con.close()
    return runtime
