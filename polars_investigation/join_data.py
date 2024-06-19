import os
from pathlib import Path
from time import monotonic, sleep

import dask.dataframe as dd
import duckdb
import modin.pandas as mpd
import modin.utils
import pandas as pd
import ray
from memory import memory_consumption_in_bytes
from pyspark.sql import SparkSession

DATASETS_PATH = Path("../datasets")
PARQUETS_PATH = DATASETS_PATH / "train.parquet"
USERS_SESSION_PARQUETS_PATH = DATASETS_PATH / "users_session.parquet"
USERS_PARQUETS_PATH = DATASETS_PATH / "users.parquet"


def pl_join(cpu_count: int):
    os.environ["POLARS_MAX_THREADS"] = str(cpu_count)
    import polars as pl

    train_data = pl.read_parquet(PARQUETS_PATH).rename({"session": "session_id"})
    train_data.count()
    users_data = pl.read_parquet(USERS_PARQUETS_PATH)
    users_data.count()
    users_session_data = pl.read_parquet(USERS_SESSION_PARQUETS_PATH)
    start = monotonic()
    train_data.join(users_session_data, on="session_id", how="left").join(
        users_data.rename({"id": "user_id"}), on="user_id"
    )
    return monotonic() - start, memory_consumption_in_bytes()


def pd_join(_: int):
    train_data = pd.read_parquet(PARQUETS_PATH).rename(
        columns={"session": "session_id"}
    )
    users_data = pd.read_parquet(USERS_PARQUETS_PATH)
    users_session_data = pd.read_parquet(USERS_SESSION_PARQUETS_PATH)
    start = monotonic()
    train_data.join(users_session_data, on="session_id", how="left", lsuffix="l_").join(
        users_data.rename({"id": "user_id"}), on="user_id"
    )
    return monotonic() - start, memory_consumption_in_bytes()


def spark_join(cpu_count: int):
    spark = (
        SparkSession.builder.master(f"local[{cpu_count}]")
        .appName("TestSparkPerformance")
        .config("spark.driver.memory", "16g")
        .config("spark.driver.maxResultSize", "8g")
        .config("spark.executor.cores", cpu_count)
        .getOrCreate()
    )
    spark.read.parquet("../datasets/flight_parquet").count()
    train_data = spark.read.parquet(str(PARQUETS_PATH))
    users = spark.read.parquet(str(USERS_PARQUETS_PATH))
    users_sessions = spark.read.parquet(str(USERS_SESSION_PARQUETS_PATH))
    start = monotonic()
    train_data.join(
        users_sessions, train_data.session == users_sessions.session_id, "left"
    ).join(users, users.id == users_sessions.user_id).collect()
    runtime = monotonic() - start
    spark.stop()
    return runtime, memory_consumption_in_bytes()


def dask_join():
    train_data = dd.read_parquet(PARQUETS_PATH).rename(
        columns={"session": "session_id"}
    )
    users_data = dd.read_parquet(USERS_PARQUETS_PATH)
    users_session_data = dd.read_parquet(USERS_SESSION_PARQUETS_PATH)
    start = monotonic()
    train_data.join(users_session_data, on="session_id", how="left", lsuffix="l_").join(
        users_data.rename(columns={"id": "user_id"}), on="user_id", lsuffix="l_"
    ).compute()
    return monotonic() - start, memory_consumption_in_bytes()


def modin_join(cpu_count: int):
    ray.init(num_cpus=cpu_count)
    train_data = mpd.read_parquet(PARQUETS_PATH).rename(
        columns={"session": "session_id"}
    )
    users_data = mpd.read_parquet(USERS_PARQUETS_PATH)
    users_session_data = mpd.read_parquet(USERS_SESSION_PARQUETS_PATH)
    start = monotonic()
    modin.utils.execute(
        train_data.join(users_session_data, on="session_id", how="left").join(
            users_data.rename({"id": "user_id"}), on="user_id"
        )
    )
    runtime = monotonic() - start
    ray.shutdown()
    return runtime, memory_consumption_in_bytes()


def duckdb_join(cpu_count: int):
    con = duckdb.connect()
    con.sql(f"SET threads TO {cpu_count};")
    con.read_parquet(PARQUETS_PATH.as_posix()).fetchall()
    train_data = con.read_parquet(PARQUETS_PATH.as_posix()).set_alias("td")
    users_data = con.read_parquet(USERS_PARQUETS_PATH.as_posix()).set_alias("u")
    users_session_data = con.read_parquet(
        USERS_SESSION_PARQUETS_PATH.as_posix()
    ).set_alias("us")
    start = monotonic()
    train_data.join(users_session_data, "td.session == us.session_id", "left").join(
        users_data, "us.user_id == u.id"
    ).fetchall()
    runtime = monotonic() - start
    con.close()
    return runtime, memory_consumption_in_bytes()
