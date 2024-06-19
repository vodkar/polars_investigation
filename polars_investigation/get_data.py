import dask.dataframe as dd
import modin.pandas as mpd
import pandas as pd
from paths import TRAIN_PARQUET_NAME, USERS_PARQUET, USERS_SESSION_PARQUET


def pl_get_data(cpu_count: int):
    os.environ["POLARS_MAX_THREADS"] = str(cpu_count)
    import polars as pl

    train_data = pl.read_parquet(TRAIN_PARQUET_NAME)
    train_data.count()
    users_data = pl.read_parquet(USERS_PARQUET)
    users_data.count()
    users_session_data = pl.read_parquet(USERS_SESSION_PARQUET)

    return train_data, users_data, users_session_data


def pd_get_data():
    train_data = pd.read_parquet(TRAIN_PARQUET_NAME)
    users_data = pd.read_parquet(USERS_PARQUET)
    users_session_data = pd.read_parquet(USERS_SESSION_PARQUET)

    return train_data, users_data, users_session_data


def spark_get_data(spark):
    train_data = spark.read.parquet(str(TRAIN_PARQUET_NAME))
    users = spark.read.parquet(str(USERS_PARQUET))
    users_sessions = spark.read.parquet(str(USERS_SESSION_PARQUET))

    return train_data, users, users_sessions


def dask_get_data():
    train_data = dd.read_parquet(TRAIN_PARQUET_NAME)
    users_data = dd.read_parquet(USERS_PARQUET)
    users_session_data = dd.read_parquet(USERS_SESSION_PARQUET)

    return train_data, users_data, users_session_data


def modin_get_data():
    train_data = mpd.read_parquet(TRAIN_PARQUET_NAME).rename(
        columns={"session": "session_id"}
    )
    users_data = mpd.read_parquet(USERS_PARQUET)
    users_session_data = mpd.read_parquet(USERS_SESSION_PARQUET)

    return train_data, users_data, users_session_data


def duckdb_get_data(duckdb_connection):
    train_data = duckdb_connection.read_parquet(TRAIN_PARQUET_NAME.as_posix())
    users_data = duckdb_connection.read_parquet(USERS_PARQUET.as_posix())
    users_session_data = duckdb_connection.read_parquet(
        USERS_SESSION_PARQUET.as_posix()
    )

    return train_data, users_data, users_session_data
