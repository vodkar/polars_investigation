from typing import ContextManager

import dask.dataframe as dd
import duckdb
import modin.pandas as mpd
import pandas as pd
import polars as pl
import pytest
from dataframe_operations.base import BaseDataFrameOperations, T
from dataframe_operations.dask import DaskDataFrameOperations
from dataframe_operations.duckdb import DuckDBDataFrameOperations
from dataframe_operations.modin import ModinDataFrameOperations
from dataframe_operations.pandas import PandasDataFrameOperations
from dataframe_operations.polars import PolarsDataFrameOperations
from dataframe_operations.pyspark import PysparkDataFrameOperations

INPUT_DATA = (
    {
        "session": [1, 2, 3, 4, 5],
        "type": [0, 1, 2, 0, 1],
    },
    {
        "session_id": [1, 2, 3, 4],
        "user_id": [1, 2, 3, 4],
    },
    {
        "id": [1, 2, 3, 4, 5],
        "name": ["Alice", "Bob", "Charlie", "David", "Eve"],
    },
)
PD_INPUT_DATA_1, PD_INPUT_DATA_2, PD_INPUT_DATA_3 = (
    pd.DataFrame(INPUT_DATA[0]).convert_dtypes(dtype_backend="pyarrow"),
    pd.DataFrame(INPUT_DATA[1]).convert_dtypes(dtype_backend="pyarrow"),
    pd.DataFrame(INPUT_DATA[2]).convert_dtypes(dtype_backend="pyarrow"),
)


EXPECTED_DATA = (
    pd.DataFrame(
        {
            "session": [1, 2, 3, 4],
            "type": [0, 1, 2, 0],
            "user_id": [1, 2, 3, 4],
            "name": ["Alice", "Bob", "Charlie", "David"],
        }
    )
    .reset_index(drop=True)
    .convert_dtypes(dtype_backend="pyarrow")
)


@pytest.mark.parametrize(
    "dataframe_operations_ctx,dataframes",
    (
        pytest.param(
            PolarsDataFrameOperations.setup(1),
            (
                pl.DataFrame(INPUT_DATA[0]),
                pl.DataFrame(INPUT_DATA[1]),
                pl.DataFrame(INPUT_DATA[2]),
            ),
            id="polars",
        ),
        pytest.param(
            PandasDataFrameOperations.setup(1),
            (PD_INPUT_DATA_1, PD_INPUT_DATA_2, PD_INPUT_DATA_3),
            id="pandas",
        ),
        pytest.param(
            DaskDataFrameOperations.setup(1),
            (
                dd.from_pandas(PD_INPUT_DATA_1, npartitions=1),
                dd.from_pandas(PD_INPUT_DATA_2, npartitions=1),
                dd.from_pandas(PD_INPUT_DATA_3, npartitions=1),
            ),
            id="dask",
        ),
        pytest.param(
            ModinDataFrameOperations.setup(1, ignore_reinit=True),
            (
                mpd.DataFrame(INPUT_DATA[0]),
                mpd.DataFrame(INPUT_DATA[1]),
                mpd.DataFrame(INPUT_DATA[2]),
            ),
            id="modin",
        ),
        pytest.param(
            DuckDBDataFrameOperations.setup(1),
            (
                duckdb.sql("SELECT * FROM PD_INPUT_DATA_1"),
                duckdb.sql("SELECT * FROM PD_INPUT_DATA_2"),
                duckdb.sql("SELECT * FROM PD_INPUT_DATA_3"),
            ),
            id="duckdb",
        ),
    ),
)
def test_join(
    dataframe_operations_ctx: ContextManager[BaseDataFrameOperations[T]],
    dataframes: tuple[T, T, T],
):
    with dataframe_operations_ctx as df_ops:
        result = df_ops.join(dataframes)

        pd.testing.assert_frame_equal(
            df_ops.to_pandas(result).reset_index(drop=True)[EXPECTED_DATA.columns],
            EXPECTED_DATA,
        )


def test_spark_join():
    with PysparkDataFrameOperations.setup(1) as dataframe_operations:
        input_data_1 = dataframe_operations.spark.createDataFrame(
            pd.DataFrame(INPUT_DATA[0])
        )
        input_data_2 = dataframe_operations.spark.createDataFrame(
            pd.DataFrame(INPUT_DATA[1])
        )
        input_data_3 = dataframe_operations.spark.createDataFrame(
            pd.DataFrame(INPUT_DATA[2])
        )
        result = dataframe_operations.join((input_data_1, input_data_2, input_data_3))

        pd.testing.assert_frame_equal(
            dataframe_operations.to_pandas(result)
            .sort_values(by="session")
            .reset_index(drop=True),
            EXPECTED_DATA,
        )
