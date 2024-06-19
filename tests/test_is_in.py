from typing import Any, ContextManager

import dask.dataframe as dd
import duckdb
import modin.pandas as mpd
import pandas as pd
import polars as pl
import pytest
from dataframe_operations.dask import DaskDataFrameOperations
from dataframe_operations.duckdb import DuckDBDataFrameOperations
from dataframe_operations.modin import ModinDataFrameOperations
from dataframe_operations.pandas import PandasDataFrameOperations
from dataframe_operations.polars import PolarsDataFrameOperations
from dataframe_operations.pyspark import PysparkDataFrameOperations

INPUT_DATA = {
    "type": [0, 1, 2, 0, 1, 2],
}

EXPECTED_DATA = (
    pd.DataFrame({"type": [1, 2, 1, 2]})
    .reset_index(drop=True)
    .convert_dtypes(dtype_backend="pyarrow")
)
PD_INPUT_DATA = pd.DataFrame(INPUT_DATA).convert_dtypes(dtype_backend="pyarrow")


@pytest.mark.parametrize(
    "dataframe_operations_ctx,input_data",
    (
        pytest.param(
            PolarsDataFrameOperations.setup(1),
            pl.DataFrame(INPUT_DATA),
            id="polars",
        ),
        pytest.param(
            PandasDataFrameOperations.setup(1),
            PD_INPUT_DATA,
            id="pandas",
        ),
        pytest.param(
            DaskDataFrameOperations.setup(1),
            dd.from_pandas(PD_INPUT_DATA, npartitions=1),
            id="dask",
        ),
        pytest.param(
            ModinDataFrameOperations.setup(1, ignore_reinit=True),
            mpd.DataFrame(INPUT_DATA),
            id="modin",
        ),
        pytest.param(
            DuckDBDataFrameOperations.setup(1),
            duckdb.sql("SELECT * FROM PD_INPUT_DATA"),
            id="duckdb",
        ),
    ),
)
def test_is_in(
    dataframe_operations_ctx: ContextManager[PolarsDataFrameOperations],
    input_data: Any,
):
    with dataframe_operations_ctx as df_ops:
        result = df_ops.is_in(input_data)

        pd.testing.assert_frame_equal(
            df_ops.to_pandas(result).reset_index(drop=True),
            EXPECTED_DATA,
        )


def test_spark_is_in():
    with PysparkDataFrameOperations.setup(1) as dataframe_operations:
        input_data = dataframe_operations.spark.createDataFrame(
            pd.DataFrame(INPUT_DATA)
        )
        result = dataframe_operations.is_in(input_data)

        pd.testing.assert_frame_equal(
            dataframe_operations.to_pandas(result).reset_index(drop=True),
            EXPECTED_DATA,
        )
