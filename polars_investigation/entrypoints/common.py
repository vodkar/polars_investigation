import os
from pathlib import Path

from dataframe_operations.base import (
    BenchmarkResult,
    benchmarks_to_csv,
    repeatably_run_in_process,
)
from dataframe_operations.dask import DaskDataFrameOperations
from dataframe_operations.duckdb import DuckDBDataFrameOperations
from dataframe_operations.modin import ModinDataFrameOperations
from dataframe_operations.pandas import PandasDataFrameOperations
from dataframe_operations.polars import PolarsDataFrameOperations
from dataframe_operations.pyspark import PysparkDataFrameOperations
from paths import DATASET_SIZES

OPERATION_TYPES = [
    PandasDataFrameOperations,
    PolarsDataFrameOperations,
    DuckDBDataFrameOperations,
    ModinDataFrameOperations,
    PysparkDataFrameOperations,
]

REPEAT_N_TIMES = 3


def benchmark_loop(func, csv_name: str):
    for operations_type in OPERATION_TYPES:
        for size in DATASET_SIZES:
            benchmarks: list[BenchmarkResult] = []
            for cpu_count in [1, 2, int(os.cpu_count()) // 2, int(os.cpu_count())]:
                results = repeatably_run_in_process(
                    func, REPEAT_N_TIMES, operations_type, cpu_count, size
                )
                benchmarks.extend(results)
            benchmarks_to_csv(benchmarks, Path(csv_name))
