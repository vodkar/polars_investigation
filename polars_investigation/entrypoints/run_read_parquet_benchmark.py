from functools import partial
from typing import Type
from dataframe_operations.base import (
    T,
    BaseDataFrameOperations,
    BenchmarkResult,
    benchmark,
)
from entrypoints.common import benchmark_loop
from paths import DATASETS_PATH, TRAIN_PARQUET_NAME


def read_parquet_benchmark(
    dataframe_type: Type[BaseDataFrameOperations[T]],
    cpu_count: int,
    dataset_size: str,
) -> BenchmarkResult:
    with dataframe_type.setup(cpu_count) as dataframe_operations:
        return benchmark(
            partial(
                dataframe_operations.read_parquet,
                DATASETS_PATH / dataset_size / TRAIN_PARQUET_NAME,
            ),
            "read_parquet",
            dataframe_operations.provider_name,
            cpu_count,
            dataset_size,
            spark_get_memory=getattr(dataframe_operations, "memory_consumption", None),
        )


def run():
    benchmark_loop(read_parquet_benchmark, "read_parquet.csv")


if __name__ == "__main__":
    run()
