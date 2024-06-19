from functools import partial

from dataframe_operations.base import (
    BaseDataFrameOperations,
    BenchmarkResult,
    T,
    benchmark,
)
from entrypoints.common import benchmark_loop
from paths import DATASETS_PATH


def is_in(
    dataframe_type: type[BaseDataFrameOperations[T]], cpu_count: int, dataset_size: str
) -> BenchmarkResult:
    with dataframe_type.setup(cpu_count) as dataframe_operations:
        return benchmark(
            partial(
                dataframe_operations.is_in,
                dataframe_operations.prepare_datasets(DATASETS_PATH / dataset_size)[0],
            ),
            "is_in",
            f"{dataframe_operations.provider_name}-is_in",
            cpu_count,
            dataset_size,
            spark_get_memory=getattr(dataframe_operations, "memory_consumption", None),
        )


def run():
    benchmark_loop(is_in, "is_in.csv")


if __name__ == "__main__":
    run()
