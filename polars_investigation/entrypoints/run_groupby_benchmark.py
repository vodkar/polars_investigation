from functools import partial

from dataframe_operations.base import (
    BaseDataFrameOperations,
    BenchmarkResult,
    T,
    benchmark,
)
from entrypoints.common import benchmark_loop
from paths import DATASETS_PATH


def group(
    dataframe_type: type[BaseDataFrameOperations[T]], cpu_count: int, dataset_size: str
) -> BenchmarkResult:
    with dataframe_type.setup(cpu_count) as dataframe_operations:
        return benchmark(
            partial(
                dataframe_operations.group,
                dataframe_operations.prepare_datasets(DATASETS_PATH / dataset_size)[0],
            ),
            "group",
            dataframe_operations.provider_name,
            cpu_count,
            dataset_size,
            spark_get_memory=getattr(dataframe_operations, "memory_consumption", None),
        )


def run():
    benchmark_loop(group, "group.csv")


if __name__ == "__main__":
    run()
