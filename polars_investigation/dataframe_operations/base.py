from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
import json
import os
import subprocess
from abc import ABC, abstractmethod
from functools import partial
from multiprocessing import get_context
from pathlib import Path
from time import monotonic
from typing import Any, Callable, ContextManager, Generic, Optional, Self, Type, TypeVar

import pandas as pd
from memory import memory_consumption_in_bytes
from paths import DATASETS_PATH, MEMRAY_TRACK_FILE, TRAIN_PARQUET_NAME
from pydantic import BaseModel


class BenchmarkResult(BaseModel):
    name: str
    tool: str
    cpu_count: int
    time_in_seconds: float
    memory_in_bytes: int
    dataset_size: str


class ProcessTimer:
    def time_in_process(func, repeat: int, *args):
        results = []
        for _ in range(repeat):
            with get_context("spawn").Pool(processes=1) as pool:
                results.append(pool.apply(func, args))

        return sum(results)

    def time_in_process_single_cpu(func, repeat: int):
        return time_in_process(func, repeat, cpu_count=1)

    def time_in_process_all_cpu(func, repeat: int):
        return time_in_process(func, repeat, cpu_count=os.cpu_count())

    def timer(func, repeat: int, **kwargs):
        results = []
        for _ in range(repeat):
            results.append(func(**kwargs))
        return sum(results)

    def print_stat(description: str, duration: float, repeats: int):
        print(
            f"{description}: {duration}s | Mean: {duration / repeats}s | Reading rate: {repeats / duration} op/s"
        )


T = TypeVar("T")

DataFrames = tuple[T, T, T]


class BaseDataFrameOperations(ABC, Generic[T]):
    @classmethod
    @abstractmethod
    def setup(cls, cpu_count: int) -> ContextManager[Self]:
        pass

    @abstractmethod
    def filter(self, users_df: T) -> T:
        pass

    @abstractmethod
    def group(self, train_df: T) -> T:
        pass

    @abstractmethod
    def is_in(self, train_df: T) -> T:
        pass

    @abstractmethod
    def join(self, dataframes: DataFrames[T]) -> T:
        pass

    @abstractmethod
    def read_parquet(self, parquet: Path) -> T:
        pass

    # @abstractmethod
    # def string_manipulations(self, users_df: Any) -> None:
    #     pass

    # @abstractmethod
    # def quantiles(self) -> None:
    #     pass

    @abstractmethod
    def prepare_datasets(self, dataset_path: Path) -> DataFrames[T]:
        pass

    @staticmethod
    @abstractmethod
    def to_pandas(df: T) -> pd.DataFrame:
        pass

    @property
    @abstractmethod
    def provider_name(self) -> str:
        pass


def benchmark(
    func,
    name: str,
    tool: str,
    cpu_count: int,
    dataset_size: str,
    spark_get_memory: Optional[Callable[..., int]] = None,
) -> BenchmarkResult:
    memory_before = memory_consumption_in_bytes()
    start = monotonic()
    func()
    duration = monotonic() - start
    if "modin" in tool:
        memray_result_json = MEMRAY_TRACK_FILE.with_suffix(".json")
        subprocess.run(
            [
                "memray",
                "stats",
                "--json",
                str(MEMRAY_TRACK_FILE),
                "-o",
                str(memray_result_json),
            ]
        )
        memory_in_bytes = json.loads(memray_result_json.read_text())[
            "total_bytes_allocated"
        ]
        MEMRAY_TRACK_FILE.unlink()
        memray_result_json.unlink()
    elif "pyspark" in tool:
        if spark_get_memory:
            memory_in_bytes = spark_get_memory()
        else:
            raise ValueError(
                "`spark_get_memory` parameter func is not defined for pyspark run"
            )
    else:
        memory_in_bytes = memory_consumption_in_bytes() - memory_before

    return BenchmarkResult(
        name=name,
        tool=tool,
        cpu_count=cpu_count,
        time_in_seconds=duration,
        memory_in_bytes=memory_in_bytes,
        dataset_size=dataset_size,
    )


def benchmarks_to_csv(benchmarks: list[BenchmarkResult], filename: Path) -> None:
    data = pd.DataFrame([b.model_dump() for b in benchmarks])
    if filename.exists():
        data = pd.concat([data, pd.read_csv(filename)])
    data.to_csv(filename, index=False)


def repeatably_run_in_process(
    func,
    repeat: int,
    dataframe_type: Type[BaseDataFrameOperations[T]],
    cpu_count: int,
    dataset_size: str,
    **kwargs: dict[str, Any],
) -> list[BenchmarkResult]:
    results: list[BenchmarkResult] = []
    for pass_number in range(repeat):
        # print(
        #     f"Runs {pass_number + 1}/{repeat} pass with {dataframe_type} benchmark. cpu_count={cpu_count} | dataset_size={dataset_size}"
        # )
        # results.append(func(dataframe_type, cpu_count, dataset_size))
        ctx = get_context("spawn")
        with ProcessPoolExecutor(1, ctx) as pool:
            print(
                f"Runs {pass_number + 1}/{repeat} pass with {dataframe_type} benchmark. cpu_count={cpu_count} | dataset_size={dataset_size}"
            )
            future = pool.submit(
                func, dataframe_type, cpu_count, dataset_size, **kwargs
            )
            future = list(as_completed([future]))[0]
            try:
                results.append(future.result())
            except BrokenProcessPool:
                print(
                    f"{dataframe_type} OOM killed. cpu_count={cpu_count} | dataset_size={dataset_size}"
                )
                return []

    return results
