from dataframe_operations.base import read_parquet_benchmark
from entrypoints.common import benchmark_loop


def run():
    benchmark_loop(read_parquet_benchmark, "read_parquet.csv")


if __name__ == "__main__":
    run()
