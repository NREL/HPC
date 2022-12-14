import shutil

import pyspark.sql.functions as F
from pyspark.sql import SparkSession


def main():
    spark = SparkSession.builder.appName("test").getOrCreate()
    data = [{"a": i} for i in range(1000)]
    df = spark.createDataFrame(data)
    filename = "test_data.parquet"
    df.write.mode("overwrite").parquet(filename)
    try:
        df2 = spark.read.parquet(filename)
        total = df2.agg(F.sum("a").alias("sum_a")).collect()[0].sum_a
        assert total == sum(range(1000))
        print("DataFrame checks passed")
    finally:
        shutil.rmtree(filename)


if __name__ == "__main__":
    main()
