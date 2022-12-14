## Interactive R
```
$ singularity run instance://spark sparkR
> library(sparklyr)
> sc = spark_connect(code-examples = paste0("spark://",Sys.info()["nodename"],":7077"))
> df = spark_read_parquet(sc = sc, path = "my_data.parquet", memory = FALSE)
```
