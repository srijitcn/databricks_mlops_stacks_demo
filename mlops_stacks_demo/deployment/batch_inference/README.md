# Batch Inference
To set up batch inference job via scheduled Databricks workflow, please refer to [mlops_stacks_demo/resources/README.md](../../resources/README.md)

## Prepare the batch inference input table for the example Project
Please run the following code in a notebook to generate the example batch inference input table.

```
from pyspark.sql.functions import to_timestamp, lit
from pyspark.sql.types import IntegerType
import math
from datetime import timedelta, timezone

def rounded_unix_timestamp(dt, num_minutes=15):
    """
    Ceilings datetime dt to interval num_minutes, then returns the unix timestamp.
    """
    nsecs = dt.minute * 60 + dt.second + dt.microsecond * 1e-6
    delta = math.ceil(nsecs / (60 * num_minutes)) * (60 * num_minutes) - nsecs
    return int((dt + timedelta(seconds=delta)).replace(tzinfo=timezone.utc).timestamp())

rounded_unix_timestamp_udf = udf(rounded_unix_timestamp, IntegerType())

## Set table_name to save sample inference input
# table_name = "{env_ws_catalog}.{project_db}.feature_store_inference_input"
table_name = "mlops_dev.cicd_proj.feature_store_inference_input"

## Add conditional  
if not spark._jsparkSession.catalog().tableExists(table_name):
    df = spark.table("delta.`dbfs:/databricks-datasets/nyctaxi-with-zipcodes/subsampled`")
    df.withColumn(
        "rounded_pickup_datetime",
        to_timestamp(rounded_unix_timestamp_udf(df["tpep_pickup_datetime"], lit(15))),
    ).withColumn(
        "rounded_dropoff_datetime",
        to_timestamp(rounded_unix_timestamp_udf(df["tpep_dropoff_datetime"], lit(30))),
    ).drop(
        "tpep_pickup_datetime"
    ).drop(
        "tpep_dropoff_datetime"
    ).drop(
        "fare_amount"
    ).write.mode(
        "overwrite"
    ).saveAsTable(
        # name="hive_metastore.default.taxi_scoring_sample_feature_store_inference_input" ## original
        # name="{env_ws_catalog}.{project_db}.feature_store_inference_input" ## requires stacks update + PR
        name=table_name ## this becomes the dbutils input_table_name 
    )
```

