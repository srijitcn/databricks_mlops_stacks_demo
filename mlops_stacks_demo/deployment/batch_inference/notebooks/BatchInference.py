# Databricks notebook source
##################################################################################
# Batch Inference Notebook
#
# This notebook is an example of applying a model for batch inference against an input delta table,
# It is configured and can be executed as the batch_inference_job in the batch_inference_job workflow defined under
# ``mlops_stacks_demo/resources/batch-inference-workflow-resource.yml``
#
# Parameters:
#
#  * env (optional)  - String name of the current environment (dev, staging, or prod). Defaults to "dev"
#  * input_table_name (required)  - Delta table name containing your input data.
#  * output_table_name (required) - Delta table name where the predictions will be written to.
#                                   Note that this will create a new version of the Delta table if
#                                   the table already exists
#  * model_name (required) - The name of the model to be used in batch inference.
##################################################################################


# List of input args needed to run the notebook as a job.
# Provide them via DB widgets or notebook arguments.
#
# Name of the current environment
dbutils.widgets.dropdown("env", "dev", ["dev", "staging", "prod"], "Environment Name")
# dbutils.widgets.dropdown("env", "mlops_dev", ["mlops_dev", "dev", "staging", "prod"], "Environment Name")

# A Unity Catalog [update from original Hive-registered] Delta table containing the input features.
dbutils.widgets.text("input_table_name", "", label="Input Table Name") #the sample inference generated above 
# mlops_dev.cicd_proj.feature_store_inference_input

# A Unity Catalog Delta table containing the Groundtruth Labels and KeyID for joins.
# dbutils.widgets.text("groundtruth_table_name", "", label="Groundtruth Label Table Name") #the labels from sample inference generated above # mlops_dev.cicd_proj.feature_store_inference_groundtruth

# Delta table to store the output predictions.
dbutils.widgets.text("output_table_name", "", label="Output Table Name") #what was input in the instantiation: mlops_dev.cicd_proj.predictions

# Unity Catalog registered model name to use for the trained mode.
dbutils.widgets.text(
    "model_name", "dev.cicd_proj.mlops_stacks_demo-model", label="Full (Three-Level) Model Name"
) 

# COMMAND ----------

import os

notebook_path =  '/Workspace/' + os.path.dirname(dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get())
%cd $notebook_path

# COMMAND ----------

# MAGIC %pip install -r ../../../requirements.txt

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import sys
import os
notebook_path =  '/Workspace/' + os.path.dirname(dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get())
%cd $notebook_path
%cd ..
sys.path.append("../..")

# COMMAND ----------

# DBTITLE 1,Define input and output variables

env = dbutils.widgets.get("env")
input_table_name = dbutils.widgets.get("input_table_name")
# groundtruth_table_name = dbutils.widgets.get("groundtruth_table_name") ##
output_table_name = dbutils.widgets.get("output_table_name")
model_name = dbutils.widgets.get("model_name")
assert input_table_name != "", "input_table_name notebook parameter must be specified"
assert output_table_name != "", "output_table_name notebook parameter must be specified"
assert model_name != "", "model_name notebook parameter must be specified"
alias = "champion"
model_uri = f"models:/{model_name}@{alias}"

# COMMAND ----------

# env, input_table_name, groundtruth_table_name, output_table_name, model_name

# COMMAND ----------

# DBTITLE 1,if not exist create inference input & "label" table -- ref README.md
## Generating sample inference table -- your use-case may require one with a different schema etc. 
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

## if the table doesn't exist create it
# table_name = "{env_ws_catalog}.{project_db}.feature_store_inference_input"
# input_table_name ##"mlops_dev.cicd_proj.feature_store_inference_input"

if not spark._jsparkSession.catalog().tableExists(input_table_name):
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
    # ).drop(
    #     "fare_amount" ## (omit dropping column to pretend groundtruth exists/is available) 
    ## here we will assume that the label IS known and already joined to the input table
    ## in reality -- groundtruth labels are usually known later and then joined (much later) to the input table -- here we assume that they are available and joined to the input table byt timestamps + zip codes   
    ## REF: # https://docs.databricks.com/en/lakehouse-monitoring/create-monitor-api.html#example-notebooks
    ).write.mode(
        "overwrite"
    ).saveAsTable(
        # name="hive_metastore.default.taxi_scoring_sample_feature_store_inference_input" ## original
        # name="{env_ws_catalog}.{project_db}.feature_store_inference_input" ## requires stacks update + PR
        name=input_table_name ## this is the dbutils input_table_name 
    )

# COMMAND ----------

from mlflow import MlflowClient

# Get model version from alias
client = MlflowClient(registry_uri="databricks-uc")
model_version = client.get_model_version_by_alias(model_name, alias).version

# COMMAND ----------

# Get datetime
from datetime import datetime

ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# COMMAND ----------

# DBTITLE 1,Load model and run inference
from predict import predict_batch

predict_batch(spark, model_uri, input_table_name, output_table_name, model_version, ts)
dbutils.notebook.exit(output_table_name)

# COMMAND ----------



# COMMAND ----------

# DBTITLE 1,Label for Monitoring Model Performance
# https://docs.databricks.com/en/lakehouse-monitoring/create-monitor-api.html#example-notebooks

# df = spark.table("delta.`dbfs:/databricks-datasets/nyctaxi-with-zipcodes/subsampled`")

# df0 = (df
#        .withColumn("rounded_pickup_datetime",
#                      to_timestamp(rounded_unix_timestamp_udf(df["tpep_pickup_datetime"], 
#                                                             lit(15))
#                                  ),
#                     )
#        .withColumn("rounded_dropoff_datetime",
#                    to_timestamp(rounded_unix_timestamp_udf(df["tpep_dropoff_datetime"], 
#                                                            lit(30))
#                                ),
#                   )
#        .drop("tpep_pickup_datetime")
#        .drop("tpep_dropoff_datetime")
#        .drop("fare_amount")
#        .write.mode("overwrite").saveAsTable(name=groundtruth_table_name)
#         ## this is the dbutils groundtruth_table_name
#       )#.show()

# df1 = (df
#        .withColumn("rounded_pickup_datetime",
#                    to_timestamp(rounded_unix_timestamp_udf(df["tpep_pickup_datetime"], 
#                                                            lit(15))
#                                ),
#                    )
#        .withColumn("rounded_dropoff_datetime",
#                    to_timestamp(rounded_unix_timestamp_udf(df["tpep_dropoff_datetime"], 
#                                                            lit(30))
#                                ),
#                   )
#        .drop("tpep_pickup_datetime")
#        .drop("tpep_dropoff_datetime")
#        .drop("fare_amount")
#       )#.show()

# COMMAND ----------

# DBTITLE 1,test join Inference with Labels on timestamps + zips
# df1.join(df0, 
#          on=["rounded_pickup_datetime", "rounded_dropoff_datetime", 
#              "trip_distance", "pickup_zip", "dropoff_zip"], 
#          how="left"
#         ).dropDuplicates().count()

# COMMAND ----------

# df.count()

# COMMAND ----------


