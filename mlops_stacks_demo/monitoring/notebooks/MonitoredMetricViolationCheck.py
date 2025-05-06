# Databricks notebook source
##################################################################################
# This notebook runs a sql query and set the result as job task value
#
# This notebook has the following parameters:
#
#  * table_name_under_monitor (required)  - The name of a table that is currently being monitored
#  * metric_to_monitor (required)  - Metric to be monitored for threshold violation
#  * metric_violation_threshold (required)  - Threshold value for metric violation
#  * num_evaluation_windows (required)  - Number of windows to check for violation
#  * num_violation_windows (required)  - Number of windows that need to violate the threshold
##################################################################################

# List of input args needed to run the notebook as a job.
# Provide them via DB widgets or notebook arguments.
#
# Name of the table that is currently being monitored
dbutils.widgets.text(
    "table_name_under_monitor", "mlops_dev.cicd_proj.predictions", label="Full (three-Level) table name"
)
# Metric to be used for threshold violation check
dbutils.widgets.text(
    "metric_to_monitor", "root_mean_squared_error", label="Metric to be monitored for threshold violation"
)

# Threshold value to be checked
dbutils.widgets.text(
    "metric_violation_threshold", "100", label="Threshold value for metric violation"
)

# Threshold value to be checked
dbutils.widgets.text(
    "num_evaluation_windows", "5", label="Number of windows to check for violation"
)

# Threshold value to be checked
dbutils.widgets.text(
    "num_violation_windows", "2", label="Number of windows that need to violate the threshold"
)

# COMMAND ----------

import os
import sys
notebook_path =  '/Workspace/' + os.path.dirname(dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get())
%cd $notebook_path
%cd ..
sys.path.append("../..")

## include 
sys.path.append(f"notebook_path.replace('notebooks','')/metric_violation_check_query.py") #mlops_stacks_demo/monitoring/metric_violation_check_query.py
#E check
%ls -lah

# COMMAND ----------

# ## check
# [print(p) for p in sys.path];

# COMMAND ----------

### check/alternative to test if still not working
# module_path = "metric_violation_check_query.py"

# # Load the module
# import importlib

# spec = importlib.util.spec_from_file_location("metric_violation_check_query", module_path)
# metric_violation_check_query = importlib.util.module_from_spec(spec)
# sys.modules["metric_violation_check_query"] = metric_violation_check_query
# spec.loader.exec_module(metric_violation_check_query)

# # Now you can use the imported module
# sql_query = metric_violation_check_query.sql_query

# COMMAND ----------

from metric_violation_check_query import sql_query

table_name_under_monitor = dbutils.widgets.get("table_name_under_monitor")
metric_to_monitor = dbutils.widgets.get("metric_to_monitor")
metric_violation_threshold = dbutils.widgets.get("metric_violation_threshold")
num_evaluation_windows = dbutils.widgets.get("num_evaluation_windows")
num_violation_windows = dbutils.widgets.get("num_violation_windows")

formatted_sql_query = sql_query.format(
    table_name_under_monitor=table_name_under_monitor,
    metric_to_monitor=metric_to_monitor,
    metric_violation_threshold=metric_violation_threshold,
    num_evaluation_windows=num_evaluation_windows,
    num_violation_windows=num_violation_windows)
is_metric_violated = bool(spark.sql(formatted_sql_query).toPandas()["query_result"][0])

dbutils.jobs.taskValues.set("is_metric_violated", is_metric_violated)