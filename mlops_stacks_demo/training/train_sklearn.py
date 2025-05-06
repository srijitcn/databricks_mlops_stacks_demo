# Install required packages (if not pre-installed)
# %pip install scikit-learn mlflow

import mlflow
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from mlflow import MlflowClient

mlflow.set_registry_uri('databricks-uc')

# Initialize MLflow experiment (Databricks automatically tracks runs)
mlflow.set_experiment("/Users/srijit.nair@databricks.com/Iris-Classification")  # Replace with your workspace path

# Load and prepare data
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, 
    test_size=0.2, random_state=42
)

with mlflow.start_run(run_name="RandomForest_Baseline") as run:
    # Enable autologging (logs metrics, params, and models)
    mlflow.sklearn.autolog()
    
    # Model training
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Predictions and evaluation
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Manual logging for additional clarity
    mlflow.log_params({
        "n_estimators": 100,
        "random_state": 42
    })
    
    mlflow.log_metrics({
        "test_accuracy": accuracy
    })
    
    # Log classification report as text artifact
    report = classification_report(y_test, y_pred, target_names=iris.target_names)
    mlflow.log_text(report, "classification_report.txt")
    
    # Log model with input example and signature
    input_example = X_train[:1]
    signature = mlflow.models.infer_signature(input_example, model.predict(input_example))
    
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="iris_model",
        input_example=input_example,
        signature=signature,
        registered_model_name="Iris_Classifier"  # Auto-registers model in MLflow Model Registry
    )

# Display results
print(f"Model accuracy: {accuracy:.4f}")
print(f"MLflow Run ID: {run.info.run_id}")
print(f"Model URI: runs:/{run.info.run_id}/iris_model")
