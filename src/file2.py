import mlflow
import mlflow.sklearn
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from mlflow.models.signature import infer_signature
import dagshub

# Initialize DagsHub integration
dagshub.init(repo_owner='Sudip-8345', repo_name='ML-Flow-Experiment-Tracking', mlflow=True)
mlflow.set_tracking_uri('https://dagshub.com/Sudip-8345/ML-Flow-Experiment-Tracking.mlflow')

# Set the experiment name
mlflow.set_experiment("MLOPS-Exp2")

# Load data
wine = load_wine()
X = wine.data
y = wine.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)

# Parameters
max_depth = 8
n_estimators = 6

with mlflow.start_run():
    # Train model
    rf = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators, random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    # Log model with input example and signature
    input_example = X_test[:1]
    signature = infer_signature(X_test, y_pred)
    mlflow.sklearn.log_model(rf, "model", input_example=input_example, signature=signature)

    # Log params and metrics
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=wine.target_names, yticklabels=wine.target_names)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig("confusion_matrix.png")
    mlflow.log_artifact("confusion_matrix.png")

    # Tagging
    mlflow.set_tags({'Author': 'Sudip', 'Project': 'Wine Quality Prediction'})

    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
