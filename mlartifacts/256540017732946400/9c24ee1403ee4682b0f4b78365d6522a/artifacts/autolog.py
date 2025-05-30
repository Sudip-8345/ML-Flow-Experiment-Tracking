import mlflow
import mlflow.sklearn
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Load Wine dataset
wine = load_wine()
X = wine.data
y = wine.target

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)

# Define the params for RF model
max_depth = 10
n_estimators = 5

mlflow.autolog()
# mention your experiment name
mlflow.set_experiment("MLOPS-Exp1")

with mlflow.start_run():
    # Train the model
    rf = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators, random_state=42)
    rf.fit(X_train, y_train)

    # Make predictions
    y_pred = rf.predict(X_test)

    # Log model
    mlflow.sklearn.log_model(rf, "rf_model")

    # Log parameters
    # mlflow.log_param("max_depth", max_depth)
    # mlflow.log_param("n_estimators", n_estimators)

    # Log metrics
    accuracy = accuracy_score(y_test, y_pred)
    # mlflow.log_metric("accuracy", accuracy)

    # Log confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=wine.target_names, yticklabels=wine.target_names)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig("confusion_matrix.png")

    # Log confusion matrix as an artifact
    # mlflow.log_artifact("confusion_matrix.png")
    mlflow.log_artifact(__file__)  # Log the script itself

    mlflow.set_tags({'Author':'Sudip','Project':'Wine Quality Prediction'})

    
    print(f"Accuracy: {accuracy}")