# train.py
# Train a simple Logistic Regression model on the Iris dataset and save it.

from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

def train_and_save_model(output_path="model.joblib"):
    # Load the Iris dataset
    iris = load_iris(as_frame=True)
    X = iris.data
    y = iris.target

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train the model
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)

    # Test and show accuracy
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Test Accuracy: {accuracy:.2f}")

    # Save the trained model
    joblib.dump(model, output_path)
    print(f"Model saved as: {output_path}")

if __name__ == "__main__":
    train_and_save_model()
