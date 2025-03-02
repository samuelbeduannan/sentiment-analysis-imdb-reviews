
import argparse
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import mlflow
def train(args):

    input_train_data_path = args.input_train_data
    input_test_data_path = args.input_test_data
    output_data_path = args.output_data
    method = args.method
    max_features = args.max_features
    
    # Load training data
    train_data = pd.read_csv(os.path.join(input_train_data_path, "train.csv"))
    validation_data = pd.read_csv(os.path.join(input_test_data_path, "test.csv"))
    X_train = train_data["text"]
    y_train = train_data["label"]

    print("data has been read")
    X_validation = validation_data["text"]
    y_validation = validation_data["label"]

    # set the tracking server uri using the arn of the tracking server you created
    mlflow.set_tracking_uri(os.environ['MLFLOW_TRACKING_ARN'])

    # Enable autologging in MLFlow
    mlflow.autolog()

    
    # Feature extraction
    if method == "tfidf":
        vectorizer = TfidfVectorizer(max_features=max_features)
    elif method == "bow":
        vectorizer = CountVectorizer(max_features=max_features)
    else:
        raise ValueError("Invalid feature extraction method. Choose 'tfidf' or 'bow'.")
    
    X_train_vec = vectorizer.fit_transform(X_train)
    X_validation_vec = vectorizer.transform(X_validation)

    print("vectorizing completed")
    # Train SVM
    svm = SVC(C=args.C, kernel=args.kernel, gamma=args.gamma)
    svm.fit(X_train_vec, y_train)
    print("training completed")
    # Evaluate and log accuracy
    y_pred = svm.predict(X_validation_vec)
    print("prediction done")
    accuracy = accuracy_score(y_validation, y_pred)
    f1 = f1_score(y_validation, y_pred, average="macro")
    precision = precision_score(y_validation, y_pred, average="macro")
    recall = recall_score(y_validation, y_pred, average="macro")
    print(f"validation:accuracy={accuracy}")
    print(f"validation:f1={f1}")
    print(f"validation:precision={precision}")
    print(f"validation:recall={recall}")
    
    # Save model and vectorizer
    os.makedirs(output_data_path, exist_ok=True)
    import joblib
    joblib.dump(svm, os.path.join(output_data_path, "svm_model.pkl"))
    joblib.dump(vectorizer, os.path.join(output_data_path, "vectorizer.pkl"))
    print("Model and vectorizer saved.")

    # Register the model with MLflow
    run_id = mlflow.last_active_run().info.run_id
    artifact_path = "model"
    model_uri = "runs:/{run_id}/{artifact_path}".format(run_id=run_id, artifact_path=artifact_path)
    model_details = mlflow.register_model(model_uri=model_uri, name="svm-experiment-model")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-train-data", type=str, default="/opt/ml/input/data/train")
    parser.add_argument("--input-test-data", type=str, default="/opt/ml/input/data/test") 
    parser.add_argument("--output-data", type=str, default="/opt/ml/model") 
    parser.add_argument("--method", type=str, default="tfidf")
    parser.add_argument("--max-features", type=int, default=5000)
    parser.add_argument("--C", type=float, default=1.5)
    parser.add_argument("--kernel", type=str, default="rbf")
    parser.add_argument("--gamma", type=float, default=0.4)
    args = parser.parse_args()
    train(args)
