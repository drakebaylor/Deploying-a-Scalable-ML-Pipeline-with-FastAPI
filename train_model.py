import os
import pandas as pd
from sklearn.model_selection import train_test_split

from ml.data import process_data
from ml.model import (
    compute_model_metrics,
    inference,
    load_model,
    performance_on_categorical_slice,
    save_model,
    train_model,
)


project_path = "C://Users//drake//Documents//WGU//DevOps//Deploying-a-Scalable-ML-Pipeline-with-FastAPI"
data_path = os.path.join(project_path, "data", "census.csv")
print(data_path)

# TODO: Load the census.csv data
data = pd.read_csv(data_path)

# TODO: Split the provided data into train and test datasets
train, test = train_test_split(data, test_size=0.20, random_state=42)

# DO NOT MODIFY
cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

# TODO: Process the training data
X_train, y_train, encoder, lb = process_data(
    train,
    categorical_features=cat_features,
    label="salary",
    training=True
)

# Process the test data
X_test, y_test, _, _ = process_data(
    test,
    categorical_features=cat_features,
    label="salary",
    training=False,
    encoder=encoder,
    lb=lb
)

# TODO: Train the model on the training dataset
model = train_model(X_train, y_train)

# Save the model and the encoder
model_dir = os.path.join(project_path, "model")
os.makedirs(model_dir, exist_ok=True)

model_path = os.path.join(model_dir, "model.pkl")
save_model(model, model_path)

encoder_path = os.path.join(model_dir, "encoder.pkl")
save_model(encoder, encoder_path)

lb_path = os.path.join(model_dir, "label_binarizer.pkl")
save_model(lb, lb_path)


model = load_model(model_path)

# TODO: Run inference on the test dataset
preds = inference(model, X_test)

# Calculate and print overall model metrics
p, r, fb = compute_model_metrics(y_test, preds)
print(f"Overall Precision: {p:.4f} | Recall: {r:.4f} | F1: {fb:.4f}")

# TODO: Compute the performance on model slices
slice_report_path = os.path.join(project_path, "slice_output.txt")
if os.path.exists(slice_report_path):
    os.remove(slice_report_path)

for col in cat_features:
    for slice_value in sorted(test[col].unique()):
        count = test[test[col] == slice_value].shape[0]
        p, r, fb = performance_on_categorical_slice(
            data=test,
            column_name=col,
            slice_value=slice_value,
            categorical_features=cat_features,
            label="salary",
            encoder=encoder,
            lb=lb,
            model=model
        )
        with open(slice_report_path, "a") as f:
            print(f"{col}: {slice_value}, Count: {count:,}", file=f)
            print(f"Precision: {p:.4f} | Recall: {r:.4f} | F1: {fb:.4f}", file=f)
            print("-" * 50, file=f)

print(f"Slice performance metrics written to: {slice_report_path}")
