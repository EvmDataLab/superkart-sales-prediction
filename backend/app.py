
# Import necessary libraries
import os
import joblib  # For loading the serialized model
import pandas as pd  # For data manipulation
from flask import Flask, request, jsonify  # For creating the Flask API

# Initialize Flask app with a name
superkart_api = Flask("SuperKart")

# Load the trained model pipeline (preprocessing steps + model) saved in the notebook
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "superkart_model.joblib")
model = joblib.load(MODEL_PATH)

# Input schema expected by the model
NUMERIC_FEATURES = ['Product_Weight', 'Product_Allocated_Area', 'Product_MRP', 'Store_Age_Years']
CATEGORICAL_FEATURES = ['Product_Sugar_Content', 'Store_Size', 'Store_Location_City_Type',
                        'Store_Type', 'Product_Id_char', 'Product_Type_Category']
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def prepare_input(df):
    """Validate the input dataframe and return only the model features in a clean format."""
    missing = [col for col in FEATURES if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required fields: {missing}")

    df = df[FEATURES].copy()  # extra columns (if any) are ignored
    for col in NUMERIC_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="raise")  # raises ValueError for non-numeric values
    if df.isnull().any().any():
        raise ValueError("Input contains missing values")
    return df


# Define a route for the home page
@superkart_api.get('/')
def home():
    return "Welcome to the SuperKart System"


# Define an endpoint to predict sales for a single product (JSON in, JSON out)
@superkart_api.post('/v1/predict')
def predict_sales():
    # Get JSON data from the request
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({'error': 'Request body must be a JSON object.'}), 400

    # Validate and convert the input into the format expected by the model
    try:
        input_data = prepare_input(pd.DataFrame([data]))
    except ValueError as e:
        return jsonify({'error': str(e)}), 400

    # Make a prediction using the trained model
    try:
        prediction = float(model.predict(input_data)[0])
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {e}'}), 500

    # Return the prediction as a JSON response
    return jsonify({'Sales': round(prediction, 2)})


# Define an endpoint to predict sales for a batch of products (CSV file in, JSON out)
@superkart_api.post('/v1/predictbatch')
def predict_sales_batch():
    # Get the uploaded CSV file from the request
    file = request.files.get('file')
    if file is None:
        return jsonify({'error': "No file found. Upload a CSV file using the form field name 'file'."}), 400

    # Read the file into a DataFrame and validate it
    try:
        input_data = prepare_input(pd.read_csv(file))
    except Exception as e:
        return jsonify({'error': f'Invalid CSV file: {e}'}), 400

    # Make predictions for the batch data
    try:
        predictions = model.predict(input_data).tolist()
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {e}'}), 500

    # Create an output dictionary mapping row index to predicted sales
    output_dict = {str(i): round(float(pred), 2) for i, pred in enumerate(predictions)}

    return jsonify(output_dict)


# Run the Flask app (in production the app is served by Gunicorn, see the Dockerfile)
if __name__ == '__main__':
    superkart_api.run(host="0.0.0.0", port=7860, debug=False)
