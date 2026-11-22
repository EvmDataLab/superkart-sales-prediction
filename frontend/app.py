
import io
import streamlit as st
import pandas as pd
import requests

# Base URL of the Flask backend (the container name "backend" is resolved by the Docker network)
BACKEND_URL = "http://backend:7860"

# Columns required in the CSV used for batch prediction
REQUIRED_COLUMNS = [
    "Product_Weight", "Product_Sugar_Content", "Product_Allocated_Area", "Product_MRP",
    "Store_Size", "Store_Location_City_Type", "Store_Type", "Product_Id_char",
    "Store_Age_Years", "Product_Type_Category",
]


def api_error(response):
    """Extract a readable error message from a failed API response."""
    try:
        return response.json().get("error", response.text)
    except ValueError:
        return response.text


# Page title
st.title("SuperKart System")
st.write(
    "Enter the product and store details below to predict the total sales."
)

# Input fields for product and store data (defaults match the example used in the notebook)
Product_Weight = st.number_input("Product Weight", min_value=0.0, value=12.66)
Product_Sugar_Content = st.selectbox("Product Sugar Content", ["Low Sugar", "Regular", "No Sugar"])
Product_Allocated_Area = st.number_input("Product Allocated Area", min_value=0.0, value=0.027, format="%.3f")
Product_MRP = st.number_input("Product MRP", min_value=0.0, value=117.08)
Store_Size = st.selectbox("Store Size", ["Small", "Medium", "High"], index=1)
Store_Location_City_Type = st.selectbox("Store Location City Type", ["Tier 1", "Tier 2", "Tier 3"], index=1)
Store_Type = st.selectbox("Store Type", ["Supermarket Type1", "Supermarket Type2", "Departmental Store", "Food Mart"], index=1)
Product_Id_char = st.selectbox("Product ID Character", ["FD", "DR", "NC"])
Store_Age_Years = st.number_input("Store Age (Years)", min_value=0, value=16)
Product_Type_Category = st.selectbox("Product Type Category", ["Perishables", "Non Perishables"], index=1)

# Create JSON payload
product_data = {
    "Product_Weight": Product_Weight,
    "Product_Sugar_Content": Product_Sugar_Content,
    "Product_Allocated_Area": Product_Allocated_Area,
    "Product_MRP": Product_MRP,
    "Store_Size": Store_Size,
    "Store_Location_City_Type": Store_Location_City_Type,
    "Store_Type": Store_Type,
    "Product_Id_char": Product_Id_char,
    "Store_Age_Years": Store_Age_Years,
    "Product_Type_Category": Product_Type_Category
}

# Single Prediction
if st.button("Predict", type='primary'):
    try:
        response = requests.post(
            f"{BACKEND_URL}/v1/predict",
            json=product_data,
            timeout=30
        )
        if response.status_code == 200:
            predicted_sales = response.json()["Sales"]
            st.success(f"Predicted Product Store Sales Total: \u20b9{predicted_sales:,.2f}")
        else:
            st.error(f"Prediction failed ({response.status_code}): {api_error(response)}")
    except requests.exceptions.RequestException as e:
        st.error(f"Unable to connect to the prediction API: {e}")

# Batch Prediction
st.subheader("Batch Prediction")
st.write("Upload a CSV file with these columns: " + ", ".join(REQUIRED_COLUMNS))

uploaded_file = st.file_uploader(
    "Upload a CSV file",
    type=["csv"]
)

if uploaded_file is not None:
    file_bytes = uploaded_file.getvalue()
    file_key = (uploaded_file.name, len(file_bytes))

    try:
        batch_df = pd.read_csv(io.BytesIO(file_bytes))
    except Exception as e:
        batch_df = None
        st.error(f"Unable to read the uploaded file as a CSV: {e}")

    if batch_df is not None:
        st.write("Preview of the uploaded data:")
        st.dataframe(batch_df.head(), use_container_width=True)

        if st.button("Predict for Batch", type='primary'):
            try:
                response = requests.post(
                    f"{BACKEND_URL}/v1/predictbatch",
                    files={"file": ("batch.csv", file_bytes, "text/csv")},
                    timeout=120
                )
                if response.status_code == 200:
                    results = response.json()
                    result_df = batch_df.copy()
                    result_df["Predicted_Product_Store_Sales_Total"] = [results[str(i)] for i in range(len(batch_df))]
                    # keep the results in the session so they stay visible after the download button is clicked
                    st.session_state["batch_result"] = (file_key, result_df)
                else:
                    st.session_state.pop("batch_result", None)
                    st.error(f"Batch prediction failed ({response.status_code}): {api_error(response)}")
            except requests.exceptions.RequestException as e:
                st.error(f"Unable to connect to the prediction API: {e}")

        saved = st.session_state.get("batch_result")
        if saved is not None and saved[0] == file_key:
            st.success("Predictions completed successfully!")
            st.dataframe(saved[1], use_container_width=True)
            st.download_button(
                "Download predictions as CSV",
                saved[1].to_csv(index=False).encode("utf-8"),
                file_name="superkart_batch_predictions.csv",
                mime="text/csv"
            )
