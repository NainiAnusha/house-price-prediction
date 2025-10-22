import streamlit as st
import pandas as pd
import joblib

# Load the model
loaded_model = joblib.load('model.pkl')

# Title
st.title("House Price Prediction App")

# Upload CSV
uploaded_file = st.file_uploader("Upload your CSV file for input data", type=["csv"])

if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file)
        st.subheader("Uploaded Data:")
        st.dataframe(data)

        # Check if 'city' exists and encode it
        if 'city' in data.columns:
            data['city_encoded'] = data['city'].astype('category').cat.codes

        # Drop non-numeric columns
        numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns.tolist()

        selected_index = st.selectbox("Select a row for prediction:", data.index)

        st.subheader("Modify or Enter Input Data:")
        input_data = {}
        for col in numeric_columns:
            input_data[col] = st.number_input(
                f"{col}:",
                value=float(data.loc[selected_index, col]),
                step=0.01,
                format="%.2f",
                key=col
            )

        input_df = pd.DataFrame([input_data])

        if st.button("Predict"):
            # Match model's expected columns
            expected_features = loaded_model.feature_names_in_
            missing_cols = [col for col in expected_features if col not in input_df.columns]
            for col in missing_cols:
                input_df[col] = 0  # Fill missing ones with 0

            input_df = input_df[expected_features]  # Arrange columns in correct order

            prediction = loaded_model.predict(input_df)
            st.success(f"Predicted Value: *{prediction[0]:,.2f}*")

    except Exception as e:
        st.error(f"Error processing the file: {e}")

else:
    st.info("Please upload a CSV file to begin.")