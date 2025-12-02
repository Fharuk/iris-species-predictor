# ==============================
# Streamlit Iris Species Predictor
# ==============================
import streamlit as st
import pandas as pd
import joblib

# Load model and encoder safely
model = joblib.load("tuned_decision_tree_iris.pkl")
try:
    label_encoder = joblib.load("scaler_iris.pkl")
except:
    label_encoder = None  

st.set_page_config(page_title="Iris Species Predictor", layout="centered")

st.title("🌸 Iris Species Predictor")
st.write("""
Enter the sepal and petal measurements below, and the model will predict the iris species.
""")

# User input sliders
sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.8)
sepal_width  = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.0)
petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 4.35)
petal_width  = st.slider("Petal Width (cm)", 0.1, 2.5, 1.3)

# Create dataframe from inputs
input_data = pd.DataFrame({
    "SepalLengthCm": [sepal_length],
    "SepalWidthCm": [sepal_width],
    "PetalLengthCm": [petal_length],
    "PetalWidthCm": [petal_width]
})

# Prediction button
if st.button("Predict Species"):
    prediction = model.predict(input_data)
    
    # Convert to species name
    if label_encoder:
        # Force the prediction to be a 1D integer array to satisfy Scikit-Learn
        clean_prediction = prediction.astype(int).ravel()
        species_name = label_encoder.inverse_transform(clean_prediction)[0]
    else:
        # fallback in case label encoder wasn't saved
        species_map = {0: "Iris-setosa", 1: "Iris-versicolor", 2: "Iris-virginica"}
        species_name = species_map[prediction[0]]
    
    st.success(f"Predicted Iris Species: 🌿 {species_name}")
    
    # Optional: show feature importance
    if hasattr(model, "feature_importances_"):
        st.subheader("Feature Importance")
        importance_df = pd.DataFrame({
            "Feature": input_data.columns,
            "Importance": model.feature_importances_
        }).sort_values(by="Importance", ascending=False)
        st.bar_chart(importance_df.set_index("Feature"))
