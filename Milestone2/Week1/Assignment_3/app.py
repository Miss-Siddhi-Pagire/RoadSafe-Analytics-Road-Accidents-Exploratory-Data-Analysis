# Streamlit web app to explore the Iris dataset and make predictions.
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

# Basic page setup
st.set_page_config(page_title="Iris Classifier", layout="centered")
st.title("🌸 Iris Classifier")
st.write("Predict Iris species using a trained Logistic Regression model.")

# Load dataset
iris = load_iris(as_frame=True)
df = iris.frame
feature_names = iris.feature_names
target_names = iris.target_names

# Choose app mode
mode = st.sidebar.radio("Choose a mode", ["Predict Data", "Explore Data"])

# Explore data mode
if mode == "Explore Data":
    st.header("🔍 Explore the Iris Dataset")

    st.subheader("📊 First five rows")
    st.dataframe(df.head())

    st.subheader("📈 Histogram")
    feature = st.selectbox("Select feature", feature_names)
    bins = st.slider("Number of bins", 5, 30, 10)
    fig, ax = plt.subplots()
    ax.hist(df[feature], bins=bins, color="skyblue", edgecolor="black")
    ax.set_xlabel(feature)
    ax.set_ylabel("Count")
    st.pyplot(fig)

    st.subheader("🌿 Scatter Plot")
    x_feature = st.selectbox("X-axis", feature_names, index=0)
    y_feature = st.selectbox("Y-axis", feature_names, index=1)
    fig2, ax2 = plt.subplots()
    ax2.scatter(df[x_feature], df[y_feature], c=df["target"], cmap="viridis")
    ax2.set_xlabel(x_feature)
    ax2.set_ylabel(y_feature)
    st.pyplot(fig2)

# Prediction mode
else:
    st.header("🔮 Predict Iris Species")
    st.write("Adjust the sliders and click Predict:")

    # Input sliders
    inputs = []
    for feature in feature_names:
        min_val = float(df[feature].min())
        max_val = float(df[feature].max())
        default_val = float(df[feature].mean())
        val = st.slider(feature, min_value=min_val, max_value=max_val, value=default_val, step=0.1)
        inputs.append(val)

    # Load model
    try:
        model = joblib.load("model.joblib")
    except:
        st.error("Model not found. Run train.py first.")
        st.stop()

    # Make prediction
    if st.button("Predict"):
        input_df = pd.DataFrame([inputs], columns=feature_names)
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0]

        st.success(f"🌺 Predicted Species: **{target_names[prediction]}**")
        st.write("🔢 Prediction probabilities:")
        st.dataframe(pd.DataFrame([probability], columns=target_names))
