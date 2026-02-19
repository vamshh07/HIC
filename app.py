import streamlit as st
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import random

st.set_page_config(page_title="AI-HIC Predictor", layout="wide")

st.title("🚗 Multimodal AI-Based Pedestrian Head Injury Prediction System")
st.markdown("### CNN + DNN Hybrid Surrogate Model Demo")

st.markdown("---")

# Sidebar
st.sidebar.header("🔧 Input Parameters")

thickness = st.sidebar.number_input("Thickness (mm)", 0.0, 10.0, 2.5)
stiffness = st.sidebar.number_input("Stiffness (MPa)", 0.0, 1000.0, 450.0)
velocity = st.sidebar.number_input("Impact Velocity (m/s)", 0.0, 30.0, 11.0)
penetration = st.sidebar.number_input("Penetration Depth (mm)", 0.0, 50.0, 15.0)

uploaded_file = st.sidebar.file_uploader("Upload Stress Contour Image", type=["jpg", "png"])

col1, col2 = st.columns(2)

with col1:
    st.subheader("🖼 Stress Contour Image")
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, use_container_width=True)
    else:
        st.info("Upload a stress contour image.")

with col2:
    st.subheader("📊 Prediction Panel")

    if st.button("🔥 Predict HIC"):

        # Simulated Hybrid Model Prediction
        hic_prediction = (
            thickness * 50 +
            stiffness * 0.5 +
            velocity * 30 +
            penetration * 10 +
            random.uniform(-50, 50)
        )

        hic_prediction = round(hic_prediction, 2)

        if hic_prediction < 650:
            risk = "🟢 SAFE"
        elif hic_prediction <= 1000:
            risk = "🟡 MODERATE"
        else:
            risk = "🔴 DANGEROUS"

        st.success(f"Predicted HIC: {hic_prediction}")
        st.warning(f"Risk Level: {risk}")

        # Graph
        fig, ax = plt.subplots()
        ax.bar(["Predicted HIC"], [hic_prediction])
        ax.axhline(650)
        ax.axhline(1000)
        ax.set_ylabel("HIC Value")
        st.pyplot(fig)

st.markdown("---")
st.caption("AI-Assisted Simulation Acceleration for Vehicle Safety Design")