import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import random

st.set_page_config(page_title="AI-HIC Virtual Crash Optimizer", layout="wide")

st.title("🚗 Automated Virtual Crash Test & HIC Optimization System")
st.markdown("### Industrial AI + FEA Hybrid Surrogate Model with Auto Optimization")
st.markdown("---")

# Sidebar
st.sidebar.header("🔧 Upload FEA Data / Stress Image")
uploaded_csv = st.sidebar.file_uploader("Upload FEA CSV", type=["csv"])
uploaded_file = st.sidebar.file_uploader("Upload Stress Contour Image (optional)", type=["jpg", "png"])

# Load FEA CSV
fea_df = None
if uploaded_csv is not None:
    try:
        fea_df = pd.read_csv(uploaded_csv)
        required_cols = ["X", "Y", "Thickness", "Stiffness", "Velocity", "Penetration"]
        if all(col in fea_df.columns for col in required_cols):
            st.sidebar.success("FEA data loaded ✅")
        else:
            st.sidebar.error(f"CSV must contain columns: {required_cols}")
            fea_df = None
    except Exception as e:
        st.sidebar.error(f"Error reading CSV: {e}")
        fea_df = None

# Display stress contour
if uploaded_file is not None:
    st.subheader("🖼 Stress Contour Image")
    image = Image.open(uploaded_file)
    st.image(image, use_container_width=True)

# Prediction & Auto Optimization
st.subheader("📊 Automated HIC Prediction & Optimization")

HIC_SAFE_LIMIT = 650
MAX_ITERATIONS = 10  # To avoid infinite loops

if st.button("🚀 Run Automated Optimization") and fea_df is not None:

    optimized_df = fea_df.copy()
    iteration = 0
    all_safe = False

    while iteration < MAX_ITERATIONS and not all_safe:
        hic_values = []
        recommendations = []
        all_safe = True

        for idx, row in optimized_df.iterrows():
            hic = (
                row["Thickness"] * 50 +
                row["Stiffness"] * 0.5 +
                row["Velocity"] * 30 +
                row["Penetration"] * 10 +
                random.uniform(-50, 50)
            )
            hic = round(hic, 2)
            hic_values.append(hic)

            # Optimization logic
            if hic > HIC_SAFE_LIMIT:
                all_safe = False
                recommendations.append("Increase thickness/stiffness")
                # Simple adjustment strategy
                optimized_df.at[idx, "Thickness"] += 0.5  # mm increment
                optimized_df.at[idx, "Stiffness"] += 20   # MPa increment
            else:
                recommendations.append("✅ Safe")

        optimized_df["Predicted_HIC"] = hic_values
        optimized_df["Recommendation"] = recommendations
        iteration += 1

    st.success(f"Optimization completed in {iteration} iterations")

    st.subheader("Optimized HIC & Recommendations")
    st.dataframe(optimized_df)

    # Heatmap visualization
    st.subheader("Heatmap of Optimized HIC")
    plt.figure(figsize=(8, 6))
    pivot_table = optimized_df.pivot(index="Y", columns="X", values="Predicted_HIC")
    sns.heatmap(pivot_table, annot=True, cmap="coolwarm", cbar_kws={'label': 'HIC'})
    st.pyplot(plt)
