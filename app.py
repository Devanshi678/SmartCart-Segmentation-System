import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

# Page Configuration
st.set_page_config(page_title="SmartCart Segment Predictor & Benchmarks", page_icon="🛒", layout="wide")

# Custom Color Palette for Clusters (0: Red, 1: Blue, 2: Yellow/Gold, 3: Green)
COLOR_MAP = {
    "0": "red",
    "1": "blue",
    "2": "#f1c40f",
    "3": "green"
}

# Load saved artifacts and segmented dataset
@st.cache_resource
def load_artifacts():
    scaler = joblib.load("scaler.joblib")
    classifier = joblib.load("classifier.joblib")
    model_columns = joblib.load("model_columns.joblib")
    df_segmented = pd.read_csv("smartcart_segmented_customers.csv")
    return scaler, classifier, model_columns, df_segmented

scaler, classifier, model_columns, df_segmented = load_artifacts()

# Business Acumen Strategies
PERSONAS = {
    0: {
        "title": "Cluster 0: Family Shoppers",
        "desc": "Younger demographic with more children and lower spending across channels.",
        "strategy": "💰 Actionable Strategy: Offer targeted discounts, coupons, family bundle offers, and value sales."
    },
    1: {
        "title": "Cluster 1: Premium Couples",
        "desc": "Older couples with higher income, high channel purchases, and fewer children.",
        "strategy": "💎 Actionable Strategy: Enroll in exclusive loyalty programs and high-tier reward programs."
    },
    2: {
        "title": "Cluster 2: Digital Bargain Shoppers",
        "desc": "Young single demographic with high web visits but very low purchases.",
        "strategy": "⚡ Actionable Strategy: Target with flash sales, clearance deals, and promotional discounts."
    },
    3: {
        "title": "Cluster 3: High-Value VIP Singles",
        "desc": "High-income single customers with high spending and highest campaign response rate.",
        "strategy": "🏆 Actionable Strategy: Provide premium, highly personalized concierge services and early releases."
    }
}

st.title("🛒 SmartCart Customer Analytics & Segment Predictor")

# Navigation Tabs
tab1, tab2 = st.tabs(["🎯 Predict Customer Segment", "📊 Cluster Benchmarks & Averages"])

# =========================================================
# TAB 1: PREDICT CUSTOMER SEGMENT
# =========================================================
with tab1:
    st.markdown("Enter customer details below to predict their segment and receive recommended marketing strategies.")

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)

        with col1:
            age = st.number_input("Age", min_value=18, max_value=100, value=35)
            income = st.number_input("Annual Income ($)", min_value=0, max_value=300000, value=55000, step=1000)
            total_spending = st.number_input("Total Spending ($)", min_value=0, max_value=10000, value=600, step=50)
            total_children = st.number_input("Total Children", min_value=0, max_value=5, value=1)
            tenure_days = st.number_input("Customer Tenure (Days)", min_value=0, max_value=4000, value=365)

        with col2:
            education = st.selectbox("Education Level", ["Graduate", "Post-Graduate", "Undergraduate"])
            living_with = st.selectbox("Living Situation", ["Couple", "Single"])
            web_purchases = st.number_input("Web Purchases", min_value=0, max_value=50, value=3)
            catalog_purchases = st.number_input("Catalog Purchases", min_value=0, max_value=50, value=1)
            store_purchases = st.number_input("Store Purchases", min_value=0, max_value=50, value=4)
            web_visits = st.number_input("Web Visits / Month", min_value=0, max_value=30, value=6)

        submit_btn = st.form_submit_button("Predict Segment")

    if submit_btn:
        # 1. Format input payload
        input_dict = {
            "Age": age,
            "Income": income,
            "Total_spending": total_spending,
            "Total_childrens": total_children,
            "cutomer_tenure_days": tenure_days,
            "NumWebPurchases": web_purchases,
            "NumCatalogPurchases": catalog_purchases,
            "NumStorePurchases": store_purchases,
            "NumWebVisitsMonth": web_visits,
            "Education": education,
            "Living_with": living_with
        }

        input_df = pd.DataFrame([input_dict])
        input_encoded = pd.get_dummies(input_df, columns=["Education", "Living_with"], drop_first=False)

        # 2. Align input columns to model_columns
        input_ready = pd.DataFrame(0, index=[0], columns=model_columns)
        for col in input_encoded.columns:
            if col in model_columns:
                input_ready[col] = input_encoded[col].iloc[0]

        # 3. Scale input
        input_scaled = scaler.transform(input_ready)

        # 4. Predict cluster directly via Classifier
        predicted_cluster = int(classifier.predict(input_scaled)[0])

        # 5. Display prediction result and persona
        st.divider()
        st.success(f"### Predicted Segment: Cluster {predicted_cluster}")
        
        info = PERSONAS.get(predicted_cluster, {})
        st.subheader(info.get("title", ""))
        st.write(info.get("desc", ""))
        st.info(info.get("strategy", ""))

# =========================================================
# TAB 2: CLUSTER BENCHMARKS & AVERAGES
# =========================================================
with tab2:
    st.subheader("📊 Average Metrics & Category Distributions Across Clusters")
    st.markdown("Use these benchmark figures from your segmented dataset to guide your input values.")

    # Define target numerical input features
    num_input_cols = [
        "Age", "Income", "Total_spending", "Total_childrens",
        "cutomer_tenure_days", "NumWebPurchases", "NumCatalogPurchases",
        "NumStorePurchases", "NumWebVisitsMonth"
    ]

    # Calculate cluster averages
    cluster_means = df_segmented.groupby("Cluster")[num_input_cols].mean().reset_index()
    cluster_means["Cluster_Str"] = cluster_means["Cluster"].astype(str)

    # --- 1. Summary Table ---
    st.markdown("#### 📋 Cluster Numerical Averages Summary")
    st.dataframe(
        cluster_means.style.format({
            "Income": "${:,.2f}",
            "Total_spending": "${:,.2f}",
            "Age": "{:.1f}",
            "Total_childrens": "{:.1f}",
            "cutomer_tenure_days": "{:.0f}",
            "NumWebPurchases": "{:.1f}",
            "NumCatalogPurchases": "{:.1f}",
            "NumStorePurchases": "{:.1f}",
            "NumWebVisitsMonth": "{:.1f}"
        }),
        use_container_width=True
    )

    st.divider()

    # --- 2. Dynamic Numerical Feature Visualizer ---
    st.markdown("#### 📈 Numerical Metric Comparison Chart")
    selected_metric = st.selectbox(
        "Select an input metric to view average per cluster:",
        num_input_cols,
        index=1  # Defaults to Income
    )

    fig_metric = px.bar(
        cluster_means,
        x="Cluster_Str",
        y=selected_metric,
        color="Cluster_Str",
        color_discrete_map=COLOR_MAP,
        text_auto=".1f",
        title=f"Average {selected_metric} across Clusters",
        labels={"Cluster_Str": "Cluster Segment", selected_metric: f"Average {selected_metric}"}
    )
    fig_metric.update_traces(textposition="outside")
    st.plotly_chart(fig_metric, use_container_width=True)

    st.divider()

    # --- 3. Categorical Breakdown Charts ---
    st.markdown("#### 🎓 Education & 🏠 Living Situation Breakdown per Cluster")
    cat_col1, cat_col2 = st.columns(2)

    with cat_col1:
        # Education Category Breakdown
        edu_df = df_segmented.groupby(["Cluster", "Education"]).size().reset_index(name="Count")
        edu_df["Cluster_Str"] = edu_df["Cluster"].astype(str)

        fig_edu = px.bar(
            edu_df,
            x="Cluster_Str",
            y="Count",
            color="Education",
            barmode="group",
            title="Education Breakdown by Cluster",
            labels={"Cluster_Str": "Cluster Segment", "Count": "Customer Count"}
        )
        st.plotly_chart(fig_edu, use_container_width=True)

    with cat_col2:
        # Living Situation Breakdown
        living_df = df_segmented.groupby(["Cluster", "Living_with"]).size().reset_index(name="Count")
        living_df["Cluster_Str"] = living_df["Cluster"].astype(str)

        fig_living = px.bar(
            living_df,
            x="Cluster_Str",
            y="Count",
            color="Living_with",
            barmode="group",
            title="Living Situation Breakdown by Cluster",
            labels={"Cluster_Str": "Cluster Segment", "Count": "Customer Count"}
        )
        st.plotly_chart(fig_living, use_container_width=True)