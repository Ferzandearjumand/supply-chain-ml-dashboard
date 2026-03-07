import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import time
from scipy.signal import savgol_filter

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(page_title="Supply Chain ML Thesis Companion", layout="wide")

st.title("Predicting Supply Chain Performance Using Advanced Machine Learning")

st.write("""
This interactive application presents a visual companion to the thesis,
including the research problem, theoretical framework, empirical results,
and interactive model learning curves.
""")

# --------------------------------------------------
# NAVIGATION (replaces tabs)
# --------------------------------------------------

section = st.radio(
"",
[
"Overview",
"Research Model",
"Data & Variables",
"Models",
"Results",
"Learning Curves",
"Conclusion"
],
horizontal=True
)

# --------------------------------------------------
# OVERVIEW
# --------------------------------------------------

if section == "Overview":

    st.header("Problem Statement")

    st.write("""
Supply chain performance is strongly influenced by organisational
capabilities such as integration, information sharing, agility,
and risk management.

This research investigates whether advanced machine learning models
provide superior predictive performance compared with classical
regression methods when analysing supply chain capability data.
""")

    st.info("""
Key Insight

Machine learning models do not always outperform classical regression
when datasets are relatively small and structured.
""")

# --------------------------------------------------
# RESEARCH MODEL
# --------------------------------------------------

elif section == "Research Model":

    st.header("Research Model")

    st.image("research model 25Jan2026.jpg")

# --------------------------------------------------
# DATA & VARIABLES
# --------------------------------------------------

elif section == "Data & Variables":

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Independent Variables")

        st.markdown("""
• Supply Chain Integration (SCI)

• Information Sharing (IS)

• Supply Chain Agility (SCA)

• Supply Chain Risk Management (SCRM)
""")

    with col2:

        st.subheader("Dependent Variable")

        st.markdown("""
• Supply Chain Performance (SCP)

Measured through Operational and Financial Performance.
""")

    st.image("Figure_4_Explanatory_Overview.svg")

# --------------------------------------------------
# MODELS
# --------------------------------------------------

elif section == "Models":

    st.header("Predictive Models Used")

    st.markdown("""
### Linear Regression
Baseline statistical model used for comparison.

### Random Forest
Tree ensemble capturing nonlinear relationships.

### XGBoost
Gradient boosting algorithm with strong predictive performance.

### CatBoost
Boosting model optimized for categorical data.

### Artificial Neural Network
Multilayer perceptron capable of modelling nonlinear relationships.
""")

# --------------------------------------------------
# RESULTS
# --------------------------------------------------

elif section == "Results":

    st.header("Model Performance Comparison")

    st.image("Table_4_11_APA.svg")

    st.image("Figure_5_1_Ensemble_Feature_Importance.svg")

    st.image("Figure_5_2_Cross_Model_Consistency.svg")

# --------------------------------------------------
# LEARNING CURVES (YOUR ORIGINAL DASHBOARD)
# --------------------------------------------------

elif section == "Learning Curves":

    # LOAD DATA

    df = pd.read_csv("learning_curve_results.csv")

    # --------------------------------------------------
    # DATASET SUMMARY
    # --------------------------------------------------

    colA, colB, colC = st.columns(3)

    colA.metric("Total Responses", df.shape[0])
    colB.metric("Predictor Variables", 4)
    colC.metric("Models Compared", 5)

    # MODEL COLORS

    colors = {
    "LR": "#1f77b4",
    "RF": "#ff7f0e",
    "XGB": "#2ca02c",
    "CatBoost": "#d62728",
    "ANN": "#9467bd"
    }

    models = ["LR","RF","XGB","CatBoost","ANN"]

    # --------------------------------------------------
    # SIDEBAR CONTROLS (ONLY HERE)
    # --------------------------------------------------

    st.sidebar.header("Controls")

    sample_size = st.sidebar.slider(
    "Select Sample Size",
    min_value=int(df.sample_size.min()),
    max_value=int(df.sample_size.max()),
    value=120
    )

    animate = st.sidebar.checkbox("Animate Learning", False)

    stable_region = st.sidebar.checkbox(
    "Focus on Stable Learning Region (n ≥ 80)"
    )

    st.sidebar.download_button(
    label="Download Learning Curve Data",
    data=df.to_csv(index=False),
    file_name="learning_curve_results.csv",
    mime="text/csv"
    )

    # --------------------------------------------------
    # ANIMATION
    # --------------------------------------------------

    if animate:
        for i in range(int(df.sample_size.min()), sample_size+1):
            st.session_state["current_n"] = i
            time.sleep(0.03)
    else:
        st.session_state["current_n"] = sample_size

    current_n = st.session_state["current_n"]

    filtered_df = df[df["sample_size"] <= current_n]
    latest = filtered_df.tail(1)

    values = latest[models].values.flatten()

    # --------------------------------------------------
    # CURRENT PERFORMANCE
    # --------------------------------------------------

    st.subheader("Current Model Performance")

    col1, col2, col3 = st.columns([1.2,2,2])

    # TABLE

    with col1:

        ranking = latest.drop(columns=["sample_size"]).T
        ranking.columns = ["R² Score"]
        ranking = ranking.round(3)
        ranking = ranking.sort_values("R² Score", ascending=False)

        st.dataframe(ranking, use_container_width=True)

    # DOT SNAPSHOT

    with col2:

        fig_dot = go.Figure()

        fig_dot.add_trace(go.Scatter(
            x=models,
            y=[round(v,3) for v in values],
            mode="markers",
            marker=dict(
                size=16,
                color=[colors[m] for m in models]
            ),
            hovertemplate="Model: %{x}<br>R²: %{y:.3f}<extra></extra>"
        ))

        fig_dot.update_layout(
            title=f"Model Performance at n = {current_n}",
            yaxis_title="R² Score",
            xaxis_title="Model",
            height=350
        )

        st.plotly_chart(fig_dot, use_container_width=True)

    # BAR CHART

    with col3:

        fig_bar = go.Figure()

        fig_bar.add_trace(go.Bar(
            x=models,
            y=[round(v,3) for v in values],
            marker_color=[colors[m] for m in models],
            hovertemplate="Model: %{x}<br>R²: %{y:.3f}<extra></extra>"
        ))

        fig_bar.update_layout(
            title="Performance Comparison",
            yaxis_title="R² Score",
            height=350
        )

        st.plotly_chart(fig_bar, use_container_width=True)

    # --------------------------------------------------
    # LEARNING CURVES
    # --------------------------------------------------

    st.subheader("Model Learning Behaviour")

    @st.cache_data
    def smooth_data(df):

        smooth_df = df.copy()

        window = 21
        poly = 3

        smooth_df["LR"] = savgol_filter(df["LR"], window, poly)
        smooth_df["RF"] = savgol_filter(df["RF"], window, poly)
        smooth_df["XGB"] = savgol_filter(df["XGB"], window, poly)
        smooth_df["CatBoost"] = savgol_filter(df["CatBoost"], window, poly)
        smooth_df["ANN"] = savgol_filter(df["ANN"], window, poly)

        return smooth_df

    smooth_df = smooth_data(df)

    smooth_df["max_model"] = smooth_df[models].max(axis=1)
    smooth_df["min_model"] = smooth_df[models].min(axis=1)

    if stable_region:
        smooth_df = smooth_df[smooth_df["sample_size"] >= 80]

    fig = go.Figure()

    for model in models:

        fig.add_trace(go.Scatter(
            x=smooth_df["sample_size"],
            y=smooth_df[model],
            mode="lines",
            name=model,
            line=dict(color=colors[model], width=3),
            hovertemplate="Sample Size: %{x}<br>R²: %{y:.3f}"
        ))

    fig.add_vline(
    x=current_n,
    line_width=2,
    line_dash="dash",
    line_color="black"
    )

    fig.update_layout(
    title="Learning Curves (Smoothed)",
    xaxis_title="Sample Size",
    yaxis_title="R² Score",
    template="plotly_white",
    hovermode="x unified",
    height=650
    )

    st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------
# CONCLUSION
# --------------------------------------------------

elif section == "Conclusion":

    st.header("Conclusion")

    st.write("""
Machine learning models do not necessarily outperform regression
when applied to structured survey datasets.

Model complexity must align with dataset structure and size.
""")