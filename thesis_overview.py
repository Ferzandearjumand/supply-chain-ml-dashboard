import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import time
from scipy.signal import savgol_filter

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="Supply Chain ML Thesis Companion",
    layout="wide"
)

st.title("Predicting Supply Chain Performance Using Advanced Machine Learning: An Empirical Analysis of Key Operational Drivers")

st.markdown("""
This dashboard summarises the key theoretical framework, empirical analysis,
and predictive modelling results presented in the thesis.
""")

st.divider()

# --------------------------------------------------
# NAVIGATION
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

    st.header("Introduction")

    st.markdown("""
Supply Chain Performance has become a central determinant of organisational competitiveness
in modern business environments characterised by uncertainty and disruption.

This study investigates how operational capabilities influence supply chain performance
and whether advanced machine learning models provide superior predictive performance
compared with traditional regression models.
""")

    st.subheader("Research Questions")

    st.markdown("""
• How do supply chain capabilities influence supply chain performance?

• Which operational capabilities have the strongest impact?

• Do machine learning models outperform traditional regression models
when predicting supply chain performance using survey-based data?
""")

    st.subheader("Research Objectives")

    st.markdown("""
1. Examine the effect of supply chain capabilities on performance.

2. Identify the relative importance of operational drivers.

3. Compare predictive performance of regression and machine learning models.

4. Evaluate whether greater model complexity produces meaningful prediction gains.
""")

# --------------------------------------------------
# RESEARCH MODEL
# --------------------------------------------------

elif section == "Research Model":

    st.header("Conceptual Research Model")

    st.image("research model 25Jan2026.jpg")

    st.markdown("""
### Independent Variables

**Supply Chain Integration (SCI)**  
Internal, supplier, and customer integration.

**Information Sharing (IS)**  
Accuracy, timeliness, and relevance of information exchange.

**Supply Chain Agility (SCA)**  
Flexibility and responsiveness of the supply chain.

**Supply Chain Risk Management (SCRM)**  
Identification, assessment, and mitigation of supply chain risks.

### Dependent Variable

**Supply Chain Performance (SCP)**  
Measured through operational and financial performance outcomes.
""")

# --------------------------------------------------
# DATA
# --------------------------------------------------

elif section == "Data & Variables":

    st.header("Dataset and Variables")

    st.markdown("""
The empirical analysis is based on survey data collected from **222 organisations**.

The dataset captures managerial perceptions of supply chain capabilities and performance
using structured measurement scales.
""")

    st.markdown("""
### Data Preparation

• Data cleaning and validation  
• Aggregation of survey indicators  
• Construction of composite variables
""")

    st.markdown("""
### Statistical Diagnostics

• Reliability testing using Cronbach Alpha  
• Correlation analysis  
• Multicollinearity diagnostics (VIF)
""")

    st.image("Figure_4_Explanatory_Overview.svg")

# --------------------------------------------------
# MODELS
# --------------------------------------------------

elif section == "Models":

    st.header("Predictive Models")

    st.markdown("""
### Multiple Linear Regression
Baseline explanatory model widely used in supply chain research.

### Random Forest
Tree-based ensemble model capable of capturing nonlinear interactions.

### XGBoost
Gradient boosting algorithm known for strong predictive accuracy.

### CatBoost
Boosting model designed for efficient handling of structured data.

### Artificial Neural Network
Multilayer model capable of approximating complex nonlinear relationships.
""")

# --------------------------------------------------
# RESULTS
# --------------------------------------------------

elif section == "Results":

    st.header("Model Results")

    st.image("Table_4_11_APA.svg")

    st.markdown("""
The results indicate that traditional regression performs competitively
with machine learning models.

This suggests that greater algorithmic complexity does not necessarily
produce superior predictive performance when working with structured
survey datasets.
""")

    st.image("Figure_5_1_Ensemble_Feature_Importance.svg")

    st.image("Figure_5_2_Cross_Model_Consistency.svg")

# --------------------------------------------------
# LEARNING CURVES
# --------------------------------------------------

elif section == "Learning Curves":

    st.header("Model Learning Behaviour")

    st.markdown("""
### Purpose of Learning Curves

Learning curves illustrate how model predictive performance evolves as the
training dataset becomes larger.

In predictive modelling, increasing the number of observations generally improves
model accuracy because the algorithms learn more stable patterns from the data.
However, this improvement typically slows down once sufficient information
has been captured.

### What This Dashboard Demonstrates

The interactive analysis below shows how the predictive performance of each model
changes as the number of responses increases.

Key insights:

• Model accuracy improves rapidly when the dataset grows from small sample sizes.  
• After a certain point, the improvement slows and the models approach a **stability plateau**.  
• All models exhibit a broadly similar learning pattern, indicating consistent
learning behaviour across algorithms.

### How to Use the Controls

• The **sample size slider** allows simulation of model performance at different dataset sizes.  
• The **animation option** visualises how model accuracy evolves progressively.  
• The **stable learning region option** highlights the portion of the learning curve
where model performance stabilises.
""")

    df = pd.read_csv("learning_curve_results.csv")

    colors = {
    "LR": "#1f77b4",
    "RF": "#ff7f0e",
    "XGB": "#2ca02c",
    "CatBoost": "#d62728",
    "ANN": "#9467bd"
    }

    models = ["LR","RF","XGB","CatBoost","ANN"]

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

    st.subheader("Current Model Performance")

    col1, col2, col3 = st.columns([1.2,2,2])

    with col1:

        ranking = latest.drop(columns=["sample_size"]).T
        ranking.columns = ["R² Score"]
        ranking = ranking.round(3)
        ranking = ranking.sort_values("R² Score", ascending=False)

        st.dataframe(ranking, use_container_width=True)

    with col2:

        fig_dot = go.Figure()

        fig_dot.add_trace(go.Scatter(
            x=models,
            y=[round(v,3) for v in values],
            mode="markers",
            marker=dict(size=16, color=[colors[m] for m in models])
        ))

        fig_dot.update_layout(
            title=f"Model Performance at n = {current_n}",
            yaxis_title="R² Score",
            height=350
        )

        st.plotly_chart(fig_dot, use_container_width=True)

    with col3:

        fig_bar = go.Figure()

        fig_bar.add_trace(go.Bar(
            x=models,
            y=[round(v,3) for v in values],
            marker_color=[colors[m] for m in models]
        ))

        fig_bar.update_layout(
            title="Performance Comparison",
            yaxis_title="R² Score",
            height=350
        )

        st.plotly_chart(fig_bar, use_container_width=True)

    st.subheader("Learning Curves (Smoothed)")

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

    if stable_region:
        smooth_df = smooth_df[smooth_df["sample_size"] >= 80]

    fig = go.Figure()

    for model in models:

        fig.add_trace(go.Scatter(
            x=smooth_df["sample_size"],
            y=smooth_df[model],
            mode="lines",
            name=model,
            line=dict(color=colors[model], width=3)
        ))

    fig.add_vline(
    x=current_n,
    line_width=2,
    line_dash="dash",
    line_color="black"
    )

    fig.update_layout(
    title="Learning Curves",
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

    st.markdown("""
The study demonstrates that adaptive supply chain capabilities play a
critical role in determining organisational performance.

Supply Chain Agility and Risk Management emerge as the most influential
drivers of performance, while integration provides the structural
foundation that enables effective coordination.

The comparison of modelling approaches shows that advanced machine
learning models do not necessarily outperform traditional regression
when applied to structured survey datasets.

These findings highlight that organisational capabilities and managerial
decision processes remain central to supply chain performance, even in
an era of increasing analytical sophistication.
""")
