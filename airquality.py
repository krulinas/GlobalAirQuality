import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# --- App Setup ---
st.set_page_config(page_title="Air Quality Dashboard", layout="wide")

# --- Load & Clean Data ---
df = pd.read_csv("global_air.csv")
features = ['CO AQI Value', 'Ozone AQI Value', 'NO2 AQI Value', 'PM2.5 AQI Value']
target = 'AQI Value'

df[features + [target]] = df[features + [target]].apply(pd.to_numeric, errors='coerce')
df['Country'] = df['Country']
df['City'] = df['City']
df_clean = df.dropna(subset=features + [target])
df_filtered = df_clean[df_clean['Country'].isin(['China', 'France'])]

# --- Train ML Model ---
X = df_clean[features]
y = df_clean[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# --- Header & Description ---
st.markdown("<h1 style='text-align:center; color: gold;'>🌍 Global Air Quality Intelligence Dashboard</h1>", unsafe_allow_html=True)
st.markdown("""
### ✨ About the Dashboard
This dashboard helps you **visualize and predict air quality** in **China** and **France** using real-world AQI data.

#### What's Inside:
- 📡 Real-time air pollution metrics
- 🎯 Predictive model powered by machine learning
- 🌍 Comparison of AQI between countries
- 📊 Interactive visualizations with multiple styles
""")

# --- Tabs ---
tab1, tab2, tab3, tab4 = st.tabs(["🏠 Overview", "📈 Prediction Tool", "🌍 Country Comparison", "🎨 Visual Explorer"])

# --- Tab 1: Overview ---
with tab1:
    st.subheader("⚡ AQI Summary")
    col1, col2, col3 = st.columns(3)
    col1.metric("📘 Avg CO AQI", round(df_filtered['CO AQI Value'].mean(), 2))
    col2.metric("🟠 Avg Ozone AQI", round(df_filtered['Ozone AQI Value'].mean(), 2))
    col3.metric("🧪 Avg NO2 AQI", round(df_filtered['NO2 AQI Value'].mean(), 2))

    st.subheader("📊 AQI by City (China & France)")
    fig_bar = px.bar(df_filtered, x='City', y='AQI Value', color='Country',
                     title="City-wise AQI in China & France",
                     color_discrete_sequence=["#CC0E1D", "#404B56"])
    fig_bar.update_layout(template="plotly_dark", xaxis_tickangle=45, height=600)
    st.plotly_chart(fig_bar, use_container_width=True)

    st.subheader("🌍 Geo-distribution of NO2 AQI")
    if 'Latitude' in df.columns and 'Longitude' in df.columns:
        geo_df = df_filtered.dropna(subset=['NO2 AQI Value', 'Latitude', 'Longitude'])
        fig_map = px.scatter_geo(geo_df, lat='Latitude', lon='Longitude', color='NO2 AQI Value',
                                 hover_name='City', title="City-wise NO2 AQI",
                                 color_continuous_scale="Turbo")
        st.plotly_chart(fig_map, use_container_width=True)

# --- Tab 2: Prediction Tool ---
with tab2:
    st.subheader("🎯 Predict AQI Based on Pollutants")
    st.sidebar.header("🧪 Enter Pollutant Levels")
    co_val = st.sidebar.slider("CO AQI", 0, 200, 50)
    o3_val = st.sidebar.slider("Ozone AQI", 0, 200, 50)
    no2_val = st.sidebar.slider("NO2 AQI", 0, 200, 50)
    pm25_val = st.sidebar.slider("PM2.5 AQI", 0, 200, 50)

    input_df = pd.DataFrame({
        'CO AQI Value': [co_val],
        'Ozone AQI Value': [o3_val],
        'NO2 AQI Value': [no2_val],
        'PM2.5 AQI Value': [pm25_val]
    })
    prediction = model.predict(input_df)[0]
    st.metric("Predicted AQI", round(prediction, 2))

    st.subheader("📈 Model Performance")
    y_pred = model.predict(X_test)
    fig_pred = px.scatter(x=y_test, y=y_pred, labels={'x': 'Actual AQI', 'y': 'Predicted AQI'},
                          title="Actual vs Predicted AQI", template="plotly_dark")
    st.plotly_chart(fig_pred, use_container_width=True)
    st.markdown(f"**MSE:** {mean_squared_error(y_test, y_pred):.2f}")
    st.markdown(f"**R²:** {r2_score(y_test, y_pred):.4f}")

    st.subheader("📌 Heatmap of AQI Feature Correlations")
    fig_heatmap, ax = plt.subplots(figsize=(5, 3))
    sns.heatmap(df_clean[features].corr(), annot=True, cmap="YlOrRd", ax=ax)
    st.pyplot(fig_heatmap)

# --- Tab 3: Country Comparison ---
with tab3:
    st.subheader("🌍 Compare AQI Metrics Between China and France")
    selected = st.selectbox("Select Pollutant", features)
    fig_box = px.box(df_filtered, x='Country', y=selected, color='Country',
                     title=f"{selected} Distribution by Country",
                     color_discrete_sequence=["#CC0E1D", "#404B56"])
    fig_box.update_layout(template="plotly_dark")
    st.plotly_chart(fig_box, use_container_width=True)

# --- Tab 4: Visual Explorer ---
with tab4:
    st.subheader("🎨 Advanced AQI Visualizations")
    pollutant = st.selectbox("Select Pollutant", features, key="visual_pollutant_selector")
    vis_type = st.radio("Choose Visualization Type", ['Bar', 'Line', 'Box', 'Area'])

    top_n = 25
    agg_df = df_filtered.groupby(['City', 'Country'], as_index=False)[pollutant].mean()
    agg_df = agg_df.sort_values(by=pollutant, ascending=False).head(top_n)

    if vis_type == "Bar":
        fig = px.bar(agg_df, x='City', y=pollutant, color='Country',
                     color_discrete_sequence=["#CC0E1D", "#404B56"])
    elif vis_type == "Line":
        fig = px.line(agg_df, x='City', y=pollutant, color='Country',
                      color_discrete_sequence=["#CC0E1D", "#404B56"])
    elif vis_type == "Box":
        fig = px.box(df_filtered, x='Country', y=pollutant, color='Country',
                     color_discrete_sequence=["#CC0E1D", "#404B56"])
    elif vis_type == "Area":
        fig = px.area(agg_df, x='City', y=pollutant, color='Country',
                      color_discrete_sequence=["#CC0E1D", "#404B56"])

    fig.update_layout(template="plotly_dark", xaxis_tickangle=45, height=600)
    st.plotly_chart(fig, use_container_width=True)

# --- Footer ---
st.markdown("""
<hr>
<p style='text-align: center; font-size: 0.85rem; color: gray;'>
Data © <a href='https://www.kaggle.com/datasets/hasibalmuzdadid/global-air-pollution-dataset' target='_blank'>Hasib Al Muzdadid</a> | Dashboard powered by Streamlit & Plotly ⚡
</p>
""", unsafe_allow_html=True)
