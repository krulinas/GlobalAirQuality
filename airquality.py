import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

st.set_page_config(page_title="Global Air Quality Dashboard", layout="wide")

# Ragnarok Theme Color Palette
thor_colors = ['#2C3034', '#404B56', '#FAF0BF', '#FADF7F', '#CC0E1D', '#A2141A']

st.markdown("""
    <style>
        .stApp {
            background: linear-gradient(to right, #140e26, #26073a, #390b6f);
            color: white;
        }
        h1, h2, h3, .stTitle {
            color: #facc15 !important;
            text-shadow: 0 0 5px #facc15;
        }
        .element-container .stMetric {
            background-color: #1f1f3a;
            padding: 1em;
            border-radius: 1em;
            border: 1px solid #3b82f6;
            box-shadow: 0px 0px 10px #6366f1;
            text-align: center;
        }
        .stSelectbox, .stSlider, .stMultiselect {
            background-color: #1e293b;
            color: #facc15;
            border-radius: 10px;
            border: 1px solid #7c3aed;
        }
        .stSlider > div[data-baseweb="slider"] {
            background-color: #111827;
            border: 2px solid #9333ea;
            border-radius: 1em;
            padding: 1em;
            box-shadow: 0px 0px 5px #d946ef inset;
        }
        button[kind="secondary"] {
            background: #9333ea;
            border: none;
            color: white;
            box-shadow: 0 0 10px #e879f9;
        }
        .stPlotlyChart > div {
            background-color: rgba(0,0,0,0.2);
            padding: 10px;
            border-radius: 10px;
            border: 1px solid #4ade80;
        }
    </style>
""", unsafe_allow_html=True)

# Dashboard Title and Description
st.title("🌍 Global Air Quality Intelligence Dashboard")
st.markdown("""
### Overview
This dashboard provides an interactive platform to monitor and analyze global air quality levels, 
with a focus on countries such as **China** and **France**.

Key features include:
- 📡 Real-time AQI metrics
- 🤖 Machine learning-based AQI prediction
- 🌐 Country and pollutant-level comparisons
- 📊 Data visualizations for deeper insights

Use the tabs above to explore various dimensions of the dataset.
""")

# Load dataset
df = pd.read_csv("global_air.csv")
df = df[df['Country'].isin(['China', 'France'])]  # Only focus on China and France
features = ['CO AQI Value', 'Ozone AQI Value', 'NO2 AQI Value', 'PM2.5 AQI Value']
target = 'AQI Value'
df_clean = df[features + [target]].apply(pd.to_numeric, errors='coerce').dropna()

X = df_clean[features]
y = df_clean[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

tab1, tab2, tab3, tab4 = st.tabs(["🌐 Overview", "📈 Prediction Tool", "📊 Country Comparison", "📺 Visual Explorer"])

with tab1:
    col1, col2, col3 = st.columns(3)
    col1.metric("🌀 Avg CO AQI", round(df_clean['CO AQI Value'].mean(), 2))
    col2.metric("☀️ Avg Ozone AQI", round(df_clean['Ozone AQI Value'].mean(), 2))
    col3.metric("🧪 Avg NO2 AQI", round(df_clean['NO2 AQI Value'].mean(), 2))

    st.markdown("### ⚡ AQI by Country")
    fig_bar = px.bar(df, x='City', y='AQI Value', color='Country',
                     title="City-wise AQI in China & France",
                     labels={'AQI Value': 'AQI Level'},
                     color_discrete_sequence=thor_colors)
    fig_bar.update_layout(
        template='plotly_dark',
        xaxis_tickangle=-45,
        margin=dict(l=40, r=40, t=60, b=200),
        height=600
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("### 🌍 Geo-distribution of NO2 AQI")
    if 'Latitude' in df.columns and 'Longitude' in df.columns:
        geo_df = df.dropna(subset=['NO2 AQI Value', 'Latitude', 'Longitude'])
        fig_map = px.scatter_geo(geo_df, lat='Latitude', lon='Longitude', color='NO2 AQI Value',
                                 hover_name='City', title="City-wise NO2 AQI",
                                 color_continuous_scale="Turbo")
        fig_map.update_layout(template='plotly_dark')
        st.plotly_chart(fig_map, use_container_width=True)

with tab2:
    st.markdown("### 🔍 Predict AQI Based on Pollutants")
    st.sidebar.markdown("#### 🧪 **Input Pollutant Levels**")
    st.sidebar.markdown("""
    These sliders let you test different pollutant levels and see their impact on overall AQI.
    
    - **CO AQI**: Carbon Monoxide
    - **Ozone AQI**: Surface-level ozone
    - **NO₂ AQI**: Nitrogen Dioxide
    - **PM2.5 AQI**: Fine Particulate Matter
    
    🔮 Try adjusting them to simulate different air conditions in China or France.
    """)
    co_val = st.sidebar.slider("CO AQI Value", 0, 200, 50)
    o3_val = st.sidebar.slider("Ozone AQI Value", 0, 200, 50)
    no2_val = st.sidebar.slider("NO2 AQI Value", 0, 200, 50)
    pm25_val = st.sidebar.slider("PM2.5 AQI Value", 0, 200, 50)

    input_df = pd.DataFrame({
        'CO AQI Value': [co_val],
        'Ozone AQI Value': [o3_val],
        'NO2 AQI Value': [no2_val],
        'PM2.5 AQI Value': [pm25_val]
    })

    predicted_aqi = model.predict(input_df)[0]
    st.metric(label="🎯 Predicted AQI", value=round(predicted_aqi, 2))

    st.markdown("### 📉 Model Performance")
    y_pred = model.predict(X_test)
    performance_df = pd.DataFrame({"Actual AQI": y_test, "Predicted AQI": y_pred})
    fig = px.scatter(performance_df, x="Actual AQI", y="Predicted AQI", trendline="ols",
                     title="Actual vs Predicted AQI", template='plotly_dark')
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(f"**MSE:** {mean_squared_error(y_test, y_pred):.2f}")
    st.markdown(f"**R²:** {r2_score(y_test, y_pred):.4f}")

with tab3:
    st.markdown("### 🌐 Compare AQI Between Countries")
    selected_pollutant = st.selectbox("Select Pollutant for Comparison", features)
    selected_countries = ['China', 'France']  # Fixed countries only
    comp_df = df[df['Country'].isin(selected_countries)]
    fig_comp = px.box(comp_df, x='Country', y=selected_pollutant, color='Country',
                      title=f"{selected_pollutant} Distribution", template='plotly_dark',
                      color_discrete_sequence=thor_colors)
    st.plotly_chart(fig_comp, use_container_width=True)

with tab4:
    st.markdown("### 🎨 Advanced AQI Visualizations")
    selected_pollutant = st.selectbox("Select Pollutant", features, key="vis_pollutant")
    selected_visual = st.radio("Choose Visualization Type", ["Bar", "Line", "Box", "Area"])
    selected_countries = ['China', 'France']
    comp_df = df[df['Country'].isin(selected_countries)].copy()

    if selected_visual == "Bar":
        fig = px.bar(comp_df, x="City", y=selected_pollutant, color="Country",
                     color_discrete_sequence=thor_colors)
    elif selected_visual == "Line":
        fig = px.line(comp_df, x="City", y=selected_pollutant, color="Country")
    elif selected_visual == "Box":
        fig = px.box(comp_df, x="Country", y=selected_pollutant, color="Country")
    elif selected_visual == "Area":
        comp_df['Index'] = comp_df.groupby("Country").cumcount()
        fig = px.area(comp_df, x="Index", y=selected_pollutant, color="Country", line_group="Country")

    fig.update_layout(title=f"{selected_pollutant} by Country", yaxis_title="AQI Value", template='plotly_dark')
    st.plotly_chart(fig, use_container_width=True)

st.markdown("""
<hr>
<p style='text-align: center; color: gray; font-size: 0.85rem'>
Data © <a href='https://www.kaggle.com/datasets/hasibalmuzdadid/global-air-pollution-dataset' target='_blank'>Hasib Al Muzdadid</a> via Kaggle  
| Powered by ✨ using Streamlit & Plotly
</p>
""", unsafe_allow_html=True)
