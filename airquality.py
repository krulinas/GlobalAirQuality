import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# --- Streamlit Config ---
st.set_page_config(page_title="Global Air Quality Dashboard", layout="wide")

# --- Thor Ragnarok Theme Color Palette ---
thor_colors = ['#2C3034', '#404B56', '#FAF0BF', '#FADF7F', '#CC0E1D', '#A2141A']

# --- Custom Styling ---
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
    .stMetric {
        background-color: #1f1f3a;
        padding: 1em;
        border-radius: 1em;
        border: 1px solid #3b82f6;
        box-shadow: 0px 0px 10px #6366f1;
        text-align: center;
    }
    .stSelectbox, .stSlider, .stRadio {
        background-color: #1e293b !important;
        color: #facc15 !important;
    }
    .stPlotlyChart > div {
        background-color: rgba(0,0,0,0.2);
        padding: 10px;
        border-radius: 10px;
        border: 1px solid #4ade80;
    }
    </style>
""", unsafe_allow_html=True)

# --- Dashboard Title ---
st.title("🌍 Global Air Quality Intelligence Dashboard")
st.markdown("#### ⚡ Welcome to the *Ragnarok AQI Arena*")

# --- Intro & Public-Focused Explanation ---
with st.expander("ℹ️ What is this Dashboard About?", expanded=True):
    st.markdown("""
This dashboard helps you **analyze and predict Air Quality Index (AQI)** in **China** and **France**, using real-world pollutant data.  
We focus on four key pollutants:
- **CO**: Carbon Monoxide (from vehicle fuel)
- **Ozone (O3)**: Surface-level ozone (from UV reactions)
- **NO2**: Nitrogen Dioxide (from car exhaust)
- **PM2.5**: Fine particles harmful to lungs

✨ You can:
- See **average pollution** by country and city
- Compare pollution levels in **China vs France**
- Use **machine learning** to predict AQI
- View **interactive graphs** (bar, line, box, area)

Simple enough for the public. Powerful enough for analysts.
""", unsafe_allow_html=True)

# --- Load Dataset ---
df = pd.read_csv("global_air.csv")
df = df[df['Country'].isin(['China', 'France'])]  # Focus only on China and France

# --- Data Cleaning ---
features = ['CO AQI Value', 'Ozone AQI Value', 'NO2 AQI Value', 'PM2.5 AQI Value']
target = 'AQI Value'
df_clean = df[features + [target]].apply(pd.to_numeric, errors='coerce').dropna()

df_filtered = df.dropna(subset=features)

# --- Train ML Model ---
X = df_clean[features]
y = df_clean[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# --- Tabs ---
tab1, tab2, tab3, tab4 = st.tabs(["🏠 Overview", "📈 Prediction Tool", "🌐 Country Comparison", "🎨 Visual Explorer"])

# ============== TAB 1: OVERVIEW ==============
with tab1:
    col1, col2, col3 = st.columns(3)
    col1.metric("📘 Avg CO AQI", round(df_clean['CO AQI Value'].mean(), 2))
    col2.metric("🟠 Avg Ozone AQI", round(df_clean['Ozone AQI Value'].mean(), 2))
    col3.metric("🧪 Avg NO2 AQI", round(df_clean['NO2 AQI Value'].mean(), 2))

    st.markdown("### 🌟 AQI by Country")
    fig_bar = px.bar(df_filtered, x='City', y='AQI Value', color='Country',
                     labels={'AQI Value': 'AQI Level'},
                     title="City-wise AQI in China & France",
                     color_discrete_sequence=thor_colors)
    fig_bar.update_layout(template="plotly_dark", xaxis_tickangle=45, height=600, margin=dict(b=200))
    st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("### 🌍 Geo-distribution of NO2 AQI")
    if 'Latitude' in df.columns and 'Longitude' in df.columns:
        geo_df = df_filtered.dropna(subset=['NO2 AQI Value', 'Latitude', 'Longitude'])
        fig_map = px.scatter_geo(geo_df, lat='Latitude', lon='Longitude', color='NO2 AQI Value',
                                 hover_name='City', title="City-wise NO2 AQI",
                                 color_continuous_scale="Turbo")
        fig_map.update_layout(template='plotly_dark')
        st.plotly_chart(fig_map, use_container_width=True)

# ============== TAB 2: PREDICTION TOOL ==============
with tab2:
    st.header("🎯 Predict AQI Based on Pollutants")

    st.sidebar.header("🧪 Input Pollutant Levels")
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
    st.metric("🎯 Predicted AQI", round(predicted_aqi, 2))

    st.subheader("📊 Model Performance")
    y_pred = model.predict(X_test)
    performance_df = pd.DataFrame({"Actual AQI": y_test, "Predicted AQI": y_pred})

    fig_perf = px.scatter(performance_df, x="Actual AQI", y="Predicted AQI",
                          title="Actual vs Predicted AQI", trendline="ols", template="plotly_dark")
    st.plotly_chart(fig_perf, use_container_width=True)

    st.markdown(f"**MSE:** {mean_squared_error(y_test, y_pred):.2f}")
    st.markdown(f"**R²:** {r2_score(y_test, y_pred):.4f}")

# ============== TAB 3: COUNTRY COMPARISON ==============
with tab3:
    st.header("🌐 Compare AQI Between Countries")
    selected_pollutant = st.selectbox("Select Pollutant for Comparison", features)
    fig_comp = px.box(df_filtered, x='Country', y=selected_pollutant, color='Country',
                      title=f"{selected_pollutant} Distribution",
                      color_discrete_sequence=thor_colors)
    fig_comp.update_layout(template="plotly_dark")
    st.plotly_chart(fig_comp, use_container_width=True)

# ============== TAB 4: VISUAL EXPLORER ==============
with tab4:
    st.subheader("🎨 Advanced AQI Visualizations")

    pollutant = st.selectbox("Select Pollutant", features, key="pollutant")
    vis_type = st.radio("Choose Visualization Type", ['Bar', 'Line', 'Box', 'Area'])

    top_n = 30
    agg_df = df_filtered.groupby(['City', 'Country'], as_index=False)[pollutant].mean()
    agg_df = agg_df.sort_values(by=pollutant, ascending=False).head(top_n)

    if vis_type == "Bar":
        fig = px.bar(agg_df, x='City', y=pollutant, color='Country',
                     color_discrete_sequence=thor_colors)
    elif vis_type == "Line":
        fig = px.line(agg_df, x='City', y=pollutant, color='Country')
    elif vis_type == "Box":
        fig = px.box(df_filtered, x='Country', y=pollutant, color='Country')
    elif vis_type == "Area":
        agg_df['Index'] = agg_df.groupby('Country').cumcount()
        fig = px.area(agg_df, x='Index', y=pollutant, color='Country', line_group='Country')

    fig.update_layout(template="plotly_dark", xaxis_tickangle=45, height=600)
    st.plotly_chart(fig, use_container_width=True)

# --- Footer ---
st.markdown("""
<hr>
<p style='text-align: center; color: gray; font-size: 0.85rem'>
Data © <a href='https://www.kaggle.com/datasets/hasibalmuzdadid/global-air-pollution-dataset' target='_blank'>Hasib Al Muzdadid</a> via Kaggle  
| Styled by Thor: Ragnarok ⚡ | Built with ❤️ using Streamlit & Plotly
</p>
""", unsafe_allow_html=True)