import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# --- Streamlit Config ---
st.set_page_config(page_title="Global Air Quality Dashboard", layout="wide")

# --- Inject Ragnarok Theme CSS ---
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

st.title("🌍 Global Air Quality Intelligence Dashboard")

# --- Load and preprocess dataset ---
df = pd.read_csv("global_air.csv")
features = ['CO AQI Value', 'Ozone AQI Value', 'NO2 AQI Value', 'PM2.5 AQI Value']
target = 'AQI Value'
df_clean = df[features + [target]].apply(pd.to_numeric, errors='coerce').dropna()

# --- ML Model ---
X = df_clean[features]
y = df_clean[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# --- Tabs ---
tab1, tab2, tab3, tab4 = st.tabs(["🌐 Overview", "📈 Prediction Tool", "📊 Country Comparison", "📺 Visual Explorer"])

# --- TAB 1: Overview ---
with tab1:
    st.markdown("## ⚡ Welcome to the *Ragnarok AQI Arena*")
    col1, col2, col3 = st.columns(3)
    col1.metric("🌬️ Avg CO AQI", round(df_clean['CO AQI Value'].mean(), 2))
    col2.metric("☀️ Avg Ozone AQI", round(df_clean['Ozone AQI Value'].mean(), 2))
    col3.metric("🧪 Avg NO2 AQI", round(df_clean['NO2 AQI Value'].mean(), 2))

    st.markdown("### 🧭 AQI by Country")
    fig_bar = px.bar(df[df['Country'].isin(["China", "France"])],
                     x='Country', y='AQI Value', color='Country', title="Country-wise AQI Distribution",
                     labels={'AQI Value': 'AQI Level'},
                     color_discrete_sequence=px.colors.sequential.Rainbow)
    fig_bar.update_layout(template='plotly_dark')
    st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("### 🌍 Geo-distribution of NO2 AQI")
    if 'Latitude' in df.columns and 'Longitude' in df.columns:
        geo_df = df.dropna(subset=['NO2 AQI Value', 'Latitude', 'Longitude'])
        fig_map = px.scatter_geo(geo_df, lat='Latitude', lon='Longitude', color='NO2 AQI Value',
                                 hover_name='City', title="City-wise NO2 AQI",
                                 color_continuous_scale="Turbo")
        fig_map.update_layout(template='plotly_dark')
        st.plotly_chart(fig_map, use_container_width=True)

# --- TAB 2: ML Prediction ---
with tab2:
    st.markdown("### 🔍 Predict AQI Based on Pollutants")
    st.sidebar.header("Input Pollutant Levels")
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

# --- TAB 3: Comparison ---
with tab3:
    st.markdown("### 🌐 Compare AQI Between Countries")
    selected_pollutant = st.selectbox("Select Pollutant for Comparison", features)
    selected_countries = st.multiselect("Choose Countries", df['Country'].unique(), default=["China", "France"])
    comp_df = df[df['Country'].isin(selected_countries)]
    fig_comp = px.box(comp_df, x='Country', y=selected_pollutant, color='Country',
                      title=f"{selected_pollutant} Distribution", template='plotly_dark')
    st.plotly_chart(fig_comp, use_container_width=True)

# --- TAB 4: Visual Explorer ---
with tab4:
    st.markdown("### 🎨 Advanced AQI Visualizations")
    selected_pollutant = st.selectbox("Select Pollutant", features, key="vis_pollutant")
    selected_visual = st.radio("Choose Visualization Type", ["Bar", "Line", "Box", "Area"])
    selected_countries = st.multiselect("Choose Countries", df['Country'].unique(), default=["China", "France"], key="vis_country")
    comp_df = df[df['Country'].isin(selected_countries)].copy()

    if selected_visual == "Bar":
        fig = px.bar(comp_df, x="Country", y=selected_pollutant, color="Country", barmode="group",
                     color_discrete_sequence=px.colors.sequential.Rainbow)
    elif selected_visual == "Line":
        fig = px.line(comp_df, x="City", y=selected_pollutant, color="Country")
    elif selected_visual == "Box":
        fig = px.box(comp_df, x="Country", y=selected_pollutant, color="Country")
    elif selected_visual == "Area":
        comp_df['Index'] = comp_df.groupby("Country").cumcount()
        fig = px.area(comp_df, x="Index", y=selected_pollutant, color="Country", line_group="Country")

    fig.update_layout(title=f"{selected_pollutant} by Country", yaxis_title="AQI Value", template='plotly_dark')
    st.plotly_chart(fig, use_container_width=True)

# --- Footer ---
st.markdown("""
<hr>
<p style='text-align: center; color: gray; font-size: 0.85rem'>
Data © <a href='https://www.kaggle.com/datasets/hasibalmuzdadid/global-air-pollution-dataset' target='_blank'>Hasib Al Muzdadid</a> via Kaggle  
| Styled by Thor: Ragnarok ✨ using Streamlit & Plotly
</p>
""", unsafe_allow_html=True)