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

# --- Intro & Public-Focused Explanation ---
with st.expander("ℹ️ What is this Dashboard About?", expanded=True):
    st.markdown("""
This dashboard helps you **analyze and predict Air Quality Index (AQI)** in **China** and **France**, using real-world pollutant data.  
We focus on 4 key pollutants:
- **CO**: Carbon Monoxide (from vehicle fuel)
- **Ozone (O3)**: Surface-level ozone (from UV reactions)
- **NO2**: Nitrogen Dioxide (from car exhaust)
- **PM2.5**: Fine particles harmful to lungs

✨ You can:
- See **average pollution** by country and city
- Compare pollution levels in **China vs France**
- Use machine learning to predict AQI
- View interactive graphs 

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

    # --- Custom color map for fixed countries ---
    color_map = {'France': '#FF1E1E', 'China': '#FFD600'}
    fig_bar = px.bar(df_filtered, x='City', y='AQI Value', color='Country',
                     labels={'AQI Value': 'AQI Level'},
                     title="City-wise AQI in China & France",
                     color_discrete_map=color_map)

    fig_bar.update_layout(
        template="plotly_dark",
        xaxis_tickangle=45,
        height=600,
        margin=dict(b=200)
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # Optional: Show % of total AQI contributed by each country
    st.markdown("#### 📊 Percentage of AQI Contribution")
    total_aqi = df_filtered.groupby('Country')['AQI Value'].sum()
    percent_df = (total_aqi / total_aqi.sum() * 100).round(2).reset_index()
    for idx, row in percent_df.iterrows():
        emoji = "🇫🇷" if row['Country'] == "France" else "🇨🇳"
        color = color_map[row['Country']]
        st.markdown(f"<span style='color:{color}; font-size:1.1em'>{emoji} {row['Country']}: {row['AQI Value']}%</span>", unsafe_allow_html=True)

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

    # Sidebar section
    st.sidebar.header("🧪 Input Pollutant Levels")
    st.sidebar.markdown("""
     Use the sliders below to set the air pollutant levels (AQI).  
    The system will instantly predict the overall Air Quality Index based on your input.
    """)

    co_val = st.sidebar.slider("🌫️ CO AQI Value", 0, 200, 50)
    o3_val = st.sidebar.slider("☀️ Ozone AQI Value", 0, 200, 50)
    no2_val = st.sidebar.slider("🚗 NO2 AQI Value", 0, 200, 50)
    pm25_val = st.sidebar.slider("🧪 PM2.5 AQI Value", 0, 200, 50)

    # Prepare input
    input_df = pd.DataFrame({
        'CO AQI Value': [co_val],
        'Ozone AQI Value': [o3_val],
        'NO2 AQI Value': [no2_val],
        'PM2.5 AQI Value': [pm25_val]
    })

    # Predict AQI
    predicted_aqi = model.predict(input_df)[0]
    st.metric("🎯 Predicted AQI", round(predicted_aqi, 2))

    # 👁️ Show input summary
    st.markdown("### 🧾 Pollutant Levels Input")
    st.dataframe(input_df)

    # 🚦 AQI Category Interpretation (Responsive feedback)
    st.markdown("### 🩺 Air Quality Status")
    if predicted_aqi >= 150:
        st.error("⚠️ *Unhealthy* — High risk to everyone. Limit outdoor activity.")
    elif predicted_aqi >= 101:
        st.warning("😷 *Moderate to Unhealthy for Sensitive Groups*. Use precautions.")
    elif predicted_aqi >= 51:
        st.info("🌤️ *Moderate*. Acceptable but some risk for sensitive individuals.")
    else:
        st.success("✅ *Good*. Air quality is considered satisfactory.")

    # Model performance section
    st.subheader("📊 Model Performance")
    y_pred = model.predict(X_test)
    performance_df = pd.DataFrame({"Actual AQI": y_test, "Predicted AQI": y_pred})

    fig_perf = px.scatter(performance_df, x="Actual AQI", y="Predicted AQI",
                          title="Actual vs Predicted AQI", trendline="ols", template="plotly_dark")
    st.plotly_chart(fig_perf, use_container_width=True)

    st.markdown(f"**MSE (Mean Squared Error):** {mean_squared_error(y_test, y_pred):.2f}")
    st.markdown(f"**R² (Coefficient of Determination):** {r2_score(y_test, y_pred):.4f}")

# ============== TAB 3: COUNTRY COMPARISON ==============
with tab3:
    st.header("🌐 Compare AQI Between Countries")

    # User selection for pollutant and chart type
    selected_pollutant = st.selectbox("Select Pollutant for Comparison", features, key="comp_pollutant")
    vis_type_tab3 = st.radio("Select Visualization Type", ['Box', 'Bar'], horizontal=True, key="comp_chart_type")

    # Filtered data and custom colors
    comp_df = df_filtered.copy()
    color_map = {'France': '#FF1E1E', 'China': '#FFD600'}

    # Visualization logic
    if vis_type_tab3 == 'Box':
        fig = px.box(comp_df, x='Country', y=selected_pollutant, color='Country',
                     title=f"{selected_pollutant} - Box Plot Comparison",
                     color_discrete_map=color_map)
    elif vis_type_tab3 == 'Bar':
        agg_df = comp_df.groupby(['City', 'Country'], as_index=False)[selected_pollutant].mean()
        agg_df = agg_df.sort_values(by=selected_pollutant, ascending=False).head(20)
        fig = px.bar(agg_df, x='City', y=selected_pollutant, color='Country',
                     title=f"{selected_pollutant} - Top 20 City Comparison",
                     color_discrete_map=color_map)

    fig.update_layout(template="plotly_dark", xaxis_tickangle=45, height=600)
    st.plotly_chart(fig, use_container_width=True)

    # Statistical Summary Table
    st.markdown("### 📊 Statistical Summary by Country")
    summary = comp_df.groupby('Country')[selected_pollutant].agg(['mean', 'median', 'max', 'min']).round(2)
    st.dataframe(summary)

    # Top 5 Cities Table
    st.markdown("### 🏅 Top 5 Cities by AQI Level")
    top5 = comp_df[['City', 'Country', selected_pollutant]].sort_values(by=selected_pollutant, ascending=False).head(5)
    st.table(top5)

    # Smart Insight Section
    st.markdown("### 🧠 Insight")
    china_mean = summary.loc["China", "mean"]
    france_mean = summary.loc["France", "mean"]

    if china_mean > france_mean:
        insight = f"🇨🇳 **China** has a higher average {selected_pollutant} ({china_mean}) compared to 🇫🇷 France ({france_mean})."
    elif france_mean > china_mean:
        insight = f"🇫🇷 **France** has a higher average {selected_pollutant} ({france_mean}) compared to 🇨🇳 China ({china_mean})."
    else:
        insight = f"Both 🇨🇳 China and 🇫🇷 France have the same average {selected_pollutant} AQI: {china_mean}."

    st.success(insight)

    # Explanation Helper
    with st.expander("ℹ️ What does this chart show?"):
        st.markdown(f"""
This section visualizes **{selected_pollutant}** levels across cities in **China** and **France**.

- **Box Plot** shows distribution, median, and outliers per country.
- **Bar Chart** displays the top 20 cities with the highest average AQI.

📌 *AQI helps determine the potential health impact of air pollution.*
        """)

# ============== TAB 4: VISUAL EXPLORER ==============
with tab4:
    st.subheader("🎨 Advanced AQI Visualizations")

    pollutant = st.selectbox("Select Pollutant", features, key="pollutant")
    vis_type = st.radio("Choose Visualization Type", ['Bar', 'Line', 'Box', 'Area'])

    # Filter and aggregate top 30 for each country separately, then concat to ensure both are shown
    top_n = 15
    china_top = df_filtered[df_filtered['Country'] == 'China'].groupby(['City', 'Country'], as_index=False)[pollutant].mean().sort_values(by=pollutant, ascending=False).head(top_n)
    france_top = df_filtered[df_filtered['Country'] == 'France'].groupby(['City', 'Country'], as_index=False)[pollutant].mean().sort_values(by=pollutant, ascending=False).head(top_n)
    agg_df = pd.concat([china_top, france_top])

    # Custom color map for consistency
    custom_colors = {'France': '#FF1E1E', 'China': '#FFD600'}

    if vis_type == "Bar":
        fig = px.bar(agg_df, x='City', y=pollutant, color='Country',
                     color_discrete_map=custom_colors)
    elif vis_type == "Line":
        fig = px.line(agg_df, x='City', y=pollutant, color='Country',
                      color_discrete_map=custom_colors)
    elif vis_type == "Box":
        fig = px.box(df_filtered, x='Country', y=pollutant, color='Country',
                     color_discrete_map=custom_colors)
    elif vis_type == "Area":
        agg_df['Index'] = agg_df.groupby('Country').cumcount()
        fig = px.area(agg_df, x='Index', y=pollutant, color='Country',
                      line_group='Country', color_discrete_map=custom_colors)

    fig.update_layout(template="plotly_dark", xaxis_tickangle=45, height=600)
    st.plotly_chart(fig, use_container_width=True)

# --- Footer ---
st.markdown("""
<hr>
<p style='text-align: center; color: gray; font-size: 0.85rem'>
Data © <a href='https://www.kaggle.com/datasets/hasibalmuzdadid/global-air-pollution-dataset' target='_blank'>Hasib Al Muzdadid</a> via Kaggle  
| Powered by using Streamlit & Plotly | STTHK3033 Information Visualization
</p>
""", unsafe_allow_html=True)