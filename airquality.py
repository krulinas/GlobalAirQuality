import streamlit as st
import pandas as pd
import plotly.express as px

# Load dataset
df = pd.read_csv("global_air.csv")

st.set_page_config(page_title="China vs France Air Pollution", layout="wide")
st.title("🌏 China vs France: Air Pollution Dashboard")

# --- Sidebar Filters ---
st.sidebar.header("Filter by:")
countries = ["China", "France"]
st.sidebar.write("**Countries fixed for comparison**: China 🇨🇳 & France 🇫🇷")

pollutants = {
    "Carbon Monoxide (CO)": "CO AQI Value",
    "Ozone (O3)": "Ozone AQI Value",
    "Nitrogen Dioxide (NO2)": "NO2 AQI Value",
    "Particulate Matter (PM2.5)": "PM2.5 AQI Value",
    "Overall AQI": "AQI Value"
}
selected_pollutant = st.sidebar.selectbox("Select Pollutant", list(pollutants.keys()))
filtered_df = df[df['Country'].isin(countries)]

# --- Main Section ---
st.subheader(f"{selected_pollutant} Levels in China & France")
pollutant_column = pollutants[selected_pollutant]

if not filtered_df.empty:
    fig = px.bar(
        filtered_df,
        x="City",
        y=pollutant_column,
        color="Country",
        barmode="group",
        labels={pollutant_column: f"{selected_pollutant} AQI"},
        title=f"City-wise {selected_pollutant} AQI Levels"
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("No data available for China or France.")

# --- Raw Data Viewer ---
with st.expander("🔍 View Raw Data"):
    st.dataframe(filtered_df)

# --- Dataset License Info ---
with st.expander("📄 Dataset & License Info"):
    st.markdown("""
**Dataset Title**: Global Air Pollution Dataset  
**Author**: Hasib Al Muzdadid  
**Source**: [Kaggle - Global Air Pollution Dataset](https://www.kaggle.com/datasets/hasibalmuzdadid/global-air-pollution-dataset)  
**License**: Publicly available via Kaggle for educational and research use only.  
*No commercial use or redistribution intended. All rights belong to the original creator.*
""")

# --- Footer ---
st.markdown("""
<hr>
<p style='text-align: center; color: gray; font-size: 0.85rem'>
Data © <a href='https://www.kaggle.com/datasets/hasibalmuzdadid/global-air-pollution-dataset' target='_blank'>Hasib Al Muzdadid</a> via Kaggle  
| Made with 💖 by Baby using Streamlit & Plotly ✨
</p>
""", unsafe_allow_html=True)