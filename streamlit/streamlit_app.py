import streamlit as st
import folium 
from streamlit_folium import st_folium
import pandas as pd
import plotly.graph_objects as go
import os
import pickle
import joblib
import numpy as np
from sklearn.linear_model import LinearRegression

# --- 1. SETUP & DATA LOADING ---
st.set_page_config(page_title="Groundwater Forecast - VA Eastern Shore", layout="wide")
current_directory = os.path.dirname(os.path.realpath(__file__))
data_folder = current_directory 

if 'selected_station' not in st.session_state:
    st.session_state.selected_station = 'Home'

well_map = {
    'withams': 'Withams (USGS 375723075344404)',
    'greenbush': 'Green Bush (USGS 374425075400003)',
    'churchneck': 'Church Neck (USGS 372705075555903)',
    'capecharles': 'Cape Charles (USGS 371543076003401)'
}

well_details = {
    'withams': {
        "id": "375723075344404", "coords": [37.9535, -75.4852],
        "about": "The USGS 375723075344404 (66M 19 SOW 110S) is a groundwater monitoring well in Accomack County, VA; Latitude / Longitude: 37.9566, -75.5787, with an approximate elevation of 10 ft and screened within the Northern Atlantic Coastal Plain aquifer system. This well screens the Surficial (Columbia) Aquifer, which is considered an unconfined system. The well has a total depth of 36 ft and penetrated Quaternary-aged sediments; the majority of the sedimentary composition of this area is unconsolidated fine- to coarse-grained sand, gravelly sand and very thin (less than 1 ft) lenses of silty clay and represents a coastal barrier-lagoon depositional environment."
    },
    'greenbush': {
        "id": "374425075400003", "coords": [37.7656, -75.6444],
        "about": "The USGS 374425075400003 (65K 29 SOW 114C) is a groundwater monitoring well in Accomack County, VA; Latitude / Longitude: 37.7401, -75.6655; with an elevation of 45.0 ft and screened within the Northern Atlantic Coastal Plain aquifer system. This well screens the Yorktown-Eastover Lower Aquifer, which is a single, confined aquifer system. The well is 315 ft deep (hole depth of 400 ft); the Pliocene-aged marine deposits have been reached at this depth. The primary lithologic unit at this depth is bluish-grey glauconitic sand, sandy gravel and abundant shell fragments. At this depth, the upper units are separated from the lower units by a thick (greater than 50 ft), impervious clay confining layer."
    },
    'churchneck': {
        "id": "372705075555903", "coords": [37.4583, -75.9406],
        "about": "The USGS 372705075555903 (63H 6 SOW 103A) is a groundwater monitoring well in Northampton County, VA; Latitude / Longitude: 37.4519, -75.9328; with an elevation of 16.6 ft and screened within the Northern Atlantic Coastal Plain aquifer system. This well screens the Surficial (Columbia) Aquifer, which is unconfined and highly susceptible to recharge from precipitation. The well is 37 ft deep; the top portion of the well contains Holocene and Pleistocene-age, estuarine-type sediments that transition from fine silty sand to coarser sand and gravelly beds at the bottom of the borehole."
    },
    'capecharles': {
        "id": "371543076003401", "coords": [37.2650, -76.0150],
        "about": "The USGS 371543076003401 (62G 15 SOW 121) is a groundwater monitoring well in Northampton County, VA; Latitude / Longitude: 37.2621, -76.0089; with an elevation of approximately 10 ft and screened within the Northern Atlantic Coastal Plain aquifer system. This well is screened in the Yorktown-Eastover Middle Aquifer, which is a confined system. The well is 190 ft deep (hole depth of 240 ft). The lithology of the sediments transition from the shallow surficial sands to Pliocene-age fossiliferous marine sediments, composed of well-sorted sand, silt and dense shell beds that represent substantial water storage but are also subject to saltwater intrusion because of their proximity to the Chesapeake Bay."
    }
}

@st.cache_data
def load_all_data():
    well_data, well_scores = {}, {}
    for w in well_map.keys():
        pred_path = os.path.join(data_folder, f'model_predictions_{w}.pkl')
        if os.path.exists(pred_path):
            with open(pred_path, 'rb') as f:
                df = pickle.load(f)
                if 'Date' in df.columns: df = df.rename(columns={'Date': 'date'})
                df['date'] = pd.to_datetime(df['date'])
                well_data[w] = df
        score_path = os.path.join(data_folder, f'model_scores_{w}.pkl')
        if os.path.exists(score_path):
            with open(score_path, 'rb') as f:
                scores_dict = pickle.load(f)
                df_s = pd.DataFrame.from_dict(scores_dict, orient='index').reset_index()
                df_s.columns = ['Model', 'R2', 'RMSE']
                well_scores[w] = df_s
    return well_data, well_scores

well_data, well_scores = load_all_data()

# --- 2. SIDEBAR NAVIGATION & PERSONAL INFO ---
st.sidebar.title("Well Locations")

st.sidebar.info("To see exact site locations, hover over the drops on the map. You can click a marker to navigate directly to that well's analysis.")

options = ['Home'] + list(well_map.keys())
selected_key = st.sidebar.selectbox(
    "Select Station:", 
    options, 
    index=options.index(st.session_state.selected_station),
    format_func=lambda x: well_map.get(x, "Overview")
)

st.sidebar.markdown("---")
st.sidebar.subheader("Contact & Developer Info")
st.sidebar.markdown("""
**Kedir Bushira, PhD** 
(Water Resources Engineer & Data Scientist) 

Email :[kdrmohammed@gmail.com](mailto:kdrmohammed@gmail.com)

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Profile-blue?style=flat&logo=linkedin)](https://www.linkedin.com/in/kedir-bushira/)
[![GitHub](https://img.shields.io/badge/GitHub-Repo-lightgrey?style=flat&logo=github)](https://github.com/Bushira1/chesapeake-Bay-groundwater-ML-Model/tree/main)
""")

if selected_key != st.session_state.selected_station:
    st.session_state.selected_station = selected_key
    st.rerun()

# --- 3. MAIN PAGE ---
if st.session_state.selected_station == 'Home':
    st.title("Groundwater Level Forecast: Virginia Eastern Shore")
    st.markdown("#### *This project was completed as part of the Eastern University MS in Data Science Capstone Project.*")
    
    st.write("---")
    m = folium.Map(location=[37.6, -75.7], zoom_start=9, tiles="cartodbpositron")
    for key, info in well_details.items():
        folium.Marker(
            location=info['coords'], 
            tooltip=f"Navigate to {key.capitalize()}: {info['id']}",
            icon=folium.Icon(color='blue', icon='info-sign'), 
            name=key
        ).add_to(m)
    
    map_data = st_folium(m, height=450, width="100%", key="main_map")
    
    # Click Navigation logic
    if map_data and map_data.get("last_object_clicked_tooltip"):
        clicked_text = map_data["last_object_clicked_tooltip"].lower()
        for key in well_map.keys():
            if key in clicked_text:
                st.session_state.selected_station = key
                st.rerun()

    st.divider()
    st.header("About the Project")
    st.write("""
    Groundwater is one of the most important resources that humans use to survive. In the Eastern Shore of Virginia, it is the sole resource used for human consumption (drinking) and crop irrigation, therefore managing groundwater is critical to both the residents living on the Shore and the agricultural community. The majority of both drinking and irrigation water is obtained from wells that tap into the Columbia and Yorktown-Eastover multi-aquifer system (Masterson et al., 2016).
    """)

    # 

    st.write("""
    In 1997, the U.S. Environmental Protection Agency (EPA) designated this area as a Sole Source Aquifer due to the lack of any large-scale fresh-surface-water streams to be used as an alternate source of water (U.S. EPA, 1997). The primary factors that affect groundwater levels in this area include extraction rates of groundwater; and geologic structures such as the buried paleo-channels that include the Exmore and Eastville ancient river channels which can both significantly impact groundwater flow and increase the risk of saltwater intrusion (Powars et al., 2010). 
             
    Previous studies have demonstrated that machine learning, deep learning, and time-series analysis have been successful in identifying the complex nonlinear patterns in hydrogeologic data. This project implemented and compared Multiple Linear Regression (MLR), a baseline model, with more advanced deep learning architectures, specifically Convolutional Neural Networks (CNN) and Long Short-Term Memory (LSTM) networks.
    """)

    st.write("""
    These models were developed to predict Groundwater Levels (GWL) in four significant USGS monitored wells:
    * **USGS 375525075304601** (Withams - Accomack County)
    * **USGS 374425075400003** (Greenbush - Accomack County)
    * **USGS 372705075555903** (Church Neck - Northampton County)
    * **USGS 371543076003401** (Cape Charles - Northampton County)

    The models were trained using a large comprehensive daily dataset of information from 2007-2025 and included precipitation lags, evaporation, soil temperature and soil moisture to provide multiple-day predictions, including a recursive multi-day forecast out to 2026.
    """)

else:
    # --- INDIVIDUAL WELL ANALYSIS ---
    current_well = st.session_state.selected_station
    st.title(well_map[current_well])
    
    # 1. About the Well
    st.subheader("About the Well")
    st.info(well_details[current_well]['about'])
    
    if st.button("⬅ Back to Regional Map"):
        st.session_state.selected_station = 'Home'
        st.rerun()

    if current_well in well_data:
        df = well_data[current_well]
        actual_col = 'Actual' if 'Actual' in df.columns else 'gw_depth_ft'
        available_models = [c.replace('_Pred', '') for c in df.columns if '_Pred' in c]

        # 2. FORECAST PLOT
        st.header("📈 Groundwater Level Forecast")
        selected_traces = st.multiselect('Select Models:', ['Actual'] + available_models, default=['Actual', 'LSTM'])
        fig = go.Figure()
        if 'Actual' in selected_traces:
            fig.add_trace(go.Scatter(x=df['date'], y=df[actual_col], name='Observed', line=dict(color='black')))
        
        colors = {'MLR': 'blue', 'CNN': 'green', 'LSTM': 'red'}
        for m in available_models:
            if m in selected_traces:
                c_n = f"{m}_Pred" if f"{m}_Pred" in df.columns else m
                fig.add_trace(go.Scatter(x=df['date'], y=df[c_n], name=m, line=dict(color=colors.get(m))))
        
        fig.update_layout(template="plotly_white", yaxis=dict(autorange="reversed", title="Depth (ft)"),
                          xaxis=dict(dtick="M12", tickformat="%Y", range=[df['date'].min(), "2026-12-31"]))
        
        st.plotly_chart(fig, use_container_width=True)

        # 3. Score Report
        if current_well in well_scores:
            st.subheader("Model Performance Scores")
            st.dataframe(well_scores[current_well], hide_index=True)

        # 4. ERROR TRENDS
        st.divider()
        st.header("📉 Error Trend Analysis (Residuals)")
        res_m = st.selectbox("Analyze Trend for:", available_models)
        res_col = f"{res_m}_Pred" if f"{res_m}_Pred" in df.columns else res_m
        res_df = df[['date', actual_col, res_col]].dropna().copy()
        res_df['Residual'] = res_df[actual_col] - res_df[res_col]
        
        X_trend = np.array(range(len(res_df))).reshape(-1, 1)
        trend_model = LinearRegression().fit(X_trend, res_df['Residual'].values)
        res_df['Trend_Line'] = trend_model.predict(X_trend)

        fig_res = go.Figure()
        fig_res.add_trace(go.Scatter(x=res_df['date'], y=res_df['Residual'], mode='markers', marker=dict(opacity=0.3), name='Error'))
        fig_res.add_trace(go.Scatter(x=res_df['date'], y=res_df['Trend_Line'], line=dict(color='orange', width=3), name='Trend'))
        fig_res.add_hline(y=0, line_dash="dash", line_color="black")
        fig_res.update_layout(template="plotly_white", xaxis=dict(dtick="M12", tickformat="%Y"), yaxis_title="Error (ft)")
        
        st.plotly_chart(fig_res, use_container_width=True)
