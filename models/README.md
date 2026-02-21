Deep Learning-Based Groundwater Level Prediction

Author: Kedir Bushira 


Course: DTSC 691: Applied Data Science 

Project Overview
This project investigates the use of Multiple Linear Regression (MLR) as a baseline and deep learning architectures (CNN and LSTM) to forecast groundwater levels. The study focuses on the Virginia Eastern Shore, using a daily time-step dataset spanning from October 2007 through December 2025.

Target Monitoring Wells
The models predict depth for four USGS sites representing a north-to-south transect:

Withams: Northern Shore 
Green Bush: Central Shore 
Church Neck: Central Bayside 
Cape Charles: Southern Tip 

Project Structure
data/: Contains raw and processed CSVs from USGS and Open-Meteo.
notebooks/: Jupyter Notebooks for EDA, baseline modeling, and deep learning.
models/: Saved model files (MLR, CNN, LSTM).
requirements.txt: List of Python dependencies (TensorFlow, Pandas, etc.).
README.md: Project documentation.

Setup & Installation
Create a Virtual Environment:
python -m venv venv

Activate the Environment:
Windows: venv\Scripts\activate
Mac/Linux: source venv/bin/activate

Install Dependencies:
pip install -r requirements.txt

Methodology
Feature Engineering: Includes a 30-60 day lag for precipitation to account for delayed vertical infiltration.
Models:
MLR: Establishes the performance floor.
CNN: Captures periodic short-term patterns.
LSTM: Captures long-term temporal relationships in the aquifer system.
Evaluation: Performance is measured using RMSE and MAE.

Deployment
The final output is delivered via a Streamlit web application, allowing users to select a well and visualize model predictions.

Data Sources
Groundwater Data: USGS National Water Dashboard.
Climate Data: Open-Meteo API.