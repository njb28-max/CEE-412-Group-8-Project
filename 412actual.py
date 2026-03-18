import numpy as np
import pandas as pd
import altair as alt
import streamlit as st
# import geopandas as gpd
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from shapely.geometry import Point
from shapely.geometry import Point, LineString

# --- Opener ---
st.title('Loop Detectives of the Bellevue Commute')
st.caption("Tennessee Herrin, Erik Salvatier, Daniel Lucid, Nolan Bone")

st.header("Summary")
st.write("For this project, we explored Dataset 3 prepared by the teaching team. This dataset gives two main datasets for two loop detectors stationed on I-5 in Seattle and SR-520 in Bellevue. The loop detectors record the weekday velocity, flow, and occupancy over 5 minute intervals over the year of 2015. Our group chose this dataset as we were interested in commuter travel between Bellevue and Seattle and comparing the two datasets to find relationships.")

st.subheader("Question")
st.markdown("**How does traffic vary depending on hour of day, day of week and/or month and how do those patterns differ between I-5 and SR-520?**")
st.markdown("**Can you predict traffic patterns for one route based on the other?**")

st.subheader("Key Points")
st.markdown("""
- **Separated Datasets:** The dataset for I-5 and SR-520 are separated along with date and time, resulting in a severe limitation when comparing datasets.
- **Direction of traffic:** Using the provided WSDOT "Data Station Reference Guide", the dataset captures southbound I-5 traffic and westbound SR-520 traffic.
- **Weekend traffic excluded:** With the given dataset, weekends were never captured. When exploring our created database this had to be considered both writing the database and recognizing patterns.
- **Limited dataset:** The available data only gives a dataset for I-5 and SR-520 with each dataset at one point in time from a system that consists of over 800 loop detectors.
""")

# --- Import data ---
rawI5 = pd.read_csv("CEE412Project/005es16732_loop_cloutput.csv")
rawSR520 = pd.read_csv("CEE412Project/520es00972_loop_cloutput.csv")
SR520 = pd.read_excel("CEE412Project/520es00972_MW___3_MoTuWeThFr_2015-01-01_2015-12-31.xlsx", sheet_name="Volume")

# --- Setting up data ---
rawI5["DateTime"] = pd.to_datetime(rawI5["DateTime"])
rawI5 = rawI5[rawI5["DateTime"].dt.dayofweek < 5]
rawSR520["DateTime"] = pd.to_datetime(rawSR520["DateTime"])
rawSR520 = rawSR520[rawSR520["DateTime"].dt.dayofweek < 5]

# -----------------------------------------------------------------------------------------
# --- Map ---
st.header("Loop Detector Locations")

sensor_data = pd.DataFrame({
    "Sensor": ["I-5 Loop Detector", "SR-520 Loop Detector"],
    "Latitude": [47.63629, 47.639234],
    "Longitude": [-122.3234, -122.136005]
})

geometry = [Point(xy) for xy in zip(sensor_data["Longitude"], sensor_data["Latitude"])]
# gdf = gpd.GeoDataFrame(sensor_data, geometry=geometry, crs="EPSG:4326")
# st.map(gdf.rename(columns={"Latitude": "lat", "Longitude": "lon"}))

# -----------------------------------------------------------------------------------------
# --- E/R Diagram ---
st.header("E/R Diagram")
st.image("CEE412Project/ERDiagram.png")

# -----------------------------------------------------------------------------------------
# --- Data Showcase ---
# Summary
st.header("Database Showcase")

# Dataset
st.subheader("SR-520 Example")
Example = rawSR520.copy().drop(columns=["Unnamed: 0"])

toggle = st.toggle("Converted database")
if toggle:
    st.dataframe(Example.head(5))
else:
    st.dataframe(SR520[['Unnamed: 0', '2015-01-01', '2015-01-02', '2015-01-05', '2015-01-06', '2015-01-07', '2015-01-08']].head(5))

# Summary
col1, col2 = st.columns(2)
with col1:
    st.subheader("I-5 Summary")
    rawI5_summary = rawI5.copy()
    rawI5_summary = rawI5_summary.drop(columns=["Unnamed: 0", "DateTime"])
    st.dataframe(rawI5_summary.describe(), use_container_width=True)
with col2:
    st.subheader("SR-520 Summary")
    rawSR520_summary = rawSR520.copy()
    rawSR520_summary = rawSR520_summary.drop(columns=["Unnamed: 0", "DateTime"])
    st.dataframe(rawSR520_summary.describe(), use_container_width=True)

# -----------------------------------------------------------------------------------------
# --- Compare PHV ---
st.header("Comparing peak hour volume")

# Unified processing function
def process_sensor(df, sensor_name, freq="D", add_month_day=False):
    df["DateTime"] = pd.to_datetime(df["DateTime"])
    df = df.set_index("DateTime")
    
    # Resample volume (daily or weekly)
    resampled = df.resample(freq)["Volume"].sum().reset_index()
    
    # Remove weekends
    if freq == "D":
        resampled = resampled[resampled["DateTime"].dt.dayofweek < 5]  # 0=Mon, 4=Fri
    
    resampled["Highway"] = sensor_name
    
    # Add Month/Day for daily charts
    if add_month_day:
        resampled["Month"] = resampled["DateTime"].dt.month_name()
        resampled["Day"] = resampled["DateTime"].dt.day
        month_order = ["January","February","March","April","May","June",
                       "July","August","September","October","November","December"]
        resampled["Month"] = pd.Categorical(resampled["Month"], categories=month_order, ordered=True)
    
    return resampled

# Process both sensors
daily_I5 = process_sensor(rawI5, "I-5", freq="D", add_month_day=True)
daily_SR520 = process_sensor(rawSR520, "SR-520", freq="D", add_month_day=True)
combined_daily = pd.concat([daily_I5, daily_SR520], ignore_index=True)

weekly_I5 = process_sensor(rawI5, "I-5", freq="W-MON")
weekly_SR520 = process_sensor(rawSR520, "SR-520", freq="W-MON")
combined_weekly = pd.concat([weekly_I5, weekly_SR520], ignore_index=True)

# Streamlit widget for daily chart
selected_month = st.selectbox("Select a month to view:", combined_daily["Month"].cat.categories)
filtered_daily = combined_daily[combined_daily["Month"] == selected_month]

# Daily chart
daily_chart = alt.Chart(filtered_daily).mark_line(interpolate='basis').encode(
    x=alt.X("Day:Q", scale=alt.Scale(domain=[1, 31]), title='Day of Month'),
    y=alt.Y("Volume:Q", scale=alt.Scale(domain=[0, 20000]), title='Daily Volume'),
    color="Highway:N",
    tooltip=["Highway","Day","Volume"]
).properties(
    title=f"Daily Volume for {selected_month}"
)

# Weekly chart
weekly_chart = alt.Chart(combined_weekly).mark_line(interpolate='basis').encode(
    x=alt.X('DateTime:T', title='Month'),
    y=alt.Y('Volume:Q', title='Weekly Volume'),
    color="Highway:N",
    tooltip=["Highway","DateTime","Volume"]
).properties(
    title="Weekly Volume Comparison"
)

st.altair_chart(daily_chart, use_container_width=True)
st.altair_chart(weekly_chart, use_container_width=True)

# Peak timestamp in minutes Function
def daily_peak_end_time(df, sensor_name):
    df["DateTime"] = pd.to_datetime(df["DateTime"])
    df = df.set_index("DateTime")
    
    # Timestamp of max volume per day
    daily_peak = df.loc[df.groupby(df.index.date)['Volume'].idxmax()][['Volume']].copy()
    daily_peak['Date'] = daily_peak.index.date
    daily_peak['Highway'] = sensor_name
    
    # Minutes from midnight
    daily_peak['Peak_End_Minutes'] = daily_peak.index.hour * 60 + daily_peak.index.minute
    
    return daily_peak.reset_index(drop=True)

# Combined daily peak end times
i5_daily_peak_time = daily_peak_end_time(rawI5, "I-5")
sr520_daily_peak_time = daily_peak_end_time(rawSR520, "SR-520")
combined_peak_times = pd.concat([i5_daily_peak_time, sr520_daily_peak_time], ignore_index=True)

# Daily Peak Hour End Times plot
endtime_chart = alt.Chart(combined_peak_times).mark_line(interpolate='basis').encode(
    x=alt.X("Date:T", title="Month"),
    y=alt.Y(
        "Peak_End_Minutes:Q",
        title="Peak End Time",
        scale=alt.Scale(domain=[0, 24*60]),  # 0 to 1440 minutes
        axis=alt.Axis(
            values=[t*60 for t in range(0, 25, 2)],  # every 2 hours
            labelExpr="datum.value / 60 < 10 ? '0' + datum.value/60 + ':00' : datum.value/60 + ':00'"
        )
    ),
    color=alt.Color("Highway:N", title="Highway"),
    tooltip=["Date:T", "Highway:N", "Peak_End_Minutes:Q", "Volume:Q"]
).properties(
    title="Daily Peak 60-Minute Volume End Times Comparison"
)

st.altair_chart(endtime_chart, use_container_width=True)

# -----------------------------------------------------------------------------------------
# --- Time of day analysis ---
st.header("Time of Day Analysis")

# I-5 hourly
rawI5 = rawI5.set_index('DateTime')

I5_average_volume_by_time = (
    rawI5.groupby(rawI5.index.time)['Volume']
    .mean()
    .reset_index()
)
I5_average_volume_by_time.columns = ['Time_of_Day', 'Average_Volume']

I5_average_volume_by_time['Time_of_Day_str'] = I5_average_volume_by_time['Time_of_Day'].apply(lambda t: t.strftime('%H:%M'))

# SR-520 hourly
rawSR520 = rawSR520.set_index('DateTime')

SR520_average_volume_by_time = (
    rawSR520.groupby(rawSR520.index.time)['Volume']
    .mean()
    .reset_index()
)
SR520_average_volume_by_time.columns = ['Time_of_Day', 'Average_Volume']

SR520_average_volume_by_time['Time_of_Day_str'] = SR520_average_volume_by_time['Time_of_Day'].apply(lambda t: t.strftime('%H:%M'))

# Combine datasets and add Highway label
I5_average_volume_by_time["Highway"] = "I-5"
SR520_average_volume_by_time["Highway"] = "SR-520"

combined_hourly_data = pd.concat([I5_average_volume_by_time, SR520_average_volume_by_time], ignore_index=True)

# Plot chart
chart = alt.Chart(combined_hourly_data).mark_line().encode(
    x=alt.X('Time_of_Day_str:N', title='Time of Day', sort=combined_hourly_data['Time_of_Day_str'].tolist()),
    y=alt.Y('Average_Volume:Q', title='Average Volume'),
    color=alt.Color('Highway:N', title='Highway'),
    tooltip=['Highway', 'Time_of_Day_str:N', 'Average_Volume:Q']
).properties(
    title="Average 5-Minute Traffic Volume by Time of Day"
)

st.altair_chart(chart, use_container_width=True)

# I-5 daily
I5_daily_volume = rawI5.groupby(rawI5.index.date)['Volume'].sum()

I5_daily_volume = pd.DataFrame({'Volume': I5_daily_volume})

I5_daily_volume['Day_of_Week'] = pd.to_datetime(I5_daily_volume.index).day_name()

I5_day_of_week_average_volume = (
    I5_daily_volume.groupby('Day_of_Week')['Volume']
    .mean()
    .reindex(['Monday','Tuesday','Wednesday','Thursday','Friday'])
    .reset_index()
)

I5_day_of_week_average_volume.columns = ['Day_of_Week', 'Average_Volume']

# SR-520 daily
SR520_daily_volume = rawSR520.groupby(rawSR520.index.date)['Volume'].sum()

SR520_daily_volume = pd.DataFrame({'Volume': SR520_daily_volume})

SR520_daily_volume['Day_of_Week'] = pd.to_datetime(SR520_daily_volume.index).day_name()

SR520_day_of_week_average_volume = (
    SR520_daily_volume.groupby('Day_of_Week')['Volume']
    .mean()
    .reindex(['Monday','Tuesday','Wednesday','Thursday','Friday'])
    .reset_index()
)

SR520_day_of_week_average_volume.columns = ['Day_of_Week', 'Average_Volume']

# Daily Bar chart
I5_day_of_week_average_volume["Highway"] = "I-5"
SR520_day_of_week_average_volume["Highway"] = "SR-520"

combined_daily_data = pd.concat([
    I5_day_of_week_average_volume,
    SR520_day_of_week_average_volume
])

chart = alt.Chart(combined_daily_data).mark_bar().encode(
    x=alt.X(
        'Day_of_Week:N',
        sort=['Monday','Tuesday','Wednesday','Thursday','Friday'],
        title='Day of Week'
    ),
    y=alt.Y('Average_Volume:Q', title='Average Volume'),
    color=alt.Color('Highway:N', title='Highway'),
    xOffset='Highway:N',
    tooltip=['Highway', 'Day_of_Week', 'Average_Volume']
).properties(
    title="Average Daily Volume by Day of the Week: I5 vs SR520"
)

st.altair_chart(chart, use_container_width=True)

# -----------------------------------------------------------------------------------------
# --- Speed-volume relationship ---
st.header("Speed-Volume Relationship")

rawI5 = rawI5.reset_index()
rawSR520 = rawSR520.reset_index()

rawI5['DateTime'] = pd.to_datetime(rawI5['DateTime'])
rawSR520['DateTime'] = pd.to_datetime(rawSR520['DateTime'])

rawI5['Highway'] = 'I-5'
rawSR520['Highway'] = 'SR-520'

combined_point_data = pd.concat([rawI5, rawSR520])

# widget
selected_highway = st.radio("Select Highway", options=combined_point_data['Highway'].unique())    
filtered_data = combined_point_data[combined_point_data['Highway'] == selected_highway]

col1, col2 = st.columns(2)

# hexplot option
with col1:
    hexbin_chart = alt.Chart(filtered_data).mark_rect().encode(
        x=alt.X('Volume:Q', bin=alt.Bin(maxbins=50), title='Volume'),
        y=alt.Y('Speed:Q', bin=alt.Bin(maxbins=50), title='Speed (mph)'),
        color=alt.Color('count()', scale=alt.Scale(type='log'), title='Density'),
        tooltip=['count()']
    )
    st.altair_chart(hexbin_chart, use_container_width=True)

# scatter plot option
with col2:
    points = alt.Chart(combined_point_data.sample(frac=0.02, random_state=42)).mark_point(
        filled=True, size=20
    ).encode(
        x=alt.X('Volume:Q',title='Volume'),
        y=alt.Y('Speed:Q', title='Speed (mph)'),
        color='Highway:N'
    )
    st.altair_chart(points, use_container_width=True)

# -----------------------------------------------------------------------------------------
# --- Machine Learning Model ---

st.header("Modeling I-5")

# modeled data
def timelag(df1, df2, shiftn):
  df1[['SpeedLag1',
       'VolLag1',
       'VPLLag1',
       'OccLag1']] = df2[['Speed',
                          'Volume',
                          'Volume Per Lane',
                          'Occupancy']].shift(shiftn)
  df1 = df1.dropna().reset_index(drop=True)

  if 'Unnamed: 0' in df1.columns:
    df1 = df1.drop('Unnamed: 0', axis = 1)
  return df1

def dfSort(df):
  sorted = df.dropna().reset_index(drop=True)
  cutoff_train = sorted["DateTime"].quantile(0.70)
  cutoff_val= sorted["DateTime"].quantile(0.85)

  train = sorted[sorted["DateTime"] < cutoff_train]
  val   = sorted[(sorted["DateTime"] >= cutoff_train) & (sorted["DateTime"] < cutoff_val)]
  test  = sorted[sorted["DateTime"] >= cutoff_val]
  return sorted, train, val, test

def trainModel(traindat, valdat, testdat, model, pred_val):
  xval = traindat[['SpeedLag1', 'VolLag1', 'VPLLag1', 'OccLag1']]
  model.fit(xval, traindat[pred_val].to_frame())

  return model

clf = Lasso(alpha=0.1)

dfi5_test = timelag(rawI5.copy(), rawSR520.copy(), 3)
dfi5_test_sorted, trainm, valm, testm = dfSort(dfi5_test)
clf = trainModel(trainm, valm, testm, clf, 'Volume')
prediction = clf.predict(dfi5_test.copy()[['SpeedLag1', 'VolLag1', 'VPLLag1', 'OccLag1']])
pred_s = pd.Series(prediction)
dfpred = pd.DataFrame({'DateTime':dfi5_test['DateTime'], 'Volume': pred_s})
dfpred = dfpred.dropna().reset_index(drop=True)

# comparing data
dfpred = dfpred[dfpred["DateTime"].dt.dayofweek < 5]

rawI5_compare = rawI5.copy()
rawI5_compare = rawI5_compare.drop(columns=["Unnamed: 0", "LoopID", "Speed", "Volume Per Lane", "Occupancy", "Highway"])

dfpred["Data"] = "Measured I-5 Volume"
rawI5_compare["Data"] = "Modeled I-5 Volume"

model = pd.concat([dfpred, rawI5_compare], ignore_index=True)

model["DateTime"] = pd.to_datetime(model["DateTime"])
model["Week"] = model["DateTime"].dt.to_period("W").dt.start_time

model["Date"] = model["DateTime"].dt.date  # just the day
daily_volume = (
    model.groupby(["Date", "Data"])["Volume"]
    .sum()
    .reset_index()
)

chart = alt.Chart(daily_volume).mark_line().encode(
    x=alt.X('Date:T', title='Date'),
    y=alt.Y('Volume:Q', title='Daily Volume'),
    color=alt.Color('Data:N', title='Data Source'),
    tooltip=['Data', 'Date:T', 'Volume:Q']
).interactive()

st.altair_chart(chart, use_container_width=True)
