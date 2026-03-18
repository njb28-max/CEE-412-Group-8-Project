import numpy as np
import pandas as pd
import altair as alt
import streamlit as st

st.write('CEE 412 Group Project')

# --- Load data ---
rawI5 = pd.read_csv("CEE412Project/005es16732_loop_cloutput.csv")
rawSR520 = pd.read_csv("CEE412Project/520es00972_loop_cloutput.csv")

# --- Unified processing function ---
def process_sensor(df, sensor_name, freq="D", add_month_day=False):
    df["DateTime"] = pd.to_datetime(df["DateTime"])
    df = df.set_index("DateTime")
    
    # Resample volume (daily or weekly)
    resampled = df.resample(freq)["Volume"].sum().reset_index()
    
    # --- Remove weekends ---
    if freq == "D":
        resampled = resampled[resampled["DateTime"].dt.dayofweek < 5]  # 0=Mon, 4=Fri
    
    resampled["LoopID"] = sensor_name
    
    # Add Month/Day for daily charts
    if add_month_day:
        resampled["Month"] = resampled["DateTime"].dt.month_name()
        resampled["Day"] = resampled["DateTime"].dt.day
        month_order = ["January","February","March","April","May","June",
                       "July","August","September","October","November","December"]
        resampled["Month"] = pd.Categorical(resampled["Month"], categories=month_order, ordered=True)
    
    return resampled

# --- Process both sensors ---
daily_I5 = process_sensor(rawI5, "I5", freq="D", add_month_day=True)
daily_SR520 = process_sensor(rawSR520, "SR520", freq="D", add_month_day=True)
combined_daily = pd.concat([daily_I5, daily_SR520], ignore_index=True)

weekly_I5 = process_sensor(rawI5, "I5", freq="W-MON")
weekly_SR520 = process_sensor(rawSR520, "SR520", freq="W-MON")
combined_weekly = pd.concat([weekly_I5, weekly_SR520], ignore_index=True)

# --- Streamlit widget for daily chart ---
selected_month = st.selectbox("Select a month to view:", combined_daily["Month"].cat.categories)
filtered_daily = combined_daily[combined_daily["Month"] == selected_month]

# --- Daily chart ---
daily_chart = alt.Chart(filtered_daily).mark_line(interpolate='basis').encode(
    x=alt.X("Day:Q", scale=alt.Scale(domain=[1, 31])),
    y=alt.Y("Volume:Q", scale=alt.Scale(domain=[0, 20000])),
    color="LoopID:N",
    tooltip=["LoopID","Day","Volume"]
).properties(
    title=f"Daily Volume for {selected_month}"
)

# --- Weekly chart ---
weekly_chart = alt.Chart(combined_weekly).mark_line(interpolate='basis').encode(
    x="DateTime:T",
    y="Volume:Q",
    color="LoopID:N",
    tooltip=["LoopID","DateTime","Volume"]
).properties(
    title="Weekly Volume Comparison: I5 vs SR520"
)

# --- Display charts ---
st.altair_chart(daily_chart, use_container_width=True)
st.altair_chart(weekly_chart, use_container_width=True)