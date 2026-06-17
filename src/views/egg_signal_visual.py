import pandas as pd
import streamlit as st
import numpy as np
import plotly.graph_objects as go


MOTOR_CHANNEL_DEFAULTS = ("C3", "Cz", "C4")
EVENT_MARKERS = {"T0", "T1", "T2"}


def normalise_channel_name(channel_name):
    return "".join(character for character in channel_name.upper() if character.isalnum())


def default_channel_index(channel_names):
    normalised_channels = [normalise_channel_name(channel) for channel in channel_names]

    for preferred_channel in MOTOR_CHANNEL_DEFAULTS:
        preferred_normalised = normalise_channel_name(preferred_channel)
        for index, channel in enumerate(normalised_channels):
            if channel == preferred_normalised:
                return index
        for index, channel in enumerate(normalised_channels):
            if channel.startswith(preferred_normalised):
                return index

    return 0


def visible_annotations(raw, start_time, end_time):
    annotations = raw.annotations
    rows = []

    for onset, duration, description in zip(
        annotations.onset, annotations.duration, annotations.description
    ):
        if start_time <= onset <= end_time:
            rows.append(
                {
                    "Onset (s)": float(onset),
                    "Duration (s)": float(duration),
                    "Description": description,
                }
            )

    return pd.DataFrame(rows)


def render_eeg_visualisation(raw):
    st.subheader("EEG Signal Visualisation")

    if not raw.ch_names:
        st.warning("No EEG channels found in this file.")
        return

    sfreq = raw.info["sfreq"]
    duration = float(raw.duration)
    default_end_time = min(10.0, duration)

    selected_channel = st.selectbox(
        "Channel",
        raw.ch_names,
        index=default_channel_index(raw.ch_names),
    )

    start_col, end_col = st.columns(2)
    with start_col:
        start_time = st.number_input(
            "Start time (seconds)",
            min_value=0.0,
            max_value=duration,
            value=0.0,
            step=1.0,
        )
    with end_col:
        end_time = st.number_input(
            "End time (seconds)",
            min_value=0.0,
            max_value=duration,
            value=default_end_time,
            step=1.0,
        )

    if end_time <= start_time:
        st.error("End time must be greater than start time.")
        return

    start_sample = int(start_time * sfreq)
    stop_sample = min(int(end_time * sfreq), raw.n_times)

    if stop_sample <= start_sample:
        st.error("Selected time range does not contain any EEG samples.")
        return

    signal = raw.get_data(
        picks=[selected_channel], start=start_sample, stop=stop_sample
    )[0]
    signal_microvolts = signal * 1_000_000
    times = start_sample / sfreq + np.arange(signal_microvolts.size) / sfreq

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=times,
            y=signal_microvolts,
            mode="lines",
            name=selected_channel,
        )
    )

    events_in_window = visible_annotations(raw, start_time, end_time)
    for _, event in events_in_window.iterrows():
        description = str(event["Description"])
        if description in EVENT_MARKERS:
            fig.add_vline(
                x=event["Onset (s)"],
                line_dash="dash",
                line_color="red",
                annotation_text=description,
                annotation_position="top",
            )

    fig.update_layout(
        xaxis_title="Time (s)",
        yaxis_title=f"{selected_channel} amplitude (µV)",
        margin={"l": 20, "r": 20, "t": 30, "b": 20},
        height=420,
    )

    st.plotly_chart(fig, use_container_width=True)

    st.write("Visible annotations")
    if events_in_window.empty:
        st.info("No annotations found in the selected time range.")
    else:
        st.dataframe(events_in_window, hide_index=True)
