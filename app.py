# app.py
import pandas as pd
import streamlit as st
import mne
import tempfile


def main():
    st.title("EDF EEG Viewer")

    uploaded_file = st.file_uploader("Upload PhysioNet EEG EDF file", type=["edf"])

    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".edf") as tmp:
            tmp.write(uploaded_file.read())
            edf_path = tmp.name

        raw = mne.io.read_raw_edf(edf_path, preload=False, verbose=False)

        st.subheader("Basic File Information")
        st.write("Sampling frequency:", raw.info["sfreq"], "Hz")
        n_channels = len(raw.ch_names)
        st.write("Number of channels:", n_channels)
        st.write("Recording duration:", raw.duration, "seconds")

        channel_df = pd.DataFrame(
            {
                "No.": range(1, n_channels + 1),
                "Channel Name": raw.ch_names,
            }
        )
        st.subheader("Channels")
        st.dataframe(channel_df, hide_index=True)

        st.subheader("Annotations")
        if raw.annotations:
            st.write(raw.annotations.to_data_frame())
        else:
            st.write("No annotations found.")


if __name__ == "__main__":
    main()
