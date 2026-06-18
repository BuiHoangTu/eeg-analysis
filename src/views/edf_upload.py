import pandas as pd
import streamlit as st
import mne
import tempfile


def upload_edf_file():
    """
    Handle EDF file upload and display basic file information.
    
    Returns:
        raw object if file uploaded, None otherwise
    """
    st.subheader("Upload EDF File")

    uploaded_file = st.file_uploader("Upload PhysioNet EEG EDF file", type=["edf"])

    if uploaded_file:
        file_key = f"{uploaded_file.name}:{uploaded_file.size}"
        if st.session_state.get("uploaded_edf_key") != file_key:
            st.session_state["uploaded_edf_key"] = file_key
            st.session_state["show_visualisation"] = False

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

        if st.button("Submit", type="primary"):
            st.session_state["show_visualisation"] = True

        if st.session_state.get("show_visualisation", False):
            return raw

    return None
