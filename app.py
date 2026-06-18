import streamlit as st

from src.views.edf_upload import upload_edf_file
from src.views.egg_signal_visual import render_eeg_visualisation
from src.views.motor_imagery_inference import render_motor_imagery_inference


def main():
    st.title("EDF EEG Viewer")

    raw = upload_edf_file()

    if raw is not None:
        render_eeg_visualisation(raw)
        render_motor_imagery_inference(raw)


if __name__ == "__main__":
    main()
