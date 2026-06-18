import pandas as pd
import mne
from mne.io.edf.edf import RawEDF

from src.eeg.config import DEFAULT_CHANNELS, EVENT_LABELS


class PreprocessingError(ValueError):
    pass


def normalise_channel_name(channel_name):
    return "".join(
        character for character in channel_name.upper() if character.isalnum()
    )


def resolve_channel_names(raw_channel_names, requested_channels=DEFAULT_CHANNELS):
    normalised_lookup = {
        normalise_channel_name(channel_name): channel_name
        for channel_name in raw_channel_names
    }
    resolved_channels = []
    missing_channels = []

    for requested_channel in requested_channels:
        requested_normalised = normalise_channel_name(requested_channel)
        exact_match = normalised_lookup.get(requested_normalised)
        if exact_match is not None:
            resolved_channels.append(exact_match)
            continue

        prefix_matches = [
            channel_name
            for channel_name in raw_channel_names
            if normalise_channel_name(channel_name).startswith(requested_normalised)
        ]
        if prefix_matches:
            resolved_channels.append(prefix_matches[0])
        else:
            missing_channels.append(requested_channel)

    if missing_channels:
        raise PreprocessingError(
            "Missing required EEG channels: " + ", ".join(missing_channels)
        )

    return resolved_channels


def prepare_motor_imagery_epochs(
    raw: RawEDF,
    selected_channels=DEFAULT_CHANNELS,
    l_freq=8.0,
    h_freq=30.0,
    notch_freq=60.0,
    tmin=0.5,
    tmax=3.5,
    resample_sfreq=None,
):
    resolved_channels = resolve_channel_names(raw.ch_names, selected_channels)
    working_raw = raw.copy().pick(resolved_channels).load_data()

    if notch_freq is not None:
        working_raw.notch_filter(notch_freq, verbose=False)

    working_raw.filter(l_freq=l_freq, h_freq=h_freq, verbose=False)

    if resample_sfreq is not None:
        working_raw.resample(resample_sfreq, verbose=False)

    events, event_id = mne.events_from_annotations(working_raw, verbose=False)
    selected_event_id = {
        event: event_id[event] for event in EVENT_LABELS if event in event_id  # type: ignore
    }

    if not selected_event_id:
        raise PreprocessingError(
            "No T1/T2 motor imagery annotations found in the uploaded EDF file."
        )

    epochs = mne.Epochs(
        working_raw,
        events,
        event_id=selected_event_id,
        tmin=tmin,
        tmax=tmax,
        baseline=None,
        preload=True,
        verbose=False,
    )

    if len(epochs) == 0:
        raise PreprocessingError("No usable T1/T2 epochs were created from this file.")

    metadata_rows = []
    event_code_to_name = {code: name for name, code in selected_event_id.items()}

    for epoch_index, event in enumerate(epochs.events):
        event_name = event_code_to_name[
            event[2]
        ]  # sample_index, previous_event_id, event_id
        onset_seconds = event[0] / epochs.info["sfreq"]
        metadata_rows.append(
            {
                "epoch_index": epoch_index,
                "onset_seconds": float(onset_seconds),
                "duration_seconds": float(tmax - tmin),
                "true_event": event_name,
                "true_label": EVENT_LABELS[event_name],
            }
        )

    labels = [row["true_event"] for row in metadata_rows]

    return epochs, labels, pd.DataFrame(metadata_rows)
