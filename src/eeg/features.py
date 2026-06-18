from mne import Epochs
import numpy as np
import pandas as pd


DEFAULT_BANDS = {
    "mu": (8.0, 13.0),
    "beta": (13.0, 30.0),
}


def extract_log_bandpower_features(epochs: Epochs, bands=DEFAULT_BANDS, epsilon=1e-12):

    psd = epochs.compute_psd(
        method="welch",
        fmin=min(low for low, _ in bands.values()),
        fmax=max(high for _, high in bands.values()),
        verbose=False,
    )
    powers = psd.get_data()
    freqs = psd.freqs

    feature_columns = []
    feature_names = []

    for channel_index, channel_name in enumerate(epochs.ch_names):
        for band_name, (low_freq, high_freq) in bands.items():
            band_mask = (freqs >= low_freq) & (freqs < high_freq)
            if not np.any(band_mask):
                band_power = np.full(len(epochs), epsilon)
            else:
                band_power = powers[:, channel_index, :][:, band_mask].mean(axis=1)
            feature_columns.append(np.log(band_power + epsilon))
            feature_names.append(f"{channel_name}_{band_name}")

    X = np.column_stack(feature_columns)
    features_df = pd.DataFrame(X, columns=feature_names)

    return features_df
