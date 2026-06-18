## Download the data:

```bash 
cd data
wget -r -N -c -np https://physionet.org/files/eegmmidb/1.0.0/
```

## Log Band-Power Feature Extraction

EEG features were extracted using a frequency-domain band-power approach. 
First, each epoch was transformed into its power spectral density (PSD) representation using Welch's method. 
This method estimates the PSD by dividing the signal into overlapping segments, computing the Fourier transform of each segment, and averaging the resulting spectra, producing a stable estimate of spectral power.

The PSD was computed only over the frequency range required by the selected EEG bands. 
For each channel and frequency band, the average spectral power within the corresponding frequency interval was calculated. 
Specifically, frequency bins whose frequencies fell within the band limits were identified, and the mean PSD value across those bins was computed for each epoch.

The frequency bands used were:


* mu: 8–13 Hz
* bete: 13–30 Hz

For each channel–band combination, the resulting band-power value was transformed using the natural logarithm. 
The logarithmic transformation reduces the skewness of the power distribution, compresses large values, and produces features that are generally more suitable for statistical analysis and machine-learning classifiers.

The final feature vector for each epoch consisted of the log-transformed band-power values from all channel–band combinations. 
Feature names were constructed by concatenating the channel name and frequency band (e.g., `C3_mu`, `C4_beta`).
