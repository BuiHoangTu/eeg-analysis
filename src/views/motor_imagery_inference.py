import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from src.eeg.features import extract_log_bandpower_features
from src.eeg.inference import (
    InferenceError,
    list_available_models,
    load_model_artifact,
    predict_epochs,
    validate_feature_names,
)
from src.eeg.epoch_processing import EVENT_LABELS, PreprocessingError, prepare_motor_imagery_epochs


DEFAULT_PREPROCESSING = {
    "channels": ["C3", "Cz", "C4"],
    "l_freq": 8.0,
    "h_freq": 30.0,
    "notch_freq": 60.0,
    "tmin": 0.5,
    "tmax": 3.5,
    "resample_sfreq": None,
}


def artifact_name(model_path):
    return model_path.stem.replace("_", " ").title()


def artifact_label(model_path, artifact):
    return artifact.get("model_name", artifact_name(model_path))


def render_model_summary(artifact):
    preprocessing = {**DEFAULT_PREPROCESSING, **artifact.get("preprocessing", {})}

    st.write("**Preprocessing settings**")
    st.write("Channels:", ", ".join(artifact.get("channels", preprocessing["channels"])))
    st.write("Bandpass:", f"{preprocessing['l_freq']}–{preprocessing['h_freq']} Hz")
    st.write("Notch:", "None" if preprocessing["notch_freq"] is None else f"{preprocessing['notch_freq']} Hz")
    st.write("Epoch window:", f"{preprocessing['tmin']}–{preprocessing['tmax']} seconds after cue onset")
    st.write("Feature method:", artifact.get("feature_method", "log_bandpower"))

    metrics = artifact.get("metrics")
    if metrics:
        st.write("**Model evaluation summary**")
        metrics_display = dict(metrics)

        cm = metrics_display.pop("confusion_matrix", None)

        st.table(
            pd.DataFrame(
                {
                    "Metric": metrics_display.keys(),
                    "Value": [str(v) for v in metrics_display.values()],
                }
            )
        )
        if cm is not None:
            labels = ["T1", "T2"]

            cm_df = pd.DataFrame(
                cm,
                index=[f"Actual {label}" for label in labels],
                columns=[f"Predicted {label}" for label in labels],
            )

            st.subheader("Confusion Matrix")
            st.dataframe(cm_df, use_container_width=True)


def build_results_table(event_metadata, predictions, confidence):
    results = event_metadata.copy()
    results["predicted_event"] = predictions
    results["predicted_label"] = results["predicted_event"].map(EVENT_LABELS).fillna(
        results["predicted_event"]
    )
    if confidence is not None:
        results["confidence"] = confidence
    results["correct"] = results["true_event"] == results["predicted_event"]
    return results


def render_metrics(results):
    st.write("**Uploaded-file metrics**")
    accuracy = accuracy_score(results["true_event"], results["predicted_event"])
    f1 = f1_score(results["true_event"], results["predicted_event"], labels=["T1", "T2"], average="macro")
    st.write("Accuracy:", round(float(accuracy), 3))
    st.write("Macro F1:", round(float(f1), 3))

    matrix = confusion_matrix(results["true_event"], results["predicted_event"], labels=["T1", "T2"])
    matrix_df = pd.DataFrame(matrix, index=["True T1", "True T2"], columns=["Pred T1", "Pred T2"])
    st.dataframe(matrix_df)


def render_prediction_timeline(results):
    fig = go.Figure()
    colours = {"T1": "#1f77b4", "T2": "#ff7f0e"}

    for row_name, column_name in (("Ground truth", "true_event"), ("Prediction", "predicted_event")):
        fig.add_trace(
            go.Scatter(
                x=results["onset_seconds"],
                y=[row_name] * len(results),
                mode="markers",
                marker={
                    "size": 14,
                    "color": [colours.get(value, "#777777") for value in results[column_name]],
                },
                text=results[column_name],
                name=row_name,
            )
        )

    fig.update_layout(
        xaxis_title="Time (s)",
        yaxis_title="",
        height=260,
        margin={"l": 20, "r": 20, "t": 30, "b": 20},
    )
    st.plotly_chart(fig, use_container_width=True)


def run_inference(raw, artifact):
    preprocessing = {**DEFAULT_PREPROCESSING, **artifact.get("preprocessing", {})}
    channels = artifact.get("channels", preprocessing["channels"])

    epochs, _, event_metadata = prepare_motor_imagery_epochs(
        raw,
        selected_channels=channels,
        l_freq=preprocessing["l_freq"],
        h_freq=preprocessing["h_freq"],
        notch_freq=preprocessing["notch_freq"],
        tmin=preprocessing["tmin"],
        tmax=preprocessing["tmax"],
        resample_sfreq=preprocessing["resample_sfreq"],
    )
    features = extract_log_bandpower_features(epochs)
    feature_names = list(features.columns)
    validate_feature_names(artifact, feature_names)
    predictions, confidence = predict_epochs(artifact, features.to_numpy())

    return build_results_table(event_metadata, predictions, confidence)


def render_motor_imagery_inference(raw):
    st.subheader("Motor Imagery Inference")
    st.write(
        "Predict imagined left vs right fist epochs. T1 is imagined left fist, "
        "T2 is imagined right fist, and T0 rest annotations are ignored. This view "
        "is intended for PhysioNet runs R04, R08, and R12."
    )

    model_paths = list_available_models()
    if not model_paths:
        st.warning("No pre-trained model artifacts found in models/*.joblib. The app does not train models online.")
        return

    loaded_artifacts = {}
    model_labels = []
    for model_path in model_paths:
        try:
            artifact = load_model_artifact(model_path)
        except (InferenceError, OSError, ValueError) as error:
            st.warning(f"Skipping {model_path.name}: {error}")
            continue
        label = artifact_label(model_path, artifact)
        loaded_artifacts[label] = artifact
        model_labels.append(label)

    if not model_labels:
        st.warning("No valid pre-trained model artifacts could be loaded.")
        return

    selected_model = st.selectbox("Select pre-trained model", model_labels)
    artifact = loaded_artifacts[selected_model]
    render_model_summary(artifact)

    inference_context = {
        "uploaded_edf_key": st.session_state.get("uploaded_edf_key"),
        "selected_model": selected_model,
    }

    if st.button("Run inference"):
        try:
            st.session_state["motor_imagery_results"] = run_inference(raw, artifact)
            st.session_state["motor_imagery_results_context"] = inference_context
        except (PreprocessingError, InferenceError, RuntimeError, ValueError) as error:
            st.error(str(error))
            return

    results = st.session_state.get("motor_imagery_results")
    results_context = st.session_state.get("motor_imagery_results_context")
    if results is not None and results_context == inference_context:
        st.write("**Epoch-level predictions**")
        st.dataframe(results, hide_index=True)
        render_metrics(results)
        st.write("**Ground truth vs prediction timeline**")
        render_prediction_timeline(results)
