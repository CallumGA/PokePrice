#!/usr/bin/env python3
# explain_model.py

import os
import json
import numpy as np
import pandas as pd
import torch
import joblib
import shap
import matplotlib.pyplot as plt
from safetensors.torch import load_file
from network import PricePredictor

# --- 0. Config ---
MODEL_DIR = "model"
DATA_DIR = "data"
SCALER_PATH = os.path.join(DATA_DIR, "scaler.pkl")
DATA_PATH = os.path.join(DATA_DIR, "pokemon_final_with_labels.csv")
CONFIG_PATH = os.path.join(MODEL_DIR, "config.json")
TARGET_COLUMN = "price_will_rise_30_in_6m"

# --- 1. Load model & assets ---
with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

feature_columns = config["feature_columns"]
input_size = config["input_size"]

model = PricePredictor(input_size=input_size)
model.load_state_dict(load_file(os.path.join(MODEL_DIR, "model.safetensors")))
model.eval()

scaler = joblib.load(SCALER_PATH)
full_data = pd.read_csv(DATA_PATH)

# Sanity checks
missing_cols = [c for c in feature_columns if c not in full_data.columns]
if missing_cols:
    raise ValueError(f"Missing required feature columns in CSV: {missing_cols}")

features_df = full_data[feature_columns]
if features_df.shape[1] != input_size:
    raise ValueError(
        f"Config input_size={input_size}, but CSV provides {features_df.shape[1]} features. "
        f"Ensure config['feature_columns'] matches the trained model."
    )

# --- 2. Prepare Data for SHAP ---
bg_n = min(100, len(features_df))
explain_n = min(10, len(features_df))

background_idx = features_df.sample(n=bg_n, random_state=42).index
explain_idx = features_df.sample(n=explain_n, random_state=1).index

background_data = features_df.loc[background_idx]
explain_instances = features_df.loc[explain_idx]

# Use arrays for scaler to avoid feature-name warnings
background_data_scaled = scaler.transform(background_data.values)
explain_instances_scaled = scaler.transform(explain_instances.values)

background_tensor = torch.tensor(background_data_scaled, dtype=torch.float32)  # no grad
explain_tensor = torch.tensor(explain_instances_scaled, dtype=torch.float32, requires_grad=True)

# --- Helpers ---
def get_shap_explanations(model, background_tensor, explain_tensor):
    """Try DeepExplainer then fall back to GradientExplainer. Return (explanation, explainer_used_name)."""
    try:
        print("Initializing SHAP DeepExplainer...")
        explainer = shap.DeepExplainer(model, background_tensor)
        print("Calculating SHAP values for the sample...")
        exp = explainer(explain_tensor)
        setattr(exp, "_expected_value_hint", getattr(explainer, "expected_value", None))
        return exp, "deep"
    except Exception as e:
        print(f"[DeepExplainer failed: {e}] Falling back to GradientExplainer...")
        explain_tensor.requires_grad_(True)
        grad_explainer = shap.GradientExplainer(model, background_tensor)
        exp = grad_explainer(explain_tensor)
        setattr(exp, "_expected_value_hint", getattr(grad_explainer, "expected_value", None))
        return exp, "grad"

def compute_base_value_safe(shap_explanation, instance_idx, model, background_tensor):
    """Return scalar base value robustly across SHAP versions."""
    bv = getattr(shap_explanation, "base_values", None)
    if bv is not None:
        try:
            return float(np.squeeze(bv[instance_idx]))
        except Exception:
            try:
                return float(np.squeeze(bv))
            except Exception:
                pass
    ev = getattr(shap_explanation, "_expected_value_hint", None)
    if ev is not None:
        try:
            return float(np.squeeze(ev))
        except Exception:
            try:
                return float(np.mean(ev))
            except Exception:
                pass
    with torch.no_grad():
        mu = background_tensor.mean(dim=0, keepdim=True)
        out = model(mu).detach().cpu().squeeze()
        return float(out.mean().item()) if out.numel() > 1 else float(out.item())

def stack_sample_shap_values(exp, n_features_expected):
    """
    Some SHAP versions return exp.values with shape (n_samples, 1) or other oddities.
    However, exp[i].values is typically the correct 1D (n_features,) vector.
    We rebuild a full matrix by stacking per-sample slices.
    """
    rows = []
    n_samples = len(exp.values) if hasattr(exp.values, "__len__") else len(exp)
    # Safer: iterate using the __getitem__ API
    for i in range(n_samples):
        v = np.asarray(exp[i].values).reshape(-1,)
        rows.append(v)
    M = np.vstack(rows)  # (n_samples, n_features)
    if M.shape[1] != n_features_expected:
        raise RuntimeError(
            f"Rebuilt SHAP matrix has shape {M.shape}; expected n_features={n_features_expected}."
        )
    return M

# --- 3. Compute SHAP explanations ---
shap_explanation, _ = get_shap_explanations(model, background_tensor, explain_tensor)
print("Calculation complete.")

# Attach unscaled display data for pretty plotting
shap_explanation.display_data = explain_instances.values
shap_explanation.feature_names = feature_columns

# --- 4a. Global Feature Importance (Bar / Summary) ---
print("\nGenerating global feature importance plot (summary_plot.png)...")

# Robustly build a (n_samples, n_features) matrix by stacking per-sample vectors
shap_vals_matrix = stack_sample_shap_values(shap_explanation, n_features_expected=len(feature_columns))

mean_abs_shap = np.abs(shap_vals_matrix).mean(axis=0)  # (n_features,)

# Build a fresh Explanation with values aligned to feature_names
plot_explanation = shap.Explanation(values=mean_abs_shap, feature_names=feature_columns)

plt.figure()
shap.plots.bar(plot_explanation, show=False)
plt.xlabel("mean(|SHAP value|) (average impact on model output magnitude)")
plt.savefig("summary_plot.png", bbox_inches="tight")
plt.close()
print("Saved: summary_plot.png")

# --- 4b. Local Explanation (Force Plot) ---
print("\nGenerating local explanation for one card (force_plot.html)...")
instance_to_explain_index = 0
single_explanation = shap_explanation[instance_to_explain_index]

# Some SHAP versions drop display_data on slicing; pull directly if needed
if getattr(single_explanation, "display_data", None) is None:
    row_unscaled = explain_instances.values[instance_to_explain_index]
else:
    row_unscaled = single_explanation.display_data
features_row = np.atleast_2d(np.asarray(row_unscaled, dtype=float))

base_val = compute_base_value_safe(shap_explanation, instance_to_explain_index, model, background_tensor)
phi = np.asarray(single_explanation.values).reshape(-1,)  # (n_features,)

force_plot = shap.force_plot(
    base_val,
    phi,
    features=features_row,
    feature_names=feature_columns
)
shap.save_html("force_plot.html", force_plot)
print("Saved: force_plot.html (open in a browser)")

# --- 4c. Optional: local waterfall PNG (often clearer) ---
try:
    print("Generating local waterfall plot (waterfall_single.png)...")
    plt.figure()
    shap.plots.waterfall(single_explanation, show=False, max_display=20)
    plt.savefig("waterfall_single.png", bbox_inches="tight")
    plt.close()
    print("Saved: waterfall_single.png")
except Exception as e:
    print(f"Waterfall plot skipped (reason: {e})")

# --- 5. Print metadata for the explained card ---
original_card_data = full_data.loc[explain_idx[instance_to_explain_index]]
name_val = original_card_data.get("name", "N/A")
tcgp_val = original_card_data.get("tcgplayer_id", "N/A")
label_val = original_card_data.get(TARGET_COLUMN, None)
label_str = "RISE" if bool(label_val) else "NOT RISE" if label_val is not None else "N/A"

print("\n--- Card Explained in force_plot.html / waterfall_single.png ---")
print(f"Name: {name_val}")
print(f"TCGPlayer ID: {tcgp_val}")
print(f"Actual Outcome in Dataset: {label_str}")

# TODO: convert the model into a format where i can share on hugging face as a model that can be pulled down and used
# TODO: include the SHAP charts force_plot.html and summary_plot.png explaining the model, as well as compute some other evaluation metrics for explanation in the card