"""
MOmics-ML v11 — GUI inference helper.

Single-file inference helper for loading the locked pipeline and scoring a sample.
Drop into the Streamlit GUI codebase and import `score_sample`.

Usage:
    from MOmics_v11_inference import load_pipeline, score_sample

    pipe = load_pipeline("models/MOmics_v11_locked_pipeline.pkl")
    result = score_sample(pipe, rna_dict={"BSN": 23.4, "PCLO": 18.1, ...},
                                prot_dict={"CIT": -0.4, "PTPRT": -1.1, ...},
                                met_dict={"hypotaurine": 8.2, "creatinine": 5.1, ...})
    print(result)
    # {
    #   "P_GBM_raw": 0.971,
    #   "P_GBM_calibrated": 0.351,
    #   "binary_call": "GBM",
    #   "threshold_used": 0.964,
    #   "per_layer": {"rna": 0.97, "prot": 0.03, "met": None},
    # }
"""

from __future__ import annotations
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional


# ---- Public API ----

def load_pipeline(path: str | Path) -> dict:
    """Load the bundled v11 artifact. Returns a dict with all components."""
    return joblib.load(path)


def score_sample(
    pipe: dict,
    rna_dict: Optional[dict] = None,
    prot_dict: Optional[dict] = None,
    met_dict: Optional[dict] = None,
    use_calibrated: bool = False,
    rna_input_format: str = "raw_counts",
) -> dict:
    """
    Score a single sample and return raw + calibrated probabilities.

    Parameters
    ----------
    pipe : dict
        Loaded pipeline from load_pipeline(...).
    rna_dict, prot_dict, met_dict : dict or None
        {feature_name: raw_abundance} for each layer. Pass None if a layer is
        unavailable for this sample. RNA dict expects raw read counts (will be
        log1p-transformed internally); prot/met dicts expect already-log2-
        transformed abundances (matches CPTAC standard).
        Feature names must match the locked panel (see pipe["pruned_features"]).
    use_calibrated : bool
        Default False. If True, returns the binary call based on the calibrated
        probability rather than the raw probability. Recommended: keep False
        and use raw probability for the call (the Youden threshold was derived
        from raw scores). Display the calibrated probability for clinical
        interpretation only.

    Returns
    -------
    dict with:
        P_GBM_raw          : float, raw fusion probability (0-1)
        P_GBM_calibrated   : float, isotonically-calibrated probability (0-1)
        binary_call        : "GBM" or "non-GBM"
        threshold_used     : float, the Youden threshold from discovery
        per_layer          : dict of per-layer probabilities (None if missing)
        layers_available   : list of layers present in this sample
    """
    pruned = pipe["pruned_features"]
    z_params = pipe["zscore_params"]
    sub_models = pipe["sub_models"]
    fusion_model = pipe["fusion_model"]
    threshold = pipe["youden_threshold"]
    iso = pipe.get("isotonic_calibrator", None)

    layer_inputs = {"rna": rna_dict, "prot": prot_dict, "met": met_dict}
    per_layer_p = {}
    layers_present = []

    for layer in ["rna", "prot", "met"]:
        d = layer_inputs[layer]
        if d is None:
            per_layer_p[layer] = None
            continue
        # Build feature vector in locked panel order, missing -> NaN
        x = np.array([[d.get(f, np.nan) for f in pruned[layer]]], dtype=float)
        # RNA preprocessing: only apply log1p if input is raw counts.
        # Skip log transform if user provides already-log-normalized values
        # (e.g. LinkedOmicsKB RSEM log2 data, or any pre-normalized source).
        if layer == "rna" and rna_input_format == "raw_counts":
            with np.errstate(invalid="ignore"):
                x = np.where(np.isfinite(x) & (x >= 0), np.log1p(x), x)
        elif layer == "rna" and rna_input_format not in ("raw_counts", "log_normalized"):
            raise ValueError(f"rna_input_format must be 'raw_counts' or 'log_normalized', got {rna_input_format!r}")
        # Z-score using locked discovery-cohort parameters
        mu = np.array([z_params[layer]["mean"].get(f, 0.0) for f in pruned[layer]])
        sd = np.array([z_params[layer]["std"].get(f, 1.0) for f in pruned[layer]])
        sd = np.where(sd == 0, 1.0, sd)
        x = (x - mu) / sd
        # XGBoost handles NaN natively
        per_layer_p[layer] = float(sub_models[layer].predict_proba(x)[0, 1])
        layers_present.append(layer)

    if not layers_present:
        raise ValueError("Provide at least one of rna_dict, prot_dict, met_dict.")

    # Fusion: per-layer probabilities as input, missing layers -> NaN
    fusion_input = np.array([[per_layer_p[layer] if per_layer_p[layer] is not None else np.nan
                               for layer in ["rna", "prot", "met"]]], dtype=float)
    p_raw = float(fusion_model.predict_proba(fusion_input)[0, 1])

    # Calibrated probability (if calibrator present in bundle)
    p_cal = float(iso.predict([p_raw])[0]) if iso is not None else None

    # Binary call — use raw by default (threshold was set on raw scores)
    score_for_call = p_cal if (use_calibrated and p_cal is not None) else p_raw
    call = "GBM" if score_for_call >= threshold else "non-GBM"

    return {
        "P_GBM_raw": p_raw,
        "P_GBM_calibrated": p_cal,
        "binary_call": call,
        "threshold_used": float(threshold),
        "per_layer": per_layer_p,
        "layers_available": layers_present,
    }


# ---- Optional: introspection helpers for the GUI ----

def get_required_features(pipe: dict) -> dict[str, list[str]]:
    """Return the feature names the GUI should request from the user, per layer."""
    return {layer: list(pipe["pruned_features"][layer]) for layer in ["rna", "prot", "met"]}


def get_calibration_info(pipe: dict) -> dict:
    """Return calibration metadata for the GUI to display alongside calibrated probs."""
    return pipe.get("calibration_info", {"method": "none"})


# ---- Smoke test ----

if __name__ == "__main__":
    import sys
    pkl_path = sys.argv[1] if len(sys.argv) > 1 else "models/MOmics_v11_locked_pipeline.pkl"
    p = load_pipeline(pkl_path)

    print("=== MOmics-ML v11 pipeline ===")
    print(f"Version: {p.get('version')}")
    print(f"Threshold (Youden, raw scores): {p['youden_threshold']:.3f}")
    print()
    print("Required features per layer:")
    for layer, feats in get_required_features(p).items():
        print(f"  {layer:5s} ({len(feats)}): {feats}")
    print()
    print("Calibration:", get_calibration_info(p))
    print()
    print("Test 1: full multi-omic input (typical GBM-like z-scores around 0)")
    result = score_sample(p,
        rna_dict={"BSN": 100.0, "PCLO": 100.0, "PRKCE": 100.0,
                   "PTPRT": 100.0, "CIT": 100.0, "MAPT": 100.0},
        prot_dict={"PTPRT": 0.0, "CIT": 0.0, "PCLO": 0.0, "BSN": 0.0},
        met_dict={"hypotaurine": 0.0, "creatinine": 0.0, "citricacid": 0.0})
    for k, v in result.items():
        print(f"  {k}: {v}")
    print()
    print("Test 2: RNA-only input (typical use case when proteomics unavailable)")
    result = score_sample(p,
        rna_dict={"BSN": 100.0, "PCLO": 100.0, "PRKCE": 100.0,
                   "PTPRT": 100.0, "CIT": 100.0, "MAPT": 100.0})
    for k, v in result.items():
        print(f"  {k}: {v}")
