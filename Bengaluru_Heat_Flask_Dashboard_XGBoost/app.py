"""Bengaluru Heat Intelligence — local Flask read-only dashboard.

Extract this folder inside the user's existing Bengaluru_HeatRisk_XGBoost project.
Run: python app.py   (does not train/recompute research results)
"""
from __future__ import annotations

import csv
import os
from collections import Counter
from pathlib import Path

from flask import Flask, abort, jsonify, render_template, send_file

HERE = Path(__file__).resolve().parent
PROJECT = PROJECT = HERE
app = Flask(__name__)
app.json.sort_keys = False

# No arbitrary user-supplied file paths are ever served.
FILES = {
    "satellite_rgb": "IPCV_Experiment/01_original_rgb.png",
    "median": "IPCV_Experiment/02_median_filtered.png",
    "clahe": "IPCV_Experiment/03_clahe_enhanced.png",
    "sobel": "IPCV_Experiment/04_sobel_edges.png",
    "texture": "IPCV_Experiment/05_local_texture.png",
    "preprocessing": "IPCV_Experiment/07_preprocessing_comparison.png",
    "metrics_poster": "IPCV_Experiment/All_ML_DL_Metrics_Poster.png",
    "comparison": "IPCV_Experiment/ml_vs_dl_accuracy_comparison.png",
    "baseline_confusion": "IPCV_Experiment/mlp_baseline_confusion_matrix.png",
    "texture_confusion": "IPCV_Experiment/mlp_texture_confusion_matrix.png",
    "baseline_loss": "IPCV_Experiment/mlp_baseline_loss.png",
    "texture_loss": "IPCV_Experiment/mlp_texture_loss.png",
    "xgb_confusion": "xgboost_outputs/01_confusion_matrix.png",
    "xgb_importance": "xgboost_outputs/02_feature_importance.png",
    "xgb_heatmap": "xgboost_outputs/04_bengaluru_xgboost_heatmap.png",
    "xgb_confidence": "xgboost_outputs/05_xgboost_confidence_map.png",
    "rf_importance": "urban_heat_project_outputs/feature_importance.png",
}
MAPS = {
    "satellite": "urban_heat_project_outputs/01_satellite.html",
    "indicators": "urban_heat_project_outputs/02_indicators.html",
    "heat_risk": "urban_heat_project_outputs/03_heat_risk.html",
    "mitigation": "urban_heat_project_outputs/04_mitigation_priority.html",
}


def checked_file(relative: str) -> Path | None:
    p = (PROJECT / relative).resolve()
    if not p.is_relative_to(PROJECT):
        return None
    return p if p.is_file() else None


def parse_metrics():
    for name in ("ml_vs_dl_comparison.csv", "model_comparison.csv", "mlp_dl_results.csv"):
        path = checked_file("IPCV_Experiment/" + name)
        if not path:
            continue
        with path.open(newline="", encoding="utf-8-sig") as file:
            source = list(csv.DictReader(file))
        rows = []
        for row in source:
            try:
                if not all(key in row for key in ("Features", "Model", "Accuracy", "Precision", "Recall", "F1 Score")):
                    continue
                nums = {k: float(row[k]) for k in ("Accuracy", "Precision", "Recall", "F1 Score")}
                # Original experiment scripts save sklearn 0-1 fractions.
                rows.append({"features": row["Features"], "model": row["Model"], **{
                    "accuracy": round(nums["Accuracy"] * 100, 2),
                    "precision": round(nums["Precision"] * 100, 2),
                    "recall": round(nums["Recall"] * 100, 2),
                    "f1": round(nums["F1 Score"] * 100, 2),
                }})
            except (ValueError, TypeError):
                continue
        if rows:
            return rows, name
    return [], None


def xgboost_visuals():
    """Enumerate only actual XGBoost image exports, never invent outputs."""
    folder = (PROJECT / "xgboost_outputs").resolve()
    if not folder.is_dir() or not folder.is_relative_to(PROJECT):
        return []
    visuals = []
    for path in sorted(folder.iterdir()):
        if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}:
            visuals.append({"id": path.name, "title": path.stem.replace("_", " ").replace("-", " ").title(),
                            "url": "/xgboost-visuals/" + path.name})
    return visuals


def sample_stats():
    candidates = ("IPCV_Experiment/training_samples_with_texture.csv", "urban_heat_project_outputs/training_samples.csv")
    for relative in candidates:
        path = checked_file(relative)
        if not path:
            continue
        counts = Counter()
        n = 0
        with path.open(newline="", encoding="utf-8-sig") as file:
            for row in csv.DictReader(file):
                n += 1
                risk = (row.get("Risk_Class") or "").strip()
                if risk:
                    counts[risk] += 1
        if n:
            return n, counts, relative
    return None, {}, None


@app.get("/")
def home():
    return render_template("index.html")


@app.get("/api/dashboard")
def dashboard():
    rows, source = parse_metrics()
    n, counts, samples_source = sample_stats()
    best = max(rows, key=lambda x: x["accuracy"]) if rows else None
    return jsonify({
        "status": "ok",
        "project_found": PROJECT.is_dir(),
        "samples": n,
        "samples_source": samples_source,
        "models": len({r["model"] for r in rows}),
        "configurations": len(rows),
        "best": best,
        "metrics": rows,
        "metrics_source": source,
        "risk_distribution": [{"name": key, "count": count} for key, count in counts.items()],
        "maps": {k: {"available": checked_file(v) is not None, "url": "/maps/" + k} for k, v in MAPS.items()},
        "images": {k: {"available": checked_file(v) is not None, "url": "/images/" + k} for k, v in FILES.items()},
        "xgboost_visuals": xgboost_visuals(),
        "satellite_indices_map": {"available": checked_file(MAPS["indicators"]) is not None,
                                 "url": "/maps/indicators",
                                 "layers": ["LST", "NDVI", "NDBI"]},
        "scientific_note": "Risk labels were derived from the project's heat-risk formula and are not independently measured ground truth.",
    })


@app.get("/images/<key>")
def images(key):
    if key not in FILES:
        abort(404)
    path = checked_file(FILES[key])
    if not path:
        abort(404)
    return send_file(path, mimetype="image/png", max_age=0)


@app.get("/xgboost-visuals/<filename>")
def xgboost_visual(filename):
    # Compare the name against the dynamically constructed allowlist, rather than serving arbitrary paths.
    allowed = {item["id"] for item in xgboost_visuals()}
    if filename not in allowed:
        abort(404)
    path = (PROJECT / "xgboost_outputs" / filename).resolve()
    if not path.is_file() or not path.is_relative_to((PROJECT / "xgboost_outputs").resolve()):
        abort(404)
    return send_file(path, max_age=0)


@app.get("/maps/<key>")
def maps(key):
    if key not in MAPS:
        abort(404)
    path = checked_file(MAPS[key])
    if not path:
        abort(404)
    # Only trusted, locally generated Folium HTML is served. Not user uploads.
    response = send_file(path, mimetype="text/html", max_age=0)
    response.headers["Cache-Control"] = "no-store"
    return response


if __name__ == "__main__":
    print(f"\nBengaluru Heat Intelligence — Project outputs: {PROJECT}")
    print("Open http://127.0.0.1:5000 in your browser (Ctrl+C to stop).\n")
    app.run(host="127.0.0.1", port=5000, debug=False)
