"""Bengaluru urban heat risk: VS Code version of the Colab analysis.

Run this FIRST. Requires authenticated Google Earth Engine and internet.
This is a clean reproducible pipeline, not an exact cell-for-cell notebook conversion.
"""
from pathlib import Path
import json
import shutil
import numpy as np
import pandas as pd
import ee
import geemap
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import joblib

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "urban_heat_project_outputs"
OUT.mkdir(exist_ok=True)
PROJECT_ID = "fair-veld-492517-s0"  # Original Google Cloud project; account must have access
ROI = None  # Created after Earth Engine authentication in main()
def connect_earth_engine():
    try:
        ee.Initialize(project=PROJECT_ID)
        print("Earth Engine connected.")
    except Exception:
        print("Earth Engine authentication required.")
        ee.Authenticate()
        ee.Initialize(project=PROJECT_ID)
        print("Earth Engine connected.")

def save_map(filename, layers):
    """Save interactive Earth Engine layers as HTML using Folium directly.

    Avoids geemap.foliumap, whose basemap initialization may fail for some
    geemap/python-box package combinations.
    """
    import folium

    m = folium.Map(location=[13.0, 77.6], zoom_start=10,
                   tiles="OpenStreetMap", control_scale=True)
    for raster, vis, label in layers:
        map_info = ee.Image(raster).getMapId(vis)
        folium.raster_layers.TileLayer(
            tiles=map_info["tile_fetcher"].url_format,
            attr="Google Earth Engine",
            name=label,
            overlay=True,
            control=True,
            show=True,
        ).add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    dest = OUT / filename
    m.save(str(dest))
    print("Map saved:", dest)


def export_geotiff(image, filename, scale):
    dest = OUT / filename
    if dest.exists():
        print("Using existing export:", dest)
        return dest
    print("Exporting", filename, "from Earth Engine...")
    geemap.ee_export_image(image, filename=str(dest), scale=scale, region=ROI, file_per_band=False)
    if not dest.exists() or dest.stat().st_size == 0:
        raise RuntimeError(f"Export failed: {dest}. Check EE permissions, network and export size.")
    return dest


def risk_class(score):
    return "Low" if score < 0.40 else ("Medium" if score < 0.50 else "High")


def main():
    global ROI
    connect_earth_engine()
    ROI = ee.Geometry.Rectangle([77.45, 12.85, 77.75, 13.15])
    images = (ee.ImageCollection("LANDSAT/LC09/C02/T1_L2")
              .filterBounds(ROI).filterDate("2024-01-01", "2026-01-01")
              .filter(ee.Filter.lt("CLOUD_COVER", 30))
              .filter(ee.Filter.eq("PROCESSING_LEVEL", "L2SP"))
              .sort("CLOUD_COVER"))
    count = images.size().getInfo()
    print("Suitable Landsat scenes:", count)
    if not count:
        raise RuntimeError("No Landsat images found for the original time period and filters.")
    image = ee.Image(images.first()).clip(ROI)
    date = ee.Date(image.get("system:time_start")).format("YYYY-MM-dd").getInfo()
    print("Selected Landsat scene:", date)
    qa = image.select("QA_PIXEL")
    clear = (qa.bitwiseAnd(1 << 1).eq(0)
             .And(qa.bitwiseAnd(1 << 2).eq(0))
             .And(qa.bitwiseAnd(1 << 3).eq(0))
             .And(qa.bitwiseAnd(1 << 4).eq(0))
             .And(qa.bitwiseAnd(1 << 5).eq(0)))
    image_clear = image.updateMask(clear)
    sr = image_clear.select(["SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7"])
    reflectance = sr.multiply(0.0000275).add(-0.2)
    red, nir, swir = (reflectance.select(x) for x in ["SR_B4", "SR_B5", "SR_B6"])
    valid_red_nir = red.gte(0).And(red.lte(1)).And(nir.gte(0)).And(nir.lte(1))
    valid_nir_swir = nir.gte(0).And(nir.lte(1)).And(swir.gte(0)).And(swir.lte(1))
    ndvi_den = nir.add(red)
    ndbi_den = swir.add(nir)
    ndvi = (nir.subtract(red).divide(ndvi_den).updateMask(valid_red_nir)
            .updateMask(ndvi_den.gt(0.05)).rename("NDVI"))
    ndbi = (swir.subtract(nir).divide(ndbi_den).updateMask(valid_nir_swir)
            .updateMask(ndbi_den.gt(0.05)).rename("NDBI"))
    lst = (image_clear.select("ST_B10").multiply(0.00341802)
           .add(149.0).subtract(273.15).rename("LST_C"))
    lst_norm = lst.subtract(24).divide(24).clamp(0, 1)
    vegetation_norm = ee.Image(1).subtract(ndvi.clamp(-1, 1).add(1).divide(2))
    builtup_norm = ndbi.clamp(-1, 1).add(1).divide(2)
    heat_risk = (lst_norm.multiply(0.60).add(vegetation_norm.multiply(0.25))
                 .add(builtup_norm.multiply(0.15)).rename("Heat_Risk").clip(ROI))
    priority = (heat_risk.multiply(0.65).add(vegetation_norm.multiply(0.35))
                .rename("Mitigation_Priority").clip(ROI))
    stats = priority.reduceRegion(ee.Reducer.percentile([80]), ROI, 120, maxPixels=int(1e9)).getInfo()
    threshold = stats.get("Mitigation_Priority")
    if threshold is None:
        threshold = next((v for v in stats.values() if v is not None), None)
    if threshold is None:
        raise RuntimeError("Mitigation priority percentile could not be calculated.")
    zones = priority.gte(ee.Number(threshold)).selfMask().rename("Priority_Zones")
    print("80th percentile mitigation threshold:", threshold)

    save_map("01_satellite.html", [(reflectance, {"bands": ["SR_B4", "SR_B3", "SR_B2"], "min": 0, "max": 0.3}, "Landsat RGB")])
    save_map("02_indicators.html", [
        (ndvi, {"min": -1, "max": 1, "palette": ["8c510a", "f6e8c3", "01665e"]}, "NDVI Vegetation"),
        (ndbi, {"min": -1, "max": 1, "palette": ["2166ac", "f7f7f7", "b2182b"]}, "NDBI Built-up"),
        (lst, {"min": 24, "max": 48, "palette": ["313695", "ffffbf", "a50026"]}, "LST Celsius")])
    save_map("03_heat_risk.html", [(heat_risk, {"min": 0, "max": 1, "palette": ["2166ac", "ffffbf", "d7191c"]}, "Heat Risk")])
    save_map("04_mitigation_priority.html", [
        (priority, {"min": 0, "max": 1, "palette": ["2166ac", "ffffbf", "d7191c"]}, "Mitigation Priority"),
        (zones, {"palette": ["red"]}, "Top 20 percent priority zones")])

    training = ee.Image.cat([lst, ndvi, ndbi, heat_risk, ee.Image.pixelLonLat()])
    samples = training.sample(region=ROI, scale=120, numPixels=2500, seed=42,
                              dropNulls=True, geometries=False, tileScale=4)
    data = samples.getInfo()
    df = pd.DataFrame([f["properties"] for f in data["features"]]).dropna()
    required = ["LST_C", "NDVI", "NDBI", "Heat_Risk", "longitude", "latitude"]
    if df.empty or not set(required).issubset(df):
        raise RuntimeError("No usable satellite pixel samples returned by Earth Engine.")
    df = df[df.NDVI.between(-1, 1) & df.NDBI.between(-1, 1) & df.LST_C.between(-50, 80)].copy()
    df["Risk_Class"] = df.Heat_Risk.map(risk_class)
    print("Dataset shape:", df.shape, "\nClasses:\n", df.Risk_Class.value_counts())
    if df.Risk_Class.value_counts().min() < 2 or df.Risk_Class.nunique() < 2:
        raise RuntimeError("Too few samples per risk class for a stratified train/test split.")
    df.to_csv(OUT / "training_samples.csv", index=False)
    X = df[["LST_C", "NDVI", "NDBI"]]
    y = df["Risk_Class"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y)
    model = RandomForestClassifier(n_estimators=150, max_depth=12,
                                   min_samples_leaf=5, random_state=42,
                                   class_weight="balanced")
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    report = {"accuracy": float(accuracy_score(y_test, prediction)),
              "classification_report": classification_report(y_test, prediction, output_dict=True, zero_division=0),
              "limitation": "Labels derive from the weighted heat-risk formula, not independently observed ground truth."}
    (OUT / "evaluation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    joblib.dump(model, OUT / "random_forest_model.joblib")
    print("Random Forest accuracy:", round(report["accuracy"], 4))
    ConfusionMatrixDisplay.from_predictions(y_test, prediction, labels=["Low", "Medium", "High"], cmap="Blues")
    plt.title("Random Forest: Heat Risk Classification")
    plt.tight_layout()
    plt.savefig(OUT / "random_forest_confusion_matrix.png", dpi=180)
    plt.close()
    pd.Series(model.feature_importances_, index=X.columns).sort_values().plot.barh(title="Feature Importance")
    plt.tight_layout()
    plt.savefig(OUT / "feature_importance.png", dpi=180)
    plt.close()

    export_geotiff(ee.Image.cat([lst, ndvi, ndbi]).rename(["LST_C", "NDVI", "NDBI"]),
                   "bengaluru_features.tif", 120)
    # A six-band image in the exact band order the IPCV experiment expects.
    export_geotiff(reflectance.select(["SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7"]),
                   "landsat_multiband.tif", 120)
    export_geotiff(heat_risk, "heat_risk_map.tif", 120)
    export_geotiff(priority, "mitigation_priority_map.tif", 120)
    shutil.make_archive(str(ROOT / "urban_heat_project_outputs"), "zip", OUT)
    print("\nDONE. Outputs:", OUT)
    print("Next run: python urban_heat_ipcv_experiment_VSCODE.py")


if __name__ == "__main__":
    main()
