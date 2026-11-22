"""IPCV preprocessing and optional fair ML comparison, VS Code edition.
Run the urban heat Earth Engine script first to generate source TIFF and CSV files.
"""
from pathlib import Path
import argparse
import zipfile
import numpy as np
import pandas as pd
import cv2
import rasterio
from rasterio.warp import transform as transform_coords
from scipy.ndimage import uniform_filter
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "urban_heat_project_outputs"
OUT = ROOT / "IPCV_Experiment"


def stretch_band(band, mask=None):
    valid = np.isfinite(band)
    if mask is not None:
        valid &= mask
    if not np.any(valid):
        raise ValueError("Image contains no valid pixels.")
    lo, hi = np.percentile(band[valid], [2, 98])
    if hi <= lo:
        return np.zeros(band.shape, dtype=np.uint8)
    result = np.clip((band - lo) / (hi - lo), 0, 1)
    result[~valid] = 0
    return (result * 255).astype(np.uint8)


def preprocess(path):
    OUT.mkdir(exist_ok=True)
    with rasterio.open(path) as src:
        if src.count < 3:
            raise ValueError("Requires Landsat multispectral TIFF with B2,B3,B4 as its first bands.")
        image = src.read([1, 2, 3]).astype(np.float32)
        profile = src.profile.copy()
        valid = src.dataset_mask() > 0
        print("Input", path.name, "bands", src.count, "size", src.width, "x", src.height)
    # Input first bands must be B2, B3, B4 in that order, matching original experiment.
    rgb_display = np.stack([stretch_band(image[i], valid) for i in [2, 1, 0]], axis=-1)
    median = cv2.medianBlur(rgb_display, 3)
    lab = cv2.cvtColor(median, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = cv2.cvtColor(cv2.merge((clahe.apply(l), a, b)), cv2.COLOR_LAB2RGB)
    gray = cv2.cvtColor(enhanced, cv2.COLOR_RGB2GRAY)
    sx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    sobel = np.nan_to_num(np.sqrt(sx**2 + sy**2))
    sobel_display = (sobel / sobel.max() * 255).astype(np.uint8) if sobel.max() > 0 else np.zeros_like(gray)
    g = gray.astype(np.float32)
    texture = np.maximum(uniform_filter(g*g, 7) - uniform_filter(g, 7)**2, 0)
    texture[~valid] = np.nan
    texture_display = stretch_band(texture)
    outputs = {
        "01_original_rgb.png": rgb_display,
        "02_median_filtered.png": median,
        "03_clahe_enhanced.png": enhanced,
        "04_sobel_edges.png": sobel_display,
        "05_local_texture.png": texture_display,
    }
    for filename, array in outputs.items():
        plt.imsave(OUT / filename, array, cmap="gray" if array.ndim == 2 else None)
    profile.update(count=1, dtype="float32", nodata=np.nan)
    with rasterio.open(OUT / "06_texture_feature.tif", "w", **profile) as dst:
        dst.write(texture.astype("float32"), 1)
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for ax, (filename, array) in zip(axes.flat, outputs.items()):
        ax.imshow(array, cmap="gray" if array.ndim == 2 else None)
        ax.set_title(filename.removesuffix(".png"))
        ax.axis("off")
    axes.flat[-1].axis("off")
    plt.tight_layout()
    plt.savefig(OUT / "07_preprocessing_comparison.png", dpi=170)
    plt.close()
    (OUT / "IPCV_report.txt").write_text(
        "Original input: " + str(path) + "\nMedian filtering\nCLAHE enhancement\n"
        "Sobel edge detection\nLocal variance texture extraction\n"
        "Original Earth Engine model and scores not modified.\n", encoding="utf-8")
    print("IPCV images saved in", OUT)


def compare_models():
    """Use the SAME sampled rows in baseline/texture model comparisons.
    Texture values are sampled by geographic coordinates; checks prevent a false comparison.
    """
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.pipeline import Pipeline
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.svm import SVC
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    from xgboost import XGBClassifier

    csv_path = SOURCE / "training_samples.csv"
    texture_path = OUT / "06_texture_feature.tif"
    feature_path = SOURCE / "bengaluru_features.tif"
    if not csv_path.exists() or not feature_path.exists():
        print("Skipping comparison: main script's training_samples.csv or features TIFF is missing.")
        return
    df = pd.read_csv(csv_path).dropna(subset=["LST_C", "NDVI", "NDBI", "Risk_Class", "longitude", "latitude"])
    with rasterio.open(texture_path) as src, rasterio.open(feature_path) as feat:
        # Georeferenced TIFFs must represent the same geographic area.
        if src.crs != feat.crs or src.transform != feat.transform or src.shape != feat.shape:
            raise ValueError("Texture raster is not aligned with feature raster. Do not compare until aligned.")
        xs, ys = transform_coords("EPSG:4326", src.crs,
                                  df.longitude.to_list(), df.latitude.to_list())
        samples = np.array([a[0] for a in src.sample(zip(xs, ys), masked=True)], dtype=float)
    df["Texture"] = samples
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["Texture"]).copy()
    if len(df) < 30 or df.Risk_Class.nunique() < 2 or df.Risk_Class.value_counts().min() < 4:
        print("Skipping comparison: too few valid coordinate-matched texture samples.")
        return
    df.to_csv(OUT / "training_samples_with_texture.csv", index=False)
    target = LabelEncoder().fit_transform(df["Risk_Class"])
    indices = np.arange(len(df))
    train_idx, test_idx = train_test_split(indices, test_size=0.25, random_state=42, stratify=target)
    results = []
    for featureset, columns in [("Baseline", ["LST_C", "NDBI", "NDVI"]),
                                ("Texture", ["LST_C", "NDBI", "NDVI", "Texture"])]:
        X = df[columns].astype(float)
        models = {
            "XGBoost": XGBClassifier(n_estimators=150, max_depth=6, learning_rate=0.1,
                                      subsample=0.8, colsample_bytree=0.8,
                                      objective="multi:softprob", eval_metric="mlogloss", random_state=42),
            "Random Forest": RandomForestClassifier(n_estimators=150, max_depth=12,
                              min_samples_leaf=5, random_state=42, class_weight="balanced"),
            "Decision Tree": DecisionTreeClassifier(max_depth=12, min_samples_leaf=5,
                               random_state=42, class_weight="balanced"),
            "SVM": Pipeline([("scaler", StandardScaler()),
                              ("svm", SVC(kernel="rbf", C=1.0, class_weight="balanced"))]),
        }
        for name, model in models.items():
            model.fit(X.iloc[train_idx], target[train_idx])
            pred = model.predict(X.iloc[test_idx])
            results.append({"Features": featureset, "Model": name,
                            "Accuracy": accuracy_score(target[test_idx], pred),
                            "Precision": precision_score(target[test_idx], pred, average="macro", zero_division=0),
                            "Recall": recall_score(target[test_idx], pred, average="macro", zero_division=0),
                            "F1 Score": f1_score(target[test_idx], pred, average="macro", zero_division=0)})
            print(featureset, name, "accuracy", round(results[-1]["Accuracy"], 4))
    table = pd.DataFrame(results)
    table.to_csv(OUT / "model_comparison.csv", index=False)
    for metric in ["Accuracy", "Precision", "Recall", "F1 Score"]:
        pivot = table.pivot(index="Model", columns="Features", values=metric)
        pivot.plot.bar(figsize=(9, 5), rot=0, title=f"{metric}: baseline vs texture")
        plt.ylabel(metric)
        plt.tight_layout()
        plt.savefig(OUT / f"{metric.lower().replace(' ', '_')}_comparison.png", dpi=200)
        plt.close()
    print("All model comparison results saved:", OUT)
    print("NOTE: labels are generated from the heat-risk score, not independent field observations.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=SOURCE / "landsat_multiband.tif",
                        help="Original Landsat multiband TIFF with first bands B2/B3/B4")
    parser.add_argument("--skip-models", action="store_true")
    args = parser.parse_args()
    if not args.input.is_file():
        raise FileNotFoundError(f"Missing {args.input}. Run urban_heat_risk_bengaluru_cat2_VSCODE.py first, or supply --input PATH.")
    preprocess(args.input)
    if not args.skip_models:
        compare_models()
    with zipfile.ZipFile(ROOT / "IPCV_Experiment.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for p in OUT.iterdir():
            if p.is_file():
                z.write(p, arcname=p.name)
    print("DONE. ZIP:", ROOT / "IPCV_Experiment.zip")


if __name__ == "__main__":
    main()
