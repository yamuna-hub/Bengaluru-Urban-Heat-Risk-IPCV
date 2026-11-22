
import os
import numpy as np
import pandas as pd
import tifffile
import matplotlib.pyplot as plt

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)

# --------------------------------------------------
# 1. FILE PATHS
# --------------------------------------------------

CSV_FILE = "training_samples_with_texture.csv"
FEATURE_TIF = "bengaluru_features.tif"
TEXTURE_TIF = "06_texture_feature.tif"

OUTPUT_DIR = "xgboost_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --------------------------------------------------
# 2. LOAD TRAINING DATA
# --------------------------------------------------

print("\nLoading training dataset...")

df = pd.read_csv(CSV_FILE)

features = ["LST_C", "NDBI", "NDVI", "Texture"]
X = df[features]

# Convert class labels into numeric values
class_mapping = {
    "Low": 0,
    "Medium": 1,
    "High": 2
}

reverse_mapping = {
    0: "Low",
    1: "Medium",
    2: "High"
}

y = df["Risk_Class"].map(class_mapping)

print("Training samples:", len(df))
print("Features:", features)
print("Classes:", reverse_mapping)

# --------------------------------------------------
# 3. SPLIT DATA FOR MODEL EVALUATION
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

# --------------------------------------------------
# 4. TRAIN TEXTURE-ENHANCED XGBOOST
# --------------------------------------------------

print("\nTraining texture-enhanced XGBoost model...")

model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="multi:softprob",
    num_class=3,
    eval_metric="mlogloss",
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# --------------------------------------------------
# 5. EVALUATE MODEL
# --------------------------------------------------

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="weighted")
recall = recall_score(y_test, y_pred, average="weighted")
f1 = f1_score(y_test, y_pred, average="weighted")

print("\n========== XGBOOST RESULTS ==========")
print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Low", "Medium", "High"]
    )
)

# Save metrics
with open(os.path.join(OUTPUT_DIR, "model_metrics.txt"), "w") as f:
    f.write("Texture-Enhanced XGBoost Results\n")
    f.write(f"Accuracy: {accuracy:.4f}\n")
    f.write(f"Precision: {precision:.4f}\n")
    f.write(f"Recall: {recall:.4f}\n")
    f.write(f"F1 Score: {f1:.4f}\n")

# --------------------------------------------------
# 6. CONFUSION MATRIX IMAGE
# --------------------------------------------------

cm = confusion_matrix(y_test, y_pred)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Low", "Medium", "High"]
)

display.plot(cmap="Blues", values_format="d")
plt.title("Texture-Enhanced XGBoost - Confusion Matrix")
plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIR, "01_confusion_matrix.png"),
    dpi=300
)
plt.close()

# --------------------------------------------------
# 7. FEATURE IMPORTANCE IMAGE
# --------------------------------------------------

importance = model.feature_importances_

plt.figure(figsize=(8, 5))
plt.bar(features, importance)
plt.title("XGBoost Feature Importance")
plt.xlabel("Input Features")
plt.ylabel("Importance")
plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIR, "02_feature_importance.png"),
    dpi=300
)
plt.close()

# --------------------------------------------------
# 8. LOAD SATELLITE RASTER DATA
# --------------------------------------------------

print("\nLoading satellite raster images...")

with tifffile.TiffFile(FEATURE_TIF) as tif:
    feature_image = tif.asarray()
    feature_tags = tif.pages[0].tags

with tifffile.TiffFile(TEXTURE_TIF) as tif:
    texture_image = tif.asarray()
    texture_tags = tif.pages[0].tags

print("Feature raster shape:", feature_image.shape)
print("Texture raster shape:", texture_image.shape)

# Expected shapes:
# Feature raster: (280, 275, 3)
# Texture raster: (280, 275)

if feature_image.shape[:2] != texture_image.shape:
    raise ValueError("Feature and texture raster dimensions do not match!")

# --------------------------------------------------
# 9. CHECK GEOGRAPHIC ALIGNMENT
# --------------------------------------------------

feature_transform = feature_tags["ModelTransformationTag"].value

texture_scale = texture_tags["ModelPixelScaleTag"].value
texture_tiepoint = texture_tags["ModelTiepointTag"].value

if not np.allclose(
    feature_transform,
    (
        120.0, 0.0, 0.0, 765585.0,
        0.0, -120.0, 0.0, 1455375.0,
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 1.0
    )
):
    raise ValueError("Unexpected feature raster transformation!")

if not np.allclose(texture_scale[:2], [120.0, 120.0]):
    raise ValueError("Unexpected texture pixel size!")

if not np.allclose(
    texture_tiepoint[3:5],
    [765585.0, 1455375.0]
):
    raise ValueError("Texture raster geographic position does not match!")

print("Geographic alignment checks passed.")

# --------------------------------------------------
# 10. PREPARE PIXEL FEATURES
# --------------------------------------------------

lst = feature_image[:, :, 0]
ndbi = feature_image[:, :, 1]
ndvi = feature_image[:, :, 2]
texture = texture_image

height, width = lst.shape

pixel_features = np.column_stack([
    lst.ravel(),
    ndbi.ravel(),
    ndvi.ravel(),
    texture.ravel()
])

# Exclude invalid and NoData pixels
valid_mask = (
    np.isfinite(pixel_features).all(axis=1)
    & (lst.ravel() > 0)
)

valid_pixels = pixel_features[valid_mask]

print("Total raster pixels:", height * width)
print("Valid pixels:", len(valid_pixels))

# --------------------------------------------------
# 11. TRAIN FINAL MODEL USING ALL SAMPLES
# --------------------------------------------------

print("\nTraining final model using all training samples...")

final_model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="multi:softprob",
    num_class=3,
    eval_metric="mlogloss",
    random_state=42,
    n_jobs=-1
)

final_model.fit(X, y)

# --------------------------------------------------
# 12. PREDICT BENGALURU HEAT-RISK CLASSES
# --------------------------------------------------

print("\nPredicting heat-risk classes for Bengaluru...")

predicted_classes = final_model.predict(valid_pixels).astype(np.uint8)
probabilities = final_model.predict_proba(valid_pixels)

confidence = np.max(probabilities, axis=1) * 100

# 0 = NoData, 1 = Low, 2 = Medium, 3 = High
risk_map = np.zeros(height * width, dtype=np.uint8)
risk_map[valid_mask] = predicted_classes + 1
risk_map = risk_map.reshape(height, width)

confidence_map = np.full(height * width, np.nan, dtype=np.float32)
confidence_map[valid_mask] = confidence
confidence_map = confidence_map.reshape(height, width)

# --------------------------------------------------
# 13. SAVE GEOGRAPHIC TIFF OUTPUT
# --------------------------------------------------

print("\nSaving georeferenced heat-risk TIFF...")

geo_tags = [
    (
        34264, "d", 16,
        tuple(float(v) for v in feature_transform),
        False
    ),
    (
        34735, "H",
        len(feature_tags["GeoKeyDirectoryTag"].value),
        tuple(feature_tags["GeoKeyDirectoryTag"].value),
        False
    ),
    (42113, "s", 2, "0", False)
]

risk_tif_path = os.path.join(
    OUTPUT_DIR,
    "03_bengaluru_xgboost_heat_risk.tif"
)

tifffile.imwrite(
    risk_tif_path,
    risk_map,
    photometric="minisblack",
    metadata=None,
    extratags=geo_tags
)

# --------------------------------------------------
# 14. GENERATE HEAT-RISK MAP IMAGE
# --------------------------------------------------

from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch

risk_colors = ["#000000", "#2E86DE", "#F1C40F", "#E74C3C"]

cmap = ListedColormap(risk_colors)
norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)

plt.figure(figsize=(10, 8))

plt.imshow(
    np.ma.masked_equal(risk_map, 0),
    cmap=cmap,
    norm=norm
)

legend_items = [
    Patch(facecolor="#2E86DE", label="Low Risk"),
    Patch(facecolor="#F1C40F", label="Medium Risk"),
    Patch(facecolor="#E74C3C", label="High Risk")
]

plt.legend(handles=legend_items, loc="upper right")
plt.title("Bengaluru Heat-Risk Map - Texture-Enhanced XGBoost")
plt.axis("off")
plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "04_bengaluru_xgboost_heatmap.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# --------------------------------------------------
# 15. GENERATE CONFIDENCE MAP IMAGE
# --------------------------------------------------

plt.figure(figsize=(10, 8))

image = plt.imshow(
    confidence_map,
    cmap="viridis",
    vmin=0,
    vmax=100
)

plt.colorbar(image, label="Prediction Confidence (%)")
plt.title("XGBoost Heat-Risk Prediction Confidence")
plt.axis("off")
plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "05_xgboost_confidence_map.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# --------------------------------------------------
# 16. SAVE RISK CLASS DISTRIBUTION
# --------------------------------------------------

class_counts = pd.Series(
    predicted_classes
).map(reverse_mapping).value_counts()

plt.figure(figsize=(7, 5))
class_counts.reindex(["Low", "Medium", "High"]).plot(
    kind="bar"
)

plt.title("Predicted Bengaluru Heat-Risk Class Distribution")
plt.xlabel("Risk Class")
plt.ylabel("Number of Pixels")
plt.xticks(rotation=0)
plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "06_risk_class_distribution.png"),
    dpi=300
)

plt.close()

# --------------------------------------------------
# 17. FINISHED
# --------------------------------------------------

print("\n======================================")
print("XGBOOST HEAT-RISK GENERATION COMPLETED")
print("======================================")

print("\nGenerated files:")

for filename in sorted(os.listdir(OUTPUT_DIR)):
    print(filename)

print("\nOutput folder:", os.path.abspath(OUTPUT_DIR))
