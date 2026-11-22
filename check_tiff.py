
import tifffile
import pandas as pd

# Check georeferencing information
for filename in ["bengaluru_features.tif", "06_texture_feature.tif"]:
    print("\n==========", filename, "==========")

    with tifffile.TiffFile(filename) as tif:
        page = tif.pages[0]

        for tag_name in [
            "ModelTransformationTag",
            "ModelPixelScaleTag",
            "ModelTiepointTag",
            "GeoKeyDirectoryTag",
            "GeoAsciiParamsTag",
            "GDAL_NODATA"
        ]:
            tag = page.tags.get(tag_name)

            if tag is not None:
                print(tag_name, "=", tag.value)

# Check training CSV
print("\n========== TRAINING CSV ==========")

df = pd.read_csv("training_samples_with_texture.csv")

print("Columns:", df.columns.tolist())
print("Rows:", len(df))
print("Risk classes:")
print(df["Risk_Class"].value_counts())
print("\nMissing values:")
print(df.isnull().sum())

