BENGALURU HEAT INTELLIGENCE — FLASK DASHBOARD
================================================

Runs on Python only. No React, Node.js, npm, Vite or Rollup.
Reads your existing real images, CSV results and Google Earth Engine/Folium HTML maps.
Does not change or retrain your models.

1. Extract the Bengaluru_Heat_Flask_Dashboard folder INSIDE your existing
   Bengaluru_HeatRisk_XGBoost folder, beside IPCV_Experiment/ and
   urban_heat_project_outputs/.

2. Open VS Code PowerShell and run each line separately:

cd "C:\Users\yamuna vijayaragavan\Downloads\Bengaluru_HeatRisk_XGBoost"
.\.venv\Scripts\Activate.ps1
python -m pip install -r .\Bengaluru_Heat_Flask_Dashboard\requirements.txt
cd .\Bengaluru_Heat_Flask_Dashboard
python app.py

3. Open in browser:
   http://127.0.0.1:5000

4. Press Ctrl+C in the terminal to stop.

PAGES
Overview / Satellite Intelligence / IPCV Processing Lab /
ML-DL Analytics / Heat-Risk Explorer / Mitigation Planner / Methodology.

IMPORTANT
- Missing artifacts are shown as missing; no fabricated experiment data.
- This is a LOCAL-ONLY demonstration server; keep it on 127.0.0.1.
- Folium HTML maps may need internet for external map tiles.
- All image/map paths are allowlisted in app.py.
- For privacy, keep project files, credentials, .venv, data downloads and
  any sensitive paths out of public GitHub commits.
- If the saved maps do not show, verify their names in urban_heat_project_outputs/.
- If the metrics do not show, verify IPCV_Experiment/ml_vs_dl_comparison.csv.
- This dashboard is for inspection of saved results; no new prediction is run.

Alternate root location (if you extract elsewhere):
  $env:URBAN_HEAT_PROJECT_ROOT="C:\path\to\Bengaluru_HeatRisk_XGBoost"
  python app.py

VERSION 2: XGBOOST-FIRST DISPLAY
- XGBoost charts and maps are read dynamically from xgboost_outputs/ (PNG/JPG/WEBP).
- Satellite LST / NDVI / NDBI tiles link to actual urban_heat_project_outputs/02_indicators.html;
  inside the map use layer controls to toggle the three layers.
- Satellite indicators are spectral inputs; they cannot truthfully be called XGBoost model images.
- Overview prefers XGBoost heatmap/confidence image when available, with explicit model label.
- Existing heat-risk Folium and mitigation maps remain displayed and labeled as
  satellite-derived indices rather than model-generated XGBoost predictions.
- No fake PNGs, no downloads from unknown sources, no retraining or backend changes.
- After replacing previous Flask folder, stop the server (Ctrl+C), run python app.py,
  and hard-refresh the browser (Ctrl+F5).
