BENGALURU URBAN HEAT RISK - VS CODE DEMO

Files:
- urban_heat_risk_bengaluru_cat2_VSCODE.py (run first: Earth Engine, maps, ML and TIFF exports)
- urban_heat_ipcv_experiment_VSCODE.py (run second: OpenCV preprocessing + fair baseline/texture comparison)
- requirements_urban_heat.txt

WINDOWS SETUP (VS Code terminal inside the project folder):
  py -3 -m venv .venv
  .venv\Scripts\python.exe -m pip install -r requirements_urban_heat.txt

Run:
  .venv\Scripts\python.exe urban_heat_risk_bengaluru_cat2_VSCODE.py
  .venv\Scripts\python.exe urban_heat_ipcv_experiment_VSCODE.py

On first use, Earth Engine may open a browser for authorization. Log in with a Google account
that has Earth Engine access to the configured Cloud project, fair-veld-492517-s0.
Change PROJECT_ID in the first file only if this is not your authorized Cloud project.

The first script may take time to export satellite GeoTIFF files over the network.
It saves HTML maps and trained model / evaluation files under urban_heat_project_outputs.
The second script saves preprocessing previews and comparisons under IPCV_Experiment.
Open the .html maps in a browser for the live presentation (internet required for live EE tiles).

NOTE:
- This is a reorganized clean local pipeline, not a cell-for-cell translation. The former
  notebook had invalid Python syntax and non-reproducible extra experiments.
- The first script keeps the original image filters, ROI, heat-risk weights, RF hyperparameters.
- The second script compares 4 models using the same coordinate-matched samples for baseline
  vs texture features, but results need not reproduce previous Colab metric numbers.
- The texture feature may not improve accuracy; models learn risk classes generated from a
  weighted score rather than validating true field-measured heat risk.
- The code has been syntax-checked but Earth Engine API calls cannot be exercised here without
  your authorized credentials, imagery exports and network access.
