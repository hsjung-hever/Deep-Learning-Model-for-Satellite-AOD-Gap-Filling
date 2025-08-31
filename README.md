# Deep-Learning-Model-for-Satellite-AOD-Gap-Filling

[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.11-orange.svg)](https://www.tensorflow.org/)

## What is this repository for?
Supplementary code for research on **Multivariate Deep Learning Models for Satellite AOD Gap Filling**.  Includes modular training and evaluation pipelines with U-Net, supporting univariate/multivariate settings  and reproducible experiments on different training gap sizes.


---

## Who do I talk to?
**Contact:**  
Haesoo Jung  
E-mail: hsjung0731@gmail.com  

---

## Usage
1. Prepare training and testing NetCDF datasets (`train_.nc`, `test_all.nc`).  
   - The datasets should contain Aerosol Optical Depth (AOD) and related auxiliary variables (RA, ZA, VZ, TR, RAD, UV, VIS).  

2. Install dependencies:  
   ```bash
   pip install -r requirements.txt
   ```

3. Train the AOD gap-filling model:

   ```bash
   python scripts/runmodel_dist.py --nc-path ./data/train_.nc --save-dir ./outputs
   ```

4. Evaluate the trained models:

   ```bash
   python scripts/test.py --nc-path ./data/test_all.nc --load-dir ./outputs
   ```

5. Results (trained models, training logs, evaluation metrics, figures) are saved under the `outputs/` directory.

---

## Dataset Preparation

1. **Training data (`train_.nc`)**

   * Must contain Aerosol Optical Depth (AOD) and auxiliary variables.
   * Example variables: `aod`, `lat`, `lon`, `RA`, `ZA`, `VZ`, `tr`, `rad`, `uv`, `vis`.

   Example structure:

   ```
   Dimensions:
       time: 1000
       lat: 32
       lon: 32
   Variables:
       aod(time, lat, lon)
       RA(time, lat, lon)
       ZA(time, lat, lon)
       VZ(time, lat, lon)
       tr(time, lat, lon)
       rad(time, lat, lon)
       uv(time, lat, lon)
       vis(time, lat, lon)
   ```

2. **Testing data (`test_all.nc`)**

   * Same structure as training data.
   * Gap regions are simulated automatically during evaluation (`--evaluate-gaps`).

3. **Data placement**

   ```
   data/
   ├─ train_.nc       # Training dataset
   └─ test_all.nc     # Testing dataset
   ```

---

## Code introduction

**Module introduction:**

1. `scripts/runmodel_dist.py`

   * Entry point for model training (univariate & multivariate U-Net).

2. `scripts/test.py`

   * Entry point for model evaluation and analysis (RMSE, SSIM, visualizations).

3. `src/aod_gapfill/config.py`

   * Handles configuration and hyperparameter parsing (CLI & YAML).

4. `src/aod_gapfill/data.py`

   * Dataset loader, normalization, patch generation, and gap simulation.

5. `src/aod_gapfill/model.py`

   * Defines U-Net architecture for image-based gap filling.

6. `src/aod_gapfill/metrics.py`

   * Provides evaluation metrics: RMSE, SSIM, normalization utilities.

7. `src/aod_gapfill/viz.py`

   * Visualization utilities for inputs, ground-truth, predictions.

8. `src/aod_gapfill/train.py`

   * Training pipeline for univariate and multivariate models.

9. `src/aod_gapfill/eval.py`

   * Evaluation pipeline across different training/testing gap sizes.

10. `outputs/`

    * Stores trained models, training history, evaluation results, and figures.

---

## Data Availability

The datasets used in this study are **not included** in this repository due to their large size.
Please prepare your own NetCDF datasets (`train_.nc`, `test_all.nc`) following the structure described in the *Dataset Preparation* section.

If you need access to the original data provided by the Environmental Satellite Center (NIER), please refer to the official data service channels.

---

## Acknowledgement

This study was conducted using data from the geostationary environmental satellite provided by the Environmental Satellite Center of the National Institute of Environmental Research (NIER) and was supported by NIER Project No. NIER-2025-01-03-003, *“Operation of Geostationary Environmental Satellite Ground Station and Satellite Data Service.”*

---

## License

This code is provided as supplementary material for academic reference only.
**All rights reserved. Redistribution and commercial use are not permitted.**

