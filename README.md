# MLPA Digital Twin Survival Analysis for NSCLC

A comprehensive survival analysis framework integrating **Multi-Level Parameterized Automata (MLPA) ** simulations with radiomics, clinical, and deep learning features for **Non-Small Cell Lung Cancer (NSCLC)** prognosis prediction.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> ** [Try the Interactive Demo](https://smartdrugdiscovery.org/mlpa)** - Visualize 2-D tumor growth simulation in your browser!

---

## 🎯 Overview

This implementation replicates **Gu J et al. (2025)** methodology and extends it with:
- **MLPA Integration**: 3-D cellular automaton simulations of tumor growth
- **Multiple Models**: CoxPH, Neural Cox (DeepSurv), Gradient Boosting, Ensemble
- **Integrated AUC (iAUC)**: Time-dependent discrimination assessment
- **Sensitivity Analysis**: Parameter robustness testing

**Pipeline:**
```
CT Scans → Radiomics → Biological Parameters (α, β) → 3D Cellular Automaton → Growth Features
```

---

## ✨ Features

- 🔬 **4 Survival Models**: CoxPH, Neural Cox, Gradient Boosting, Ensemble
- 📊 **4 Metrics**: C-Index, p-value, Hazard Ratio, iAUC
- 🧬 **4 Data Domains**: Clinical (7), Radiomics (1706), AutoEncoder (512), Digital Twin (6)
- 🔀 **2 Fusion Methods**: Feature-level (Table 5), Signature-level (Table 6)
- 🔍 **Sensitivity Analysis**: α/β parameter perturbation (±10%, ±20%)
- 📈 **Kaplan-Meier Curves**: Risk stratification visualization

---

## 📋 Quick Start

### Installation
```bash
# Core dependencies (required)
pip install numpy pandas scikit-learn matplotlib lifelines tqdm

# Optional (for full functionality)
pip install torch scikit-survival  # Neural Cox + iAUC
```

### Data Structure
```
project_root/
├── data3d_1706features/outputs/
│   ├── features_normalized_standard.npy
│   ├── feature_names_normalized_standard.json
│   └── patient_ids_normalized_standard.json
├── nsclc/NSCLC-Radiomics-Lung1.clinical-version3-Oct-2019.csv
├── data_3d_ae_parallel/features/features_deep_512.csv
└── digital_twin_output/all_patients_biological_parameters.csv
```

### Run Analysis
```bash
python ferretti_3d_mlpa_iauc_neural_ensemble.py
```

**Runtime:** 10-20 minutes (with Neural Cox), 2-5 minutes (CoxPH only)

---

## 📊 Key Configuration

Edit at top of script:
```python
# Feature selection
TOP_K_AE = 15
TOP_K_RADIOMICS = 10
CORRELATION_THRESHOLD = 0.70

# Cross-validation
N_SPLITS = 5
RANDOM_STATE = 42

# Neural Cox
NEURAL_COX_HIDDEN_LAYERS = [64, 32]
NEURAL_COX_DROPOUT = 0.3
NEURAL_COX_EPOCHS = 100

# Ensemble weights
ENSEMBLE_WEIGHTS = {
    'coxph': 0.4,
    'neural_cox': 0.3,
    'gradient_boosting': 0.3
}

# Model selection
RUN_COXPH = True
RUN_NEURAL_COX = True
RUN_ENSEMBLE = True
CREATE_KM_PLOTS = True

# iAUC time points (years)
IAUC_TIME_POINTS = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
```

---

## 📁 Outputs

All saved to `./ferretti_with_3d_mlpa_iauc_output/`:

| File | Description |
|------|-------------|
| `results_with_3d_mlpa_iauc_multimodel.csv` | Complete results (C-Index, HR, iAUC) |
| `results_pivot_by_model.csv` | Model comparison table |
| `results_sorted_by_iauc.csv` | Ranked by integrated AUC |
| `km_plots/*.png` | Kaplan-Meier curves (high vs low risk) |
| `sensitivity_analysis/sensitivity_summary.csv` | Parameter perturbation results |
| `sensitivity_analysis/*.png` | Robustness visualizations |

---

## 📈 Example Results
```
┌─────────────────────────────────┬──────────┬────────────────┬────────────┬────────────┬────────────────┐
│ Configuration                   │ Model    │ C-Index        │ p-value    │ HR         │ iAUC           │
├─────────────────────────────────┼──────────┼────────────────┼────────────┼────────────┼────────────────┤
│ Table5_DT_AE_R_C                │ ensemble │ 0.6523±0.0287  │  0.0012    │ 2.84±0.45  │ 0.6891±0.0312  │
│ Table5_DT_R_C                   │ neural   │ 0.6401±0.0295  │  0.0024    │ 2.67±0.38  │ 0.6734±0.0289  │
│ Table6_DT_AE_R_C                │ coxph    │ 0.6278±0.0301  │  0.0031    │ 2.54±0.41  │ 0.6612±0.0276  │
└─────────────────────────────────┴──────────┴────────────────┴────────────┴────────────┴────────────────┘
```

**Interpretation:**
- C-Index > 0.7: Excellent | 0.6-0.7: Good | 0.5-0.6: Moderate
- iAUC > 0.7: Excellent | 0.6-0.7: Good | 0.5-0.6: Moderate
- p-value < 0.05: Significant | HR > 2.0: Strong risk stratification

---

## 🧠 Features

The 3-D MLPA generates 6 features from radiomics:

| Feature | Description | Derivation |
|---------|-------------|------------|
| `Bio_Alpha` | Proliferation rate (α) | `0.01 + 0.05 × entropy` |
| `Bio_Necrosis` | Necrosis probability (β) | `0.01 + 0.08 × (1 - sphericity)` |
| `Sim_Growth_Rate` | Simulated expansion rate | 3D cellular automaton |
| `Sim_Necrosis_Ratio` | Necrotic cell fraction | 3D cellular automaton |
| `Radiomics_Entropy` | Tumor heterogeneity | First-order histogram |
| `Radiomics_Sphericity` | Shape regularity | Surface area / volume |

**Why MLPA?**
- ✅ Biological interpretability (α, β have clinical meaning)
- ✅ Physics-based constraints (realistic growth dynamics)
- ✅ Patient-specific personalization
- ✅ Longitudinal prediction capability

---

## 🔬 Model Comparison

| Model | Pros | Cons | Use Case |
|-------|------|------|----------|
| **CoxPH** | Fast, interpretable, established | Linear assumptions | Baseline, coefficient interpretation |
| **Neural Cox** | Non-linear, flexible | Black box, slower | Complex interactions |
| **Gradient Boosting** | Robust, handles interactions | Less interpretable | Tree-based ensemble |
| **Ensemble** | Best of all, reduces variance | Slower, more complex | Best performance |

---

## 🔍 Sensitivity Analysis

Tests model robustness by perturbing α and β by ±10%, ±20%:

**Metrics:**
- Risk score changes (absolute & percentage)
- Rank preservation (Spearman ρ)
- Risk category changes (% switching high/low)
- C-Index stability (ΔC-Index)

**Robust model criteria:**
- Spearman ρ > 0.95
- Category changes < 10%
- |ΔC-Index| < 0.02

---

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| "scikit-survival not installed" | `pip install scikit-survival` (enables iAUC) |
| "PyTorch not installed" | `pip install torch` (enables Neural Cox) |
| "Digital Twin file not found" | Script continues without DT features |
| Low C-Index (<0.55) | Increase `TOP_K_AE`, `TOP_K_RADIOMICS` |
| Out of memory | Reduce `NEURAL_COX_BATCH_SIZE` to 16 |
| Slow execution | Set `RUN_NEURAL_COX=False`, `N_SPLITS=3` |

---

## 📚 Citation
```bibtex
@software{phong2026mlpa,
  author       = {Huu Phong Nguyen, Delower Hossain, Ehsan Saghapour, Zhandos Sembay, Jake Y. Chen 
},
  title        = {{MLPA: A Framework Modeling for Non-Small Cell Lung Cancer Survival Prediction 
}},
  year         = {2026},
  institution  = {Systems Pharmacology AI Research Center (SPARC), 
                  University of Alabama at Birmingham},
  type         = {Software Framework},
  url          = {[https://github.com/aimed-lab/MLPA_abstract](https://github.com/aimed-lab/MLPA_abstract)},
  note         = {3-D Multi-Level Parameterized Automata for NSCLC survival prediction}
}
```

---

## 📧 Contact

**Author:** Huu Phong Nguyen (hnguye24 AT uab DOT edu)

**Affiliation:** SPARC (Systems Pharmacology AI Research Center), UAB  

**Project:** Lung Cancer MLPA Framework

---

## ⚖️ License

MIT License - See LICENSE file for details

---

**Built with ❤️ for advancing precision oncology through mechanistic AI**
