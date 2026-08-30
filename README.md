# Robust Heart Disease Risk Prediction with Machine Learning

This repository contains the data, Python implementation, and experimental outputs accompanying the manuscript:

> **Robust Heart Disease Risk Prediction Using Machine Learning with Multi-Source Data Integration and Nested Cross-Validation**

The study compares eight established machine learning classifiers on a harmonized dataset assembled from three public heart disease data sources. Its primary contribution is a reproducible, leakage-controlled evaluation framework rather than a new classification algorithm.

## Study overview

Three heterogeneous structured clinical datasets were semantically harmonized using 11 common predictors. After removing exact duplicate clinical records, the analysis retained **1,220 unique observations** from **2,518 initial records**.

The experimental design includes:

- semantic harmonization of categorical feature codes across sources;
- exact duplicate removal before model development;
- deterministic recoding of physiologically implausible values as missing;
- fold-specific median and mode imputation;
- target-independent data augmentation applied only to training partitions;
- standardization of numerical variables within each training fold;
- hyperparameter optimization using grid search;
- an independent stratified 40% holdout test set;
- 10-fold outer and 5-fold inner nested cross-validation;
- repeated stratified 10-fold cross-validation with five repetitions;
- confidence intervals and paired statistical model comparisons;
- discrimination, calibration, and feature-importance analyses.

## Machine learning models

The following classifiers are evaluated using the same training partitions and leakage-controlled pipeline:

1. Logistic Regression
2. K-Nearest Neighbors
3. Support Vector Machine
4. Decision Tree
5. Random Forest
6. Ridge Classifier
7. LightGBM
8. XGBoost

## Data sources

The analysis integrates the following publicly available datasets:

1. [Heart Disease Dataset, Kaggle](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset)
2. [Heart Attack Dataset, Kaggle](https://www.kaggle.com/datasets/pritsheta/heart-attack)
3. [Heart Disease Dataset (Comprehensive), IEEE DataPort](https://doi.org/10.21227/dz4t-cm36)

Users are responsible for reviewing and complying with the terms and licenses specified by the original data providers. Access to the IEEE DataPort source may require user authentication.

## Harmonized variables

| Variable | Description | Coding or unit |
|---|---|---|
| `age` | Patient age | Years |
| `sex` | Sex | 0 = female, 1 = male |
| `cp` | Chest pain type | 0 to 3 |
| `trestbps` | Resting blood pressure | mmHg |
| `chol` | Serum cholesterol | mg/dL |
| `fbs` | Fasting blood sugar above 120 mg/dL | 0 = false, 1 = true |
| `restecg` | Resting electrocardiographic result | 0 to 2 |
| `thalach` | Maximum heart rate achieved | Beats/min |
| `exang` | Exercise-induced angina | 0 = no, 1 = yes |
| `oldpeak` | Exercise-induced ST depression relative to rest | Continuous |
| `slope` | Slope of the peak exercise ST segment | 0 to 2 |
| `target` | Heart disease status | 0 = absent, 1 = present |

The `ca` and `thal` variables are intentionally excluded because they are not consistently available across all three sources.

## Repository structure

```text
HeartDiseaseRiskPredictionML/
├── Codes/
│   └── V9.py
├── Dataset/
│   ├── Heart Attack Data Set.csv
│   ├── heart.csv
│   ├── heart_statlog_cleveland_hungary_final.csv
│   └── dataset.csv
└── Results/
    ├── final_holdout_test_results.xlsx
    ├── nested_cv_results_with_95CI.xlsx
    ├── nested_cv_fold_level_results.xlsx
    ├── repeated_kfold_results.xlsx
    ├── friedman_test_results.xlsx
    ├── pairwise_wilcoxon_tests.xlsx
    ├── random_forest_feature_importance.xlsx
    ├── calibration_curves.png
    ├── roc_curves_all_models.png
    └── ...
```

`Codes/V9.py` reads the three source files listed in the Data sources section. The additional `Dataset/dataset.csv` file is retained in the repository but is not read by the V9 analysis script.

## Installation

Create and activate a Python virtual environment, then install the required packages.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install pandas numpy scipy scikit-learn imbalanced-learn lightgbm xgboost matplotlib openpyxl
```

On Windows PowerShell, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

## Running the analysis

The current script reads input CSV files from its working directory. The commands below create an isolated run directory and copy the required files without modifying the original datasets.

### Linux or macOS

```bash
mkdir -p run
cp Codes/V9.py run/
cp Dataset/heart.csv run/
cp Dataset/heart_statlog_cleveland_hungary_final.csv run/
cp "Dataset/Heart Attack Data Set.csv" run/Heart_Attack_Data_Set.csv
cd run
python V9.py
```

### Windows PowerShell

```powershell
New-Item -ItemType Directory -Force run
Copy-Item Codes\V9.py run\
Copy-Item Dataset\heart.csv run\
Copy-Item Dataset\heart_statlog_cleveland_hungary_final.csv run\
Copy-Item "Dataset\Heart Attack Data Set.csv" "run\Heart_Attack_Data_Set.csv"
Set-Location run
python V9.py
```

New outputs are written to `run/V9_results/`. The nested and repeated grid-search procedures evaluate many model configurations and may require substantial processing time and memory, particularly when parallel execution uses all available CPU cores.

## Leakage-control strategy

The independent holdout set is isolated before model fitting and is not used for preprocessing estimation, augmentation, hyperparameter selection, or model selection. Within cross-validation, the following operations are fitted or applied only on each training partition:

1. median imputation for `trestbps` and `chol`;
2. mode imputation for `slope`;
3. target-independent augmentation of approximately 50% additional training observations;
4. standardization of numerical predictors;
5. model fitting and hyperparameter optimization.

The augmentation procedure applies small, clinically bounded perturbations to age, cholesterol, resting blood pressure, maximum heart rate, and ST depression. Categorical variables are left unchanged. Validation and test observations are never augmented.

## Main results

### Independent holdout test

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC | Brier score |
|---|---:|---:|---:|---:|---:|---:|
| Random Forest | **0.8156** | 0.7954 | 0.8959 | **0.8427** | **0.8829** | 0.1419 |
| LightGBM | **0.8156** | **0.8034** | 0.8810 | 0.8404 | 0.8808 | **0.1377** |
| XGBoost | 0.7951 | 0.7735 | 0.8885 | 0.8270 | 0.8765 | 0.1606 |
| SVM | 0.7807 | 0.7893 | 0.8216 | 0.8051 | 0.8590 | 0.1482 |
| Decision Tree | 0.7766 | 0.7312 | **0.9405** | 0.8228 | 0.8530 | 0.1517 |
| Logistic Regression | 0.7971 | 0.7972 | 0.8476 | 0.8216 | 0.8515 | 0.1519 |
| Ridge Classifier | 0.7869 | 0.7816 | 0.8513 | 0.8149 | 0.8513 | Not available |
| KNN | 0.7828 | 0.7782 | 0.8476 | 0.8114 | 0.8463 | 0.1569 |

Random Forest and LightGBM achieved the highest holdout accuracy of **81.56%**. Random Forest produced the highest holdout ROC-AUC of **0.883**, whereas LightGBM achieved the lowest Brier score of **0.138**.

### Validation stability and statistical comparison

- Random Forest mean nested cross-validation ROC-AUC: **0.882** (95% CI: 0.861 to 0.903).
- Random Forest mean repeated cross-validation ROC-AUC: **0.883** (95% CI: 0.872 to 0.894).
- The Friedman test indicated an overall difference among classifiers: **chi-square = 24.780, p = 0.00083**.
- None of the 28 pairwise Wilcoxon comparisons remained significant after Bonferroni correction.

These findings indicate that numerical differences between the leading ensemble models should not be interpreted as definitive evidence that one model is universally superior.

### Random Forest feature importance

The five highest-ranked variables were:

| Rank | Feature | Importance |
|---:|---|---:|
| 1 | ST-segment slope | 0.3107 |
| 2 | Chest pain type | 0.2786 |
| 3 | Maximum heart rate | 0.0771 |
| 4 | ST depression (`oldpeak`) | 0.0660 |
| 5 | Serum cholesterol | 0.0639 |

Feature importance describes the fitted model's predictive behavior and must not be interpreted as evidence of causal clinical effects.

## Selected outputs

### Leakage-controlled experimental framework

![Leakage-controlled experimental framework](Results/AppendixB.png)

### ROC curves on the independent holdout test set

![ROC curves for all evaluated models](Results/roc_curves_all_models.png)

### Calibration curves

![Calibration curves](Results/calibration_curves.png)

### Random Forest feature importance

![Random Forest feature importance](Results/random_forest_feature_importance.png)

## Reproducibility notes

- The random seed is fixed at `42`.
- The holdout test fraction is `0.40` with stratification by the target.
- Initial records: `2,518`.
- Exact duplicates removed: `1,298`.
- Unique observations retained: `1,220`.
- Final class distribution: 548 negative observations and 672 positive observations.
- The holdout set originates from the same harmonized pool and is not an external clinical validation cohort.
- Precomputed outputs from the submitted manuscript are available in `Results/`.

Differences in operating system, Python version, library version, numerical backend, and parallel execution may produce small variations in fitted parameters or floating-point results.

## Clinical use disclaimer

This repository is intended for research and reproducibility purposes only. The models have not been externally or prospectively validated and are not approved for diagnosis, treatment selection, or direct clinical decision support. Predicted probabilities must not be interpreted as individualized clinical risk estimates.

## Citation

If you use the code, data harmonization procedure, or results from this repository, please cite the accompanying manuscript. Publication details and the DOI will be added after publication.

```bibtex
@article{tutuk2026robust,
  title   = {Robust Heart Disease Risk Prediction Using Machine Learning with Multi-Source Data Integration and Nested Cross-Validation},
  author  = {Tutuk, Mehmet Toygun and Aydin, Omer and Erenay, Fatih Safa and Selim, Aybeyan and Cali, Umit},
  year    = {2026},
  note    = {Manuscript submitted for publication}
}
```

## Authors

- Mehmet Toygun Tutuk
- Omer Aydin
- Fatih Safa Erenay
- Aybeyan Selim
- Umit Cali

## License and data-use conditions

No repository-wide software license is currently included. The source datasets remain subject to their original providers' terms and licenses. Please contact the authors regarding reuse or redistribution beyond research reproducibility and citation.
