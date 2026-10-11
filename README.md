
# MetroPT-3 Predictive Maintenance Analysis

## Overview

This project analyzes real operational sensor data from a metro train's
air compressor / Air Production Unit (APU). It uses exploratory data
analysis, K-Means clustering, and documented failure-window comparisons
to investigate sensor operating patterns associated with reported
air-leak / high-stress events.

**Scope:** This is a retrospective sensor-pattern and failure-window
analysis. It is not a validated system for predicting future failures.

## Dataset

- Source: [UCI MetroPT-3 Dataset](https://archive.ics.uci.edu/dataset/791/metropt3%2Bdataset)
- File: `MetroPT3(AirCompressor).csv`
- Original dataset size: 1,516,948 rows and 17 columns
- Recorded timestamp range: 2020-02-01 to 2020-09-01
- License: CC BY 4.0, as specified by the dataset source

The CSV timestamps are mostly around 10 seconds apart. Therefore, the
recorded CSV should not be described as uniformly sampled at exactly
1 Hz without further qualification.

The dataset is not included in this repository by default. Download it
from the official UCI page and place it at:

`data/metropt+3+dataset/MetroPT3(AirCompressor).csv`

Please retain the dataset attribution and license information when
redistributing any dataset material.

## Sensor Features

The clustering baseline uses seven continuous sensor features:

- `TP2`
- `TP3`
- `H1`
- `DV_pressure`
- `Reservoirs`
- `Oil_temperature`
- `Motor_current`

Features are standardized before K-Means clustering.

## Methodology

1. Load the sensor CSV and parse timestamps.
2. Create provisional binary labels for documented failure windows.
3. Standardize the seven continuous sensor features.
4. Fit K-Means with three clusters.
5. Compare cluster distributions and sensor means inside and outside
   the documented windows.
6. Save metrics, model artifacts, and a cluster-distribution figure.

The cluster count of three is a practical baseline choice, not proof
that three is the uniquely optimal number of operating states.

## Documented Failure Windows

The analysis uses these provisional intervals:

| Start | End | Event description |
|---|---|---|
| 2020-04-18 00:00 | 2020-04-18 23:59 | Air leak / high stress |
| 2020-05-29 23:30 | 2020-05-30 06:00 | Air leak / high stress |
| 2020-06-05 10:00 | 2020-06-07 14:30 | Air leak / high stress |
| 2020-07-15 14:30 | 2020-07-15 19:00 | Air leak / high stress |

The first interval's end time is provisional and should be checked
against the original event documentation before making precise claims.

A label of `1` means an observation falls inside one of these intervals.
A label of `0` means it falls outside them; it does **not** confirm that
the equipment was healthy.

## Observed Results

- Observations inside documented windows: 29,954 (approximately 1.97%).
- Observations outside those windows: 1,486,994 (approximately 98.03%).
- Mean motor current inside the windows: 5.532.
- Mean motor current outside the windows: 1.980.
- Mean oil temperature inside the windows: 75.598.
- Mean oil temperature outside the windows: 62.383.
- 99.42% of observations inside the documented windows were assigned
  to K-Means Cluster 1 in the recorded analysis.

These are descriptive findings, not causal conclusions. The clustering
comparison does not demonstrate advance warning or prospective
prediction. Cluster IDs are arbitrary identifiers, not confirmed
equipment health states.

## Evaluation Limitations

A chronological 80/20 split produced:

- Training period: 2020-02-01 to 2020-07-20 15:58:34
- Testing period: 2020-07-20 15:58:44 to 2020-09-01 03:59:50
- Positive training labels: 29,954
- Positive test labels: 0

Since the later test period contains no positive failure-window labels,
failure recall, precision, and F1 cannot be meaningfully evaluated on
that test period. No supervised classifier is presented as validated.

The current clustering analysis was fitted on the full dataset and is
retrospective. Its cluster-label association is not an independent
predictive evaluation.

## Project Structure

```text
metropt3-predictive-maintenance/
├── data/
│   └── metropt+3+dataset/
├── notebooks/
│   └── exploration.ipynb
├── src/
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── clustering.py
│   ├── feature_engineering.py
│   ├── train.py
│   └── predict.py
├── models/
├── results/
│   ├── figures/
│   └── metrics/
├── requirements.txt
└── README.md
```

## Setup

Use Python 3.13 if it matches your existing working environment.

Activate the project's virtual environment in PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Run the analysis from the project root:

```powershell
python -m src.train
```

The pipeline writes cluster sizes and profiles, failure-window
comparison CSVs, model artifacts, and a figure under `results/`,
`models/`, and `results/figures/`.

To display the current prediction capability status:

```powershell
python -m src.predict
```

This command deliberately does not produce predictions because a
validated failure-prediction classifier has not been established.

## Future Work

A defensible predictive model would require suitable event labels,
positive examples in a chronological evaluation period, leakage-safe
preprocessing, and evaluation on a genuinely later period. Until those
conditions are met, this project focuses on transparent retrospective
analysis.

## Chronological Unseen-Data Clustering Evaluation

To examine how the learned sensor clusters behave on later observations, the
dataset was sorted chronologically and split into an earlier 80% training
period and a later 20% test period.

The StandardScaler and K-Means model were fitted only on the training period.
The fitted scaler transformed the test observations, and the fitted K-Means
model assigned their clusters without refitting.

### Results

| Cluster | Training observations | Training (%) | Test observations | Test (%) |
|---|---:|---:|---:|---:|
| 0 | 688,899 | 56.767 | 159,805 | 52.673 |
| 1 | 457,588 | 37.706 | 142,973 | 47.125 |
| 2 | 67,071 | 5.527 | 612 | 0.202 |

The cluster distribution changed between the two periods. In particular,
Cluster 2 represented 5.527% of training observations but only 0.202% of
test observations. Its sensor profile also differed between the periods.

These findings indicate a change in the distribution of assigned sensor
patterns. They do not establish equipment failure, predictive accuracy, or
the cause of the distribution change.

### Automated Tests

The project includes automated tests for:
- Chronological splitting and cluster assignment.
- Rejection of an invalid training fraction.

Run the tests with:

```bash
python -m unittest discover -s tests -v
```

The current tests use synthetic data and verify basic code behavior. They
do not measure model quality on real failures.

### Important Limitations

- Documented failure windows are provisional event labels, not independently
  verified health labels for every observation.
- The later chronological test period contains no observations within the
  documented failure windows.
- This evaluation tests unseen-data clustering, not failure prediction.
- No validated failure-prediction classifier or failure-detection metrics
  are presented.
