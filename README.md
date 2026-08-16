# ClimateDT

Reproducible implementation for a multi-source geospatial data engineering
and machine learning framework for one-day-ahead rainfall and temperature
forecasting toward a Climate Digital Twin.

## Overview

The framework integrates CHIRPS v3 precipitation and ERA5-Land temperature
data over Maharashtra, India.

Four forecasting approaches are evaluated:

- Persistence baseline
- Random Forest
- XGBoost
- LSTM

## Study Design

Study period: 1 July 2016 to 30 June 2026

Sampling locations: 23

Raw merged observations: 83,996

The experimental design uses chronological training, validation, and
testing partitions to avoid future-information leakage.

## Repository Structure

- src - data processing, modelling, evaluation and plotting scripts
- data - boundary data and data documentation
- results - selected reproducibility results
- figures - publication figures

## Workflow

1. Merge and validate climate data
2. Engineer spatiotemporal features
3. Evaluate persistence baseline
4. Train and evaluate Random Forest
5. Train and evaluate XGBoost
6. Train and evaluate LSTM
7. Conduct feature ablation analysis
8. Conduct year-wise robustness analysis
9. Generate publication figures
10. Generate Maharashtra study-area map

## Installation

Install dependencies using:

pip install -r requirements.txt

## Data Availability

Large processed climate datasets are not distributed directly through
this repository. Data-access and archival information will be added with
the associated research-data release.

## Reproducibility

Execute the scripts in the src directory according to their numeric prefixes.

## Citation

Citation information will be added after publication.
