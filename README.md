# IE 423 2025-2026 Term Project Proposal - Global Food Crisis Early Warning System

## Team Members
- Begüm Acar (122203037)
- İrem Ural (121203037)
- Gamze Kılıç (122203118)
- Sercan Çavuş (122203045)

##Website: https://iremmural.github.io/ie423-2025-2026-termproject-overfitters/

## Project Objective

The objective of this project is to develop a Machine Learning–based early warning system for global food crises by combining food price data from WFP with a baseline food insecurity indicator from FAO. By analyzing historical trends, seasonal patterns, and sudden price fluctuations, the model aims to identify regions at risk of increasing food insecurity and predict major price surges up to three months in advance. This early detection system is designed to support policymakers, humanitarian organizations, and international agencies in making proactive decisions, allocating resources efficiently, and implementing preventive measures before food shortages escalate into large-scale humanitarian crises.

## Datasets

**Dataset:** Global Food Prices (2016-2024)
- Source: https://data.humdata.org/dataset/global-wfp-food-prices

**Dataset:** FAO Food Insecurity Dataset
- Source: https://data360.worldbank.org/en/indicator/FAO_FS_210091

## Repository Structure
```text
├── README.md                  → project overview and setup instructions
├── requirements.txt           → python dependencies list
├── index.html                 → the website — all content, figures, tables, results
├── data/
│   ├── raw/                   → original datasets
│   ├── processed/             → final ML-ready dataset
│   └── README.md              → dataset download links and guide
├── scripts/
│   ├── 01_load_data.py        → verifies file paths and imports data
│   ├── 02_preprocess_data.py  → handles missing values, merges, and creates features
│   ├── 03_basic_eda.py        → creates visualizations and statistical summaries
│   ├── 04_baseline_model.py   →
│   └── 05_shap_analysis.py    →
├── outputs/
│   ├── figures/               → generated charts and graphs
│   └── tables/                → generated tables
└── docs/
    └── ResearchProposalPreprocessing.md   → detailed project proposal file
```
## Installation
```bash
pip install -r requirements.txt
```
## Running the Project
```bash
python scripts/01_load_data.py
python scripts/02_preprocess_data.py
python scripts/03_basic_eda.py
```
## Main Proposal File
See: [docs/ResearchProposalPreprocessing.md](docs/ResearchProposalPreprocessing.md)

