# IE 423 Term Project Proposal — Global Food Crisis Early Warning System


## Team Information


- Begüm Acar
- İrem Ural
- Gamze Kılıç
- Sercan Çavuş

---

## Dataset Description


We use two primary datasets for this project:

1. **Global Food Prices (2016-2024)**
   - **Source:** Humanitarian Data Exchange (WFP) - [Link](https://data.humdata.org/dataset/global-wfp-food-prices)
   - **Description:** This datasets contains historical monthly food price data for various commodities across multiple countries and markets. It includes variables such as country, market, commodity type, price, unit, currency, and date, enabling the analysis of food price trends and sudden price fluctuations over time.
2. **FAO Food Insecurity Dataset**
   - **Source:** World Bank Data360 - [Link](https://data360.worldbank.org/en/indicator/FAO_FS_210091)
   - **Description:** This dataset provides country-level measurements of food insecurity, offering macro-level insights into population vulnerability. It tracks the prevalence of moderate or severe food insecurity, which estimates the percentage of people living in households classified as moderately or severely food insecure.

We selected these datasets because combining micro-level economic shocks (monthly food prices) with macro-level vulnerability metrics (FAO food insecurity scores) allows us to explore meaningful patterns that precede humanitarian emergencies. The WFP dataset initially contains millions of rows across 9 years, which we aggregate into a robust panel dataset of monthly country-product observations.

---

## Dataset Access


The raw datasets are stored in:

`data/raw/`
- `wfp_food_prices_global_2016.csv` through `2024.csv`
- `FAO_FS_210091.csv`

Raw datasets can be downloaded from:
- [WFP Global Food Prices](https://data.humdata.org/dataset/global-wfp-food-prices)
- [FAO Food Insecurity Dataset](https://data360.worldbank.org/en/indicator/FAO_FS_210091)

---


## Research Questions


### Research Question 1
Which countries and food products experience the highest price volatility, and are these patterns seasonal?

**Explanation:** Before building a forecasting model, it is crucial to understand fundamental market behaviors in different regions. While some volatility is natural and stems from recurring agricultural cycles (e.g., harvest seasons), anomalous volatility may indicate systemic supply chain problems. Using our engineered features such as rounded standard deviations and volatility rates, along with the monthly variable, this question aims to reveal whether specific staple food commodities (such as grains and fresh produce) or specific countries are inherently more unstable. This exploratory analysis will help us differentiate between expected seasonal price volatility and actual market anomalies.

---

### Research Question 2
Using past price movements and a country's current food insecurity score, can a food price crisis in a country be predicted 3 months in advance?

**Explanation:** This is a key forecasting modeling objective of our early warning system. Past price movements capture the economic momentum of a product, while the FAO food insecurity score provides the context of macro-level vulnerability in the region. By combining these micro and macro indicators, we aim to train classification models to predict the crisis label we define (a price increase of 20% or more within the next 3 months). Answering this question will provide historical data with an actionable head start for humanitarian interventions before a crisis fully materializes.

---

### Research Question 3
What are the most effective factors in determining crisis risk: product type, country vulnerability, or seasonality?

**Explanation:** A high-performance model is only partially useful if it acts like a black box; interpretability is crucial for policymakers. By analyzing the feature importance outputs of our machine learning models, this question aims to identify the underlying drivers of food crises. We want to explore whether a country's underlying vulnerability (FAO score) outweighs market momentum (price lags/fluctuations). Understanding which variables carry the greatest weight will allow organizations to allocate monitoring resources more effectively.

---


## Project Proposal


This project aims to develop a Machine Learning–based early warning system for global food crises by investigating the relationship between historical food price fluctuations and regional food insecurity prevalence. By understanding these patterns, we hope to anticipate potential crises and provide actionable insights for policymakers and humanitarian organizations. 

First, we will clean and preprocess the raw datasets by handling missing values, filtering out non-food items, and standardizing geographic codes (countryiso3). To ensure high-quality time-series analysis, a filtering mechanism will be implemented to exclude countries with fragmented timelines or insufficient historical data. We will aggregate the high-frequency price data into monthly summaries and engineer temporal features, including lag prices, rolling means, and volatility ratios. We will define our target variable (`crisis_label`) mathematically as a 20% or greater surge in a product's average price within a 3-month future window. The 20% threshold is adopted in alignment with WFP’s Alert Price Spikes (ALPS) methodology, which identifies such rapid escalations as 'Market Shocks' that fundamentally disrupt household purchasing power and food access.

Then, we will conduct an exploratory data analysis (EDA) to understand the distribution of our crisis labels, price volatility across different years, and the correlation between FAO insecurity scores and price shocks. 

Based on our research questions, we will apply descriptive statistics and machine learning classification methods to predict the `crisis_label`. Our goal is not only to build an accurate predictive model but also to generate interpretable findings that indicate *which* conditions most frequently lead to crises. 

Possible challenges include handling data sparsity (missing FAO scores for certain country-year pairs), managing temporal data leakage during cross-validation, and isolating true food crises from general national hyperinflation.

---

## Preprocessing Steps


- **Step 1 — Loading the Data**  
  Using `scripts/01_load_data.py`, we loaded 9 yearly WFP food price files and 1 FAO food insecurity file with `pandas`.  
  The WFP files were combined into a single dataset, while the FAO file was loaded separately for later merging.

---

- **Step 2 — Initial Inspection**  
  We examined the dataset structure by checking shapes, column names, and category distributions.  
  This step helped us identify irrelevant entries such as `"non-food"` observations and detect missing values in key variables like `countryiso3`, `date`, `product`, and `price`.

---

- **Step 3 — Cleaning the WFP Data**  
  In `scripts/02_preprocess_data.py`, we selected the relevant WFP columns, renamed them into a consistent format, converted `date` to datetime and `price` to numeric, and removed rows with missing critical values.  
  We also excluded the `"non-food"` category because it was not relevant to our food crisis detection objective.

---

- **Step 4 — Country-Level Filtering**  
  Continuing in `scripts/02_preprocess_data.py` to improve time-series consistency, we removed countries with fewer than 4 distinct years of WFP data.  
  We also excluded countries with fragmented year coverage, since unstable temporal patterns would weaken lag and rolling-window calculations.

---

- **Step 5 — Monthly Aggregation**  
  The WFP data was aggregated to the monthly country-product level.  
  For each country-product-month combination, we created summary variables including `avg_price`, `median_price`, `max_price`, `price_std`, and `obs_count`.

---

- **Step 6 — Time Alignment and Feature Engineering**  
  After aggregation, we aligned each country-product series to monthly frequency using `asfreq("MS")`.  
  Then we generated lag features (`price_lag_1`, `price_lag_3`, `price_lag_6`), rolling statistics (`rolling_mean_3`, `rolling_mean_6`, `rolling_std_3`, `rolling_std_6`), percentage change (`pct_change_1m`), and volatility ratios (`volatility_ratio_3`, `volatility_ratio_6`).  
  We also clipped extreme values in `pct_change_1m` to reduce the effect of outliers.

---

- **Step 7 — Target Variable Creation**  
  We created `future_price_3m` by shifting the average price 3 months ahead within each country-product group.  
  Based on this value, we defined `crisis_label` as `1` if the future price increase was at least 20%, and `0` otherwise.  
  Rows with unresolved future values were removed, and the temporary future-price column was dropped to prevent leakage.

---

- **Step 8 — FAO Preparation and Merge**  
  On the FAO side, we kept only observations corresponding to the total population, selected the relevant columns, and renamed them into a consistent structure.  
  If multiple records existed for the same country-year pair, we used the median `food_insecurity_score`.  
  The FAO dataset was then merged with the WFP monthly dataset using `countryiso3` and `year`.

---

- **Step 9 — Final Filtering and Saving**  
  After merging in `scripts/02_preprocess_data.py`, we removed countries with insufficient FAO coverage, including countries with fewer than 4 matched FAO years and countries with no FAO score at all.  
  Finally, we dropped rows with missing feature values and saved the processed dataset as:

  `data/processed/processed_food_crisis_data.csv`

---

## Initial Outputs


### Dataset Shape
- **Raw WFP Shape:** Contains approximately 2 million observations spanning from 2016 to 2024
- **Cleaned & Engineered Shape:** After monthly aggregation,dynamic filtering, NA dropping and merging, the final `model_df` contains 67,564 observations across 62 countries.

---

### Missing Value Summary
- Future dates where the 3-month forward price could not be observed were intentionally dropped to avoid labeling uncertainty.
- Some country-year pairs lacked FAO food insecurity scores. Observations missing essential engineered features or FAO scores were dropped to ensure a clean modeling dataset.

---

### Data Distributions
- **Target Variable (`crisis_label`):** The data represents an imbalanced classification problem, as a 20% price surge within 3 months represents a true anomaly rather than a daily occurrence. The exact proportion of `1`s vs `0`s is dynamically outputted by the preprocessing script.

---

## Visualizations
All outputs below are generated by `scripts/03_basic_eda.py`.

---

### Figure 1 - Global Food Insecurity Rankings Barchart
>Generated by: scripts/03_basic_eda.py → saved to `outputs/figures/Global_Food_Insecurity_Rankings_Barchart.png`

![Global Food Insecurity Rankings Barchart](https://github.com/BILGI-IE-423/ie423-2025-2026-termproject-overfitters/blob/main/outputs/figures/01_global_food_insecurity_rankings.png)

This graph compares the average food insecurity indices of the countries in the dataset, clearly revealing that countries like Afghanistan and Ethiopia are in the highest risk group. The visualization provides a hierarchy for identifying which regions are experiencing more chronic and severe food crises, offering a key indicator for your early warning system.

---

### Figure 2 - Global Food Crisis Map Bubblechart
>Generated by: scripts/03_basic_eda.py → saved to `outputs/figures/Global_Food_Crisis_Map_Bubblechart.png`

![Global Food Crisis Map Bubblechart](https://github.com/BILGI-IE-423/ie423-2025-2026-termproject-overfitters/blob/main/outputs/figures/02_global_product_crisis_map.png)

The global food crisis map highlights at-risk hotspots by showing the geographical intensity and severity of crisis events on a coordinate-based system. The balloon sizes represent the degree of impact of the crisis, providing spatial insight into which regions should be prioritized by early warning systems.

---

### Figure 3 - The Relationship Between Price Changes and Crisis Boxplot
>Generated by: scripts/03_basic_eda.py → saved to `outputs/figures/The_Relationship_Between_Price_Changes_and_Crisis_Boxplot.png`

![The Relationship Between Price Changes and Crisis Boxplot](https://github.com/BILGI-IE-423/ie423-2025-2026-termproject-overfitters/blob/main/outputs/figures/03_price_change_vs_crisis.png)

A boxplot examines the relationship between percentage changes in food prices and crisis situations, demonstrating that price fluctuations and outliers are much more pronounced during periods of crisis. This analysis proves that price shocks are one of the most critical leading indicators for an early warning system.

---

### Figure 4 - Correlation Matrix of Food Crisis Indicators Heatmap
>Generated by: scripts/03_basic_eda.py → saved to `outputs/figures/Correlation_ Matrix_of_ Food_Crisis _Indicators_Heatmap.png`

![Correlation Matrix of Food Crisis Indicators Heatmap](https://github.com/BILGI-IE-423/ie423-2025-2026-termproject-overfitters/blob/main/outputs/figures/04_correlation_heatmap.png)

The correlation matrix analyzes the linear relationships between food crisis indicators, identifying which variables have the strongest correlation with the crisis situation. This visualization provides a scientific basis for selecting the most effective features to be used in the prediction model.

---

### Figure 5 - Annual Crisis Intensity Heatmap
>Generated by: scripts/03_basic_eda.py → saved to `outputs/figures/Annual_crisis_intensity_Heatmap.png`

![Annual Crisis Intensity Heatmap](https://github.com/BILGI-IE-423/ie423-2025-2026-termproject-overfitters/blob/main/outputs/figures/06_annual_crisis_intensity.png)

This heatmap shows the distribution of food crisis intensity by year and country, visualizing the continuity of crises over time and global increase trends in specific periods. Color intensity clearly reveals temporal changes in crisis levels and chronic risk zones.

---

### Figure 6 - Distribution of Food Crisis Labels Barchart
>Generated by: scripts/03_basic_eda.py → saved to `outputs/figures/Distribution _of_Food_Crisis_Labels_Barchart.png`

![Distribution of Food Crisis Labels Barchart](https://github.com/BILGI-IE-423/ie423-2025-2026-termproject-overfitters/blob/main/outputs/figures/05_crisis_label_distribution.png)

This graph reveals class imbalance by showing the numerical distribution of "crisis present" and "no crisis" states in the dataset. This data guides the sampling strategies to be applied in order to obtain more accurate results during the training phase of the model.

---


## How to Run the Project


### 1. Clone the repository
```bash
git clone [https://github.com/BILGI-IE-423/ie423-2025-2026-termproject-overfitters/tree/main]
cd [ie423-2025-2026-termproject-overfitters]
```

---

### 2. Install required packages
```markdown
pip install -r requirements.txt
```

---

### 3. Place the dataset
Download the datasets from the links below:

https://data.humdata.org/dataset/global-wfp-food-prices (download 2016 through 2024)

https://data360.worldbank.org/en/indicator/FAO_FS_210091

After downloading, place all dataset files inside the following directory:
```bash
data/raw/
```
> **Important:** The project assumes that all raw dataset files are located in the `data/raw/` directory with their original filenames.
> If the files are missing or placed incorrectly, the scripts will not run.
> For detailed setup instructions, see `data/README.md`. The folder structure should look like:
```markdown
data/
├── raw/
    ├── FAO_FS_210091.csv
    ├── wfp_food_prices_global_2016.csv
    ├── wfp_food_prices_global_2017.csv
    ├── wfp_food_prices_global_2018.csv
    ├── wfp_food_prices_global_2019.csv
    ├── wfp_food_prices_global_2020.csv
    ├── wfp_food_prices_global_2021.csv
    ├── wfp_food_prices_global_2022.csv
    ├── wfp_food_prices_global_2023.csv
    └── wfp_food_prices_global_2024.csv
```

---

### 4. Run the scripts
Run the scripts in the following order:

```bash
python scripts/01_load_data.py
python scripts/02_preprocess_data.py
python scripts/03_basic_eda.py
```

---

### 5. Outputs
After running the scripts:

* Cleaned dataset will be saved in: `data/processed/processed_food_crisis_data.csv`
* Figures will be saved in: `outputs/figures/`
* Tables will be saved in: `outputs/tables/`

---


## Transparency and Traceability

All outputs presented in this document are generated directly from the Python scripts located in the scripts/ folder.


* Data loading and initial inspection outputs are generated by: `scripts/01_load_data.py`
* Data preprocessing and feature engineering are performed by: `scripts/02_preprocess_data.py`
* All visualizations and analysis results are generated by: `scripts/03_basic_eda.py`

---

### Data Pipeline

The project follows a structured pipeline:

* Raw data is stored in: `data/raw/`

* Processed data is generated and saved in: `data/processed/processed_food_crisis_data.csv`

* Visual outputs (plots) are saved in: `outputs/figures/`

* Tables are stored in: `outputs/tables/`

---

### Reproducibility Guarantee
* All results are reproducible by running the provided scripts.
* No manual modifications are applied to the outputs.
* Every figure and dataset can be traced back to its corresponding script.
