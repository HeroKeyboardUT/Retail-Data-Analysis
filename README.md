# DW DSS — Retail Data Warehouse + BI + AI Pipeline

Complete end-to-end Data Warehouse system for Online Retail data analysis.

## Architecture

```
Data Source (CSV) → ETL Pipeline (Python) → Data Warehouse (Star Schema)
                                                 ↓
                                    AI Models (KMeans, XGBoost, FP-Growth)
                                                 ↓
                                    Dashboard (Streamlit + Plotly)
```

## What It Does

1. **ETL Pipeline** — Extracts, cleans, transforms retail transaction data
2. **Data Warehouse** — Loads star schema (Dim_Customer, Dim_Product, Dim_Time, Fact_Sales) into PostgreSQL/SQLite
3. **AI Models** — Trains KMeans (customer segmentation), XGBoost (CLV prediction), FP-Growth (product recommendations)
4. **Dashboard** — Interactive BI dashboard + AI customer intelligence in Streamlit
5. **Report** — Auto-generates 17 figures, metrics.tex, summary.json, and pipeline.log for the LaTeX report

## Project Structure

```
DW DSS/
├── etl_pipeline.py          # Main ETL + DW + AI pipeline
├── streamlit_app.py          # BI Dashboard + AI serving layer
├── docker-compose.yml        # PostgreSQL container
├── requirements.txt          # Python dependencies
├── online_retail.csv         # Source dataset (~46 MB)
├── retail_dw_fallback.db     # SQLite fallback database
├── output/
│   └── models/               # Trained model artifacts (.pkl)
├── report/
│   ├── main.tex              # LaTeX report
│   ├── references.bib        # Bibliography (13 references)
│   ├── hcmut.jpg             # University logo
│   └── generated/
│       ├── pipeline.log      # Detailed pipeline execution log
│       ├── summary.json      # All metrics as JSON
│       ├── metrics.tex       # Auto-generated LaTeX tables
│       └── figures/          # 17 auto-generated PNG charts
└── *.pkl                     # Model artifacts (root, for backward compat)
```

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. (Optional) Start PostgreSQL via Docker:

```powershell
docker compose up -d
```

If PostgreSQL is unavailable, the pipeline **automatically falls back to SQLite**.

## Run The Pipeline

```powershell
python etl_pipeline.py --output-dir output --report-dir report\generated
```

This will:
- Clean 541,909 raw rows → 392,692 clean transactions
- Load star schema into the database
- Train KMeans (k=4), XGBoost (CLV), FP-Growth (214 rules)
- Generate 17 figures for the report
- Save metrics.tex, summary.json, and pipeline.log

## Run Streamlit Dashboard

```powershell
streamlit run streamlit_app.py
```

Features:
- **Dashboard tab**: KPIs, monthly revenue, top countries, top products, RFM distributions, cluster visualization
- **AI Intelligence tab**: Customer lookup with cluster, CLV prediction, and cross-sell recommendations
- **Pipeline Log tab**: View the full ETL execution log

## Build The LaTeX Report

```powershell
cd report
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

## Key Results

| Metric | Value |
|--------|-------|
| Clean rows | 392,692 |
| Unique customers | 4,338 |
| KMeans Silhouette | 0.6162 |
| XGBoost R² | 0.3832 |
| XGBoost RMSE | 4,753.40 |
| FP-Growth rules | 214 |
| Figures generated | 17 |

## Technologies

- **ETL**: Python, pandas, SQLAlchemy
- **Database**: PostgreSQL 16 / SQLite (fallback)
- **ML/AI**: scikit-learn, XGBoost, mlxtend
- **Visualization**: matplotlib, seaborn, Plotly
- **Dashboard**: Streamlit
- **Report**: LaTeX (with auto-generated content)
