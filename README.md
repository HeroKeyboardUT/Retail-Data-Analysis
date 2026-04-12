# 📊 Enterprise Retail Data Analysis & Intelligence

A complete, end-to-end Data Warehouse system explicitly engineered for Online Retail data. This project seamlessly integrates **ETL operations**, a **Star Schema Data Warehouse**, **Machine Learning capabilities**, and a sleek **Professional BI Dashboard**.

## 🚀 Architecture Overview

```text
Data Source (CSV) → ETL Pipeline (Python) → Data Warehouse (Star Schema)
                                                 ↓
                                    AI Models (KMeans, XGBoost, FP-Growth)
                                                 ↓
                                    Dashboard (Streamlit + Plotly)
```

## 🧠 Core Features

1. **Robust ETL Pipeline** — Automatically extracts, aggressively cleans, and transforms raw retail transaction data into actionable dimensions.
2. **Data Warehouse** — Implements a dimensional Data Warehouse (Star Schema: `Dim_Customer`, `Dim_Product`, `Dim_Time`, `Fact_Sales`) directly into PostgreSQL (or SQLite as a resilient fallback).
3. **Embedded Insights (AI/ML)**:
   - **KMeans**: Segments customers based on their RFM profile (Recency, Frequency, Monetary).
   - **XGBoost**: Predicts future Customer Lifetime Value (CLV).
   - **FP-Growth**: Establishes cross-selling recommendation rules based on deep basket analysis.
4. **Enterprise BI Dashboard** — A modern, fully dark-mode Streamlit application utilizing Plotly to vividly demonstrate analytical metrics. Features deep-dive pages into aggregate business health and personalized Customer Intelligence.
5. **Automated LaTeX Reporting** — Dynamically produces 17 analytical figures, `metrics.tex` calculation tables, `summary.json`, and run logs directly structured for academic/business reports.

---

## 📂 Project Structure

```text
Retail-Data-Analysis/
├── etl_pipeline.py          # Main ETL + DW Construction + AI Model Training script
├── streamlit_app.py         # Modern BI Dashboard (Dark Mode UI)
├── docker-compose.yml       # PostgreSQL Docker Configuration
├── requirements.txt         # Production dependencies
├── online_retail.csv        # Source dataset (Not tracked via Git)
├── retail_dw_fallback.db    # SQLite Fallback DB
├── output/
│   └── models/              # Trained Machine Learning artifacts (.pkl)
├── report/
│   ├── main.tex             # Project LaTeX Report
│   └── generated/           # Directory for auto-generated report assets
│       ├── pipeline.log     # Detailed pipeline execution trace
│       ├── summary.json     # Statistics generated out of pipeline execution
│       ├── metrics.tex      # Generated LaTeX metric tables
│       └── figures/         # Auto-generated PNG analytical charts
└── .gitignore               # Ignored system files & datasets
```

---

## 🛠️ Usage & Setup

### 1. Requirements

Make sure you have Python 3.8+ installed.

```powershell
# Create virtual environment (optional but recommended)
python -m venv venv
venv\Scripts\activate

# Install required dependencies
pip install -r requirements.txt
```

### 2. Configure the Database
By default, the application will attempt to connect to **PostgreSQL**. If it isn't deployed or accessible, the engine strictly falls back to **SQLite**.

*(Optional)* Run PostgreSQL locally using Docker:
```powershell
docker compose up -d
```

### 3. Execute the Core Pipeline
You need to run the pipeline to populate tables and train the machine learning algorithms.

```powershell
python etl_pipeline.py
```
**Results of execution:**
- Cleans 500,000+ raw transactional rows.
- Builds Star Schema relationships in the database.
- Generates Machine Learning models (`kmeans.pkl`, `xgb_model.pkl`, `rules.pkl`, `scaler.pkl`) inside `/output/models/`.
- Spawns graphs and metrics directly into the `/report/generated/` folder.

### 4. Launch the Dashboard
Run the analytics dashboard server:

```powershell
streamlit run streamlit_app.py
```
**Key UI Views:**
- **Business Intelligence**: A high-level macro view of overall retail transactions, revenue heatmaps, top products, and overall dataset health.
- **Customer Intelligence**: Detailed, comparative lookups analyzing individual user segments, estimating Lifetime Values, tracking positioning models vs Segment Averages, and giving specific targeted product recommendations.
- **System Logs**: Live-read access to execution tracing.

---

## 📄 Building the Formal Report (LaTeX)

If you wish to compile the final `.pdf` analytical report locally:

```powershell
cd report
pdflatex main.tex
pdflatex main.tex
```
> *(Run exactly twice to ensure page references load correctly. Note: references/bibliography handling is natively omitted unless `references.bib` is reimplemented).*

---

## 🛠️ Tech Stack

- **Data Engineering / ETL**: Python, `pandas`, SQLAlchemy
- **Data Storage**: PostgreSQL 16 `psycopg2` / SQLite3
- **Machine Learning**: `scikit-learn`, `xgboost`, `mlxtend`
- **Frontend / BI Display**: Streamlit, `plotly`, `matplotlib`, `seaborn`
- **Documentation**: LaTeX
