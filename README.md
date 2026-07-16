# data-science-notebooks

Weekly data analysis notebooks — pandas, visualization, and ML experiments.
Part of a daily-practice system; one narrated analysis lands here every week.

## Notebooks

| # | Notebook | Dataset | Topics |
|---|----------|---------|--------|
| 01 | [Online Retail EDA](notebooks/01_online_retail_eda.ipynb) | UCI Online Retail | data cleaning, revenue trends, time series |
| 02 | [Seasonality Deep-Dive](notebooks/02_seasonality_deep_dive.ipynb) | UCI Online Retail | day-of-week effects, autocorrelation, decomposition |

## Setup

```powershell
py -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
```

Notebooks are authored as [jupytext](https://jupytext.readthedocs.io/) percent
scripts (`.py`) and committed alongside the executed `.ipynb` so charts render
on GitHub. Data downloads on first run via `scripts/download_data.py`.
