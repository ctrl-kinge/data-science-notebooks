# %% [markdown]
# # Online Retail — Exploratory Data Analysis
#
# **Dataset:** UCI Online Retail — ~540k transactions from a UK online retailer
# (Dec 2010 – Dec 2011).
#
# **Goal:** understand the shape of retail sales data before building the
# `sales-forecaster` project: revenue trends, top products, and what a daily
# sales time series looks like.

# %%
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.append(str(Path.cwd().parent / "scripts"))
from download_data import download

df = pd.read_csv(download(), parse_dates=["InvoiceDate"])
df.head()

# %% [markdown]
# ## Data quality check

# %%
print(df.shape)
print(df.isna().sum())
print(f"Cancelled invoices: {(df['InvoiceNo'].astype(str).str.startswith('C')).sum():,}")
print(f"Non-positive quantity rows: {(df['Quantity'] <= 0).sum():,}")

# %% [markdown]
# Cancellations (invoice numbers starting with "C") and non-positive quantities
# are returns/adjustments — exclude them for sales analysis. Missing
# `CustomerID` is fine here since we analyse at transaction level, not customer
# level.

# %%
sales = df[
    (~df["InvoiceNo"].astype(str).str.startswith("C")) & (df["Quantity"] > 0) & (df["UnitPrice"] > 0)
].copy()
sales["Revenue"] = sales["Quantity"] * sales["UnitPrice"]
print(f"Clean sales rows: {len(sales):,}")

# %% [markdown]
# ## Monthly revenue

# %%
monthly = sales.set_index("InvoiceDate")["Revenue"].resample("ME").sum()
ax = monthly.plot(kind="bar", figsize=(10, 4), title="Monthly revenue")
ax.set_xticklabels([d.strftime("%Y-%m") for d in monthly.index], rotation=45)
ax.set_ylabel("Revenue (GBP)")
plt.tight_layout()

# %% [markdown]
# ## Top 10 products by revenue

# %%
top10 = sales.groupby("Description")["Revenue"].sum().nlargest(10)
top10.sort_values().plot(kind="barh", figsize=(8, 4), title="Top 10 products by revenue")
plt.tight_layout()

# %% [markdown]
# ## Daily revenue — the forecasting target

# %%
daily = sales.set_index("InvoiceDate")["Revenue"].resample("D").sum()
ax = daily.plot(figsize=(12, 4), title="Daily revenue")
ax.set_ylabel("Revenue (GBP)")
plt.tight_layout()

# %%
print(f"Days in series: {len(daily)}")
print(f"Zero-revenue days: {(daily == 0).sum()} (store closed, e.g. Saturdays)")
print(daily.describe().round(0))

# %% [markdown]
# ## Takeaways for sales-forecaster
#
# 1. **Strong trend + seasonality:** revenue climbs sharply toward
#    November–December (holiday shopping) — a forecasting model must capture
#    yearly seasonality and trend, not just recent averages.
# 2. **Weekly pattern with closed days:** zero-revenue days are structural
#    (store closed), not noise — the forecaster should model day-of-week
#    effects or exclude closed days.
# 3. **Data cleaning matters:** ~2% of rows are cancellations/adjustments that
#    would distort revenue if left in. Phase 1 of sales-forecaster needs the
#    same invoice-level cleaning step.
