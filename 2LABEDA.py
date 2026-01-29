# =========================
# LAB 2 - EDA BIVARIATE ANALYSIS (AMAZON UK)
# Dataset columns (yours): uid, asin, title, stars, reviews, price, isBestSeller, boughtInLastMonth, category
# =========================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import chi2_contingency, skew, kurtosis, probplot

# Optional (nice plots). If you don't have them installed, you can comment these out.
import seaborn as sns

pd.set_option("display.max_columns", 200)
pd.set_option("display.width", 120)

# -------------------------
# 1) LOAD DATA (EDIT PATH)
# -------------------------
df = pd.read_csv(r"C:\Users\marcl\Downloads\amz_uk_price_prediction_dataset.csv")

print("Shape:", df.shape)
print("Columns:", df.columns.tolist())
print(df.head(3))

# -------------------------
# 2) BASIC CLEANING
# -------------------------
# price -> numeric
if df["price"].dtype == "object":
    df["price"] = (
        df["price"].astype(str)
        .str.replace("£", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.strip()
    )
df["price"] = pd.to_numeric(df["price"], errors="coerce")

# stars -> numeric
df["stars"] = pd.to_numeric(df["stars"], errors="coerce")

# isBestSeller -> ensure 0/1 or boolean
# If it's already True/False, keep. If it's "True"/"False" strings, convert.
if df["isBestSeller"].dtype == "object":
    df["isBestSeller"] = df["isBestSeller"].astype(str).str.lower().map({"true": True, "false": False})

# Quick missing check
print("\nMissing ratio (key cols):")
print(df[["category", "price", "stars", "isBestSeller"]].isna().mean())

# ==========================================================
# PART 1: Best-Seller Trends Across Categories
# ==========================================================
print("\n====================")
print("PART 1: CATEGORY x isBestSeller")
print("====================")

# 1.1 Crosstab
ct = pd.crosstab(df["category"], df["isBestSeller"], dropna=False)
print("\nCrosstab (head):")
print(ct.head(10))

# Proportion of best sellers per category
# If column True is missing (no bestsellers in some subset), handle gracefully
if True in ct.columns:
    prop_best = (ct[True] / ct.sum(axis=1)).sort_values(ascending=False)
else:
    prop_best = pd.Series(dtype=float)

print("\nTop 15 categories by best-seller proportion:")
print(prop_best.head(15))

# 1.2 Chi-square test
# For chi-square, use only categories with non-null isBestSeller and category
df_chi = df[["category", "isBestSeller"]].dropna()
chi_table = pd.crosstab(df_chi["category"], df_chi["isBestSeller"])
chi2, p, dof, expected = chi2_contingency(chi_table)

print("\nChi-square test results:")
print("Chi2:", chi2)
print("p-value:", p)
print("dof:", dof)

# Cramér's V
n = chi_table.to_numpy().sum()
r, k = chi_table.shape
cramers_v = np.sqrt((chi2 / n) / (min(r - 1, k - 1))) if min(r - 1, k - 1) > 0 else np.nan

print("Cramér's V:", cramers_v)

# 1.3 Visualization: stacked bar chart (top categories by count to keep readable)
top_n = 15
top_categories = df["category"].value_counts().head(top_n).index
ct_top = pd.crosstab(df.loc[df["category"].isin(top_categories), "category"],
                     df.loc[df["category"].isin(top_categories), "isBestSeller"])

ct_top = ct_top.loc[top_categories]  # keep order by popularity

ax = ct_top.plot(kind="bar", stacked=True, figsize=(12,6))
plt.title(f"isBestSeller distribution (stacked) - Top {top_n} categories by count")
plt.xlabel("Category")
plt.ylabel("Number of products")
plt.xticks(rotation=75, ha="right")
plt.tight_layout()
plt.show()

# ==========================================================
# PART 2: Prices and Ratings Across Categories
# ==========================================================
print("\n====================")
print("PART 2: Prices & Stars by Category (OUTLIERS REMOVED)")
print("====================")

# 0) Remove outliers using IQR method
df2 = df.dropna(subset=["price"]).copy()

q1 = df2["price"].quantile(0.25)
q3 = df2["price"].quantile(0.75)
iqr = q3 - q1
lower = q1 - 1.5 * iqr
upper = q3 + 1.5 * iqr

df_no_out = df2[(df2["price"] >= lower) & (df2["price"] <= upper)].copy()

print("Original rows (with price):", len(df2))
print("Rows after IQR outlier removal:", len(df_no_out))
print("Lower bound:", lower, "| Upper bound:", upper)

# 2.1 Violin plot price across categories (top 20 by count)
top20 = df_no_out["category"].value_counts().head(20).index
plt.figure(figsize=(14,6))
sns.violinplot(data=df_no_out[df_no_out["category"].isin(top20)],
               x="category", y="price")
plt.title("Price distribution by Category (Violin) - Top 20 categories (no outliers)")
plt.xlabel("Category")
plt.ylabel("Price")
plt.xticks(rotation=75, ha="right")
plt.tight_layout()
plt.show()

# Highest median price category (NO FILTER)
median_by_cat = df_no_out.groupby("category")["price"].median().sort_values(ascending=False)
print("\nCategory with highest median price (no filter):")
print(median_by_cat.head(10))

# 2.2 Bar chart average price top 10 categories (by count)
top10 = df_no_out["category"].value_counts().head(10).index
mean_price_top10 = df_no_out[df_no_out["category"].isin(top10)].groupby("category")["price"].mean()
mean_price_top10 = mean_price_top10.loc[top10]  # keep order

plt.figure(figsize=(12,5))
mean_price_top10.plot(kind="bar")
plt.title("Average price by Category - Top 10 categories by count (no outliers)")
plt.xlabel("Category")
plt.ylabel("Average price")
plt.xticks(rotation=75, ha="right")
plt.tight_layout()
plt.show()

# Highest average price category (NO FILTER)
mean_by_cat = df_no_out.groupby("category")["price"].mean().sort_values(ascending=False)
print("\nCategory with highest average price (no filter):")
print(mean_by_cat.head(10))

# 2.3 Box plots stars by category (top 10 by count)
df_stars = df_no_out.dropna(subset=["stars"]).copy()
top10_star = df_stars["category"].value_counts().head(10).index

plt.figure(figsize=(14,6))
sns.boxplot(data=df_stars[df_stars["category"].isin(top10_star)],
            x="category", y="stars")
plt.title("Stars distribution by Category (Boxplot) - Top 10 categories (no outliers)")
plt.xlabel("Category")
plt.ylabel("Stars")
plt.xticks(rotation=75, ha="right")
plt.tight_layout()
plt.show()

# Category with highest median stars (NO FILTER)
median_stars_by_cat = df_stars.groupby("category")["stars"].median().sort_values(ascending=False)
print("\nCategory with highest median stars (no filter):")
print(median_stars_by_cat.head(10))

# ==========================================================
# PART 3: Price vs Stars
# ==========================================================
print("\n====================")
print("PART 3: Price vs Stars")
print("====================")

df_corr = df_no_out.dropna(subset=["price", "stars"]).copy()

# 3.1 Correlation coefficient
corr = df_corr["price"].corr(df_corr["stars"])
print("Correlation (price vs stars):", corr)

# 3.2 Scatter plot
# sample to avoid plotting millions of points
sample_n = 50000
df_sample = df_corr.sample(n=min(sample_n, len(df_corr)), random_state=42)

plt.figure(figsize=(8,6))
plt.scatter(df_sample["stars"], df_sample["price"], s=5, alpha=0.2)
plt.title("Scatter: Stars vs Price (sample, no outliers)")
plt.xlabel("Stars")
plt.ylabel("Price")
plt.tight_layout()
plt.show()

# 3.3 Correlation heatmap (numerical columns)
num_cols = ["stars", "reviews", "price", "boughtInLastMonth", "isBestSeller"]
# Convert bool to int for correlation
df_heat = df_no_out.copy()
df_heat["isBestSeller"] = df_heat["isBestSeller"].astype("float")
df_heat = df_heat[num_cols].dropna()

corr_mat = df_heat.corr(numeric_only=True)

plt.figure(figsize=(7,5))
sns.heatmap(corr_mat, annot=True, fmt=".2f")
plt.title("Correlation heatmap (numerical variables, no outliers)")
plt.tight_layout()
plt.show()

# 3.4 QQ plot for price normality (use sample)
price_sample = df_no_out["price"].dropna().sample(n=min(50000, df_no_out["price"].dropna().shape[0]), random_state=42)

plt.figure(figsize=(6,6))
probplot(price_sample, dist="norm", plot=plt)
plt.title("QQ plot: Price vs Normal distribution (sample, no outliers)")
plt.tight_layout()
plt.show()

# ==========================================================
# BONUS: Repeat correlation + plots WITHOUT removing outliers
# ==========================================================
print("\n====================")
print("BONUS: Without outlier removal")
print("====================")

df_bonus = df.dropna(subset=["price", "stars"]).copy()
corr_bonus = df_bonus["price"].corr(df_bonus["stars"])
print("Correlation (price vs stars) WITHOUT outlier removal:", corr_bonus)

df_bonus_sample = df_bonus.sample(n=min(sample_n, len(df_bonus)), random_state=42)

plt.figure(figsize=(8,6))
plt.scatter(df_bonus_sample["stars"], df_bonus_sample["price"], s=5, alpha=0.2)
plt.title("Scatter: Stars vs Price (sample, WITH outliers)")
plt.xlabel("Stars")
plt.ylabel("Price")
plt.tight_layout()
plt.show()

# QQ plot with outliers (sample)
price_sample2 = df["price"].dropna().sample(n=min(50000, df["price"].dropna().shape[0]), random_state=42)

plt.figure(figsize=(6,6))
probplot(price_sample2, dist="norm", plot=plt)
plt.title("QQ plot: Price vs Normal distribution (sample, WITH outliers)")
plt.tight_layout()
plt.show()
