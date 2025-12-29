import pandas as pd
import matplotlib.pyplot as plt

# =============================
# 1. LOAD DATA
# =============================
df = pd.read_csv(
    "C://Users//likit//Downloads//processed_reviews.csv",
    encoding="latin-1"
)

print("Data loaded:", df.shape)

# =============================
# 2. HANDLE DATE / TIMESTAMP
# =============================
if "timestamp" in df.columns:
    df["date"] = pd.to_datetime(df["timestamp"], errors="coerce")
elif "date" in df.columns:
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
else:
    raise ValueError("No 'date' or 'timestamp' column found")

# Drop rows with invalid dates
df = df.dropna(subset=["date"])

# =============================
# 3. CLEAN LABEL
# =============================
df = df.dropna(subset=["recommended"])
df["recommended"] = df["recommended"].astype(int)

# =============================
# 4. DAILY SENTIMENT TREND
# =============================
daily_sentiment = (
    df
    .groupby(df["date"].dt.date)["recommended"]
    .mean()
)

print("\nDaily sentiment summary:")
print(daily_sentiment.head())

# =============================
# 5. PLOT REVIEW BOMBING TREND
# =============================
plt.figure()
daily_sentiment.plot()
plt.axhline(0.5, linestyle="--")  # Neutral reference line
plt.title("Daily Average Recommendation (Review Bombing Detection)")
plt.xlabel("Date")
plt.ylabel("Average Recommendation Score")
plt.tight_layout()
plt.show()

print("\n✅ Review bombing analysis completed")
