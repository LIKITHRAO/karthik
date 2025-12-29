import pandas as pd
from ast import literal_eval

df = pd.read_csv("C://Users//likit//Downloads//processed_reviews.csv",  encoding="latin-1")

# Convert string dict → real dict
df["aspect_sentiment"] = df["aspect_sentiment"].apply(
    lambda x: literal_eval(x) if isinstance(x, str) else {}
)

# Aggregate
aspect_stats = {}

for row in df["aspect_sentiment"]:
    for aspect, score in row.items():
        aspect_stats.setdefault(aspect, []).append(score)

summary = {
    aspect: {
        "mentions": len(scores),
        "avg_sentiment": sum(scores) / len(scores)
    }
    for aspect, scores in aspect_stats.items()
}

summary_df = pd.DataFrame(summary).T.sort_values("mentions", ascending=False)
summary_df.to_csv("C://Users//likit//Downloads//ip_as.csv")

print(summary_df)
