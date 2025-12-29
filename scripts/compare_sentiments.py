import pandas as pd
from ast import literal_eval

df = pd.read_csv("C://Users//likit//Downloads//processed_reviews.csv",  encoding="latin-1")

df["aspect_sentiment"] = df["aspect_sentiment"].apply(
    lambda x: literal_eval(x) if isinstance(x, str) else {}
)

def overall_aspect_score(d):
    return sum(d.values()) / len(d) if d else 0

df["aspect_score"] = df["aspect_sentiment"].apply(overall_aspect_score)

comparison = pd.crosstab(
    df["recommended"],
    df["aspect_score"].apply(lambda x: "Positive" if x > 0 else "Negative")
)

print(comparison)
