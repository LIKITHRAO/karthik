import sys
import os
import pandas as pd

# Allow imports from src/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.preprocess import clean_text
from src.aspects import extract_aspects
from src.sentiment import train_test_pipeline
from src.aspect_sentiment import get_aspect_sentiment


# =============================
# 1. LOAD DATA
# =============================
df = pd.read_csv(
    "C://Users//likit//Downloads//ip.csv",
    encoding="latin-1"
    
)

print("Raw dataset shape:", df.shape)
print("Columns found:", df.columns.tolist())



# =============================
# 3. SKIP BLANK / INVALID ROWS
# =============================
df = df.dropna(subset=["review_text", "recommended"])
df = df[df["review_text"].astype(str).str.strip() != ""]
df["recommended"] = df["recommended"].astype(int)

print("Clean dataset shape:", df.shape)
print("Label distribution:")
print(df["recommended"].value_counts())

# =============================
# 4. TEXT PREPROCESSING
# =============================
print("\nCleaning review text...")
df["clean_review"] = df["review_text"].apply(clean_text)

# =============================
# 5. ASPECT EXTRACTION
# =============================
print("Extracting aspects...")
df["aspects"] = df["clean_review"].apply(extract_aspects)

# =============================
# 6. SENTIMENT CLASSIFICATION
# =============================
print("\nTraining sentiment model...")
model, report = train_test_pipeline(
    df,
    text_col="clean_review",
    label_col="recommended"
)

print("\nSentiment Model Evaluation:\n")
print(report)

# =============================
# 7. ASPECT-BASED SENTIMENT
# =============================
print("Calculating aspect-based sentiment...")
df["aspect_sentiment"] = df["review_text"].apply(get_aspect_sentiment)

# =============================
# 8. SAVE PROCESSED DATA
# =============================
output_path = "C://Users//likit//Downloads//processed_reviews.csv"
df.to_csv(output_path, index=False)

print("\n✅ PIPELINE COMPLETED SUCCESSFULLY")
print("Processed file saved at:", output_path)
