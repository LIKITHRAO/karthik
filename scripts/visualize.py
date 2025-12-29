import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("C://Users//likit//Downloads//ip_as.csv",  encoding="latin-1")

# Frequency plot
df["mentions"].plot(kind="bar", title="Aspect Frequency")
plt.ylabel("Number of Mentions")
plt.tight_layout()
plt.show()

# Sentiment plot
df["avg_sentiment"].plot(kind="bar", title="Average Aspect Sentiment")
plt.ylabel("Sentiment Score")
plt.axhline(0)
plt.tight_layout()
plt.show()
