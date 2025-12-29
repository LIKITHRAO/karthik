import nltk
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from src.aspects import ASPECT_KEYWORDS

nltk.download('punkt')

analyzer = SentimentIntensityAnalyzer()

def get_aspect_sentiment(review_text: str) -> dict:
    aspect_scores = {aspect: [] for aspect in ASPECT_KEYWORDS}

    sentences = nltk.sent_tokenize(review_text)

    for sentence in sentences:
        for aspect, keywords in ASPECT_KEYWORDS.items():
            if any(k in sentence for k in keywords):
                score = analyzer.polarity_scores(sentence)['compound']
                aspect_scores[aspect].append(score)

    # Average sentiment per aspect
    final_scores = {}
    for aspect, scores in aspect_scores.items():
        if scores:
            final_scores[aspect] = sum(scores) / len(scores)

    return final_scores
