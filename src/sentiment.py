from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

class SentimentModel:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2)
        )
        self.model = LogisticRegression(max_iter=1000)

    def train(self, X, y):
        X_vec = self.vectorizer.fit_transform(X)
        self.model.fit(X_vec, y)

    def predict(self, X):
        X_vec = self.vectorizer.transform(X)
        return self.model.predict(X_vec)

    def evaluate(self, X, y):
        predictions = self.predict(X)
        return classification_report(y, predictions)

def train_test_pipeline(df, text_col, label_col):
    X_train, X_test, y_train, y_test = train_test_split(
        df[text_col],
        df[label_col],
        test_size=0.2,
        stratify=df[label_col],
        random_state=42
    )

    model = SentimentModel()
    model.train(X_train, y_train)

    report = model.evaluate(X_test, y_test)
    return model, report
