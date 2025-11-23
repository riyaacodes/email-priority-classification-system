# train_model.py

import os
import joblib
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

from preprocessing import load_and_prepare_data


def main():
    # 1. Load and prepare data
    data_path = os.path.join("training_data", "emails_training_set.csv")
    df = load_and_prepare_data(data_path)

    X_text = df["text"].tolist()
    y = df["urgency"].tolist()

    # 2. Load sentence embedding model (MiniLM)
    print("Loading embedding model (all-MiniLM-L6-v2)...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    print("Encoding sentences...")
    X_embeddings = model.encode(X_text, show_progress_bar=True)

    # 3. Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_embeddings, y, test_size=0.33, random_state=42, stratify=y
    )

    # 4. Train classifier
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)

    # 5. Evaluate
    y_pred = clf.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # 6. Save classifier
    os.makedirs("models", exist_ok=True)
    model_path = os.path.join("models", "urgency_classifier.pkl")
    joblib.dump(clf, model_path)
    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    main()
