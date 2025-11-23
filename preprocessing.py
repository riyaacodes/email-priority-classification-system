# preprocessing.py

import re
import pandas as pd


def clean_text(text: str) -> str:
    """
    Very simple text cleaning:
    - lowercasing
    - removing non-alphabetic characters
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_and_prepare_data(csv_path: str) -> pd.DataFrame:
    """
    Load CSV with columns: subject, body, priority
    Combine subject + body and clean the text.
    """
    df = pd.read_csv(csv_path)

    # Make sure required columns exist
    required_cols = {"subject", "body", "priority"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"CSV must contain columns: {required_cols}")

    df["subject"] = df["subject"].fillna("")
    df["body"] = df["body"].fillna("")
    df["text"] = (df["subject"] + " " + df["body"]).apply(clean_text)

    return df
