import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter
import re

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
REPORT_FIG_DIR = BASE_DIR / "reports" / "figures"

DATA_FILE = DATA_DIR / "memegenerator_processed.csv"

REPORT_FIG_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
    print(f"Loading data from: {DATA_FILE}")
    df = pd.read_csv(DATA_FILE)
    print("Rows:", len(df))
    print("Columns:", df.columns.tolist())
    return df


def plot_caption_length(df: pd.DataFrame):
    # number of words in each caption
    df["word_count"] = df["caption"].astype(str).apply(lambda x: len(x.split()))

    plt.figure()
    df["word_count"].hist(bins=30)
    plt.xlabel("Caption length (words)")
    plt.ylabel("Number of captions")
    plt.title("Distribution of meme caption lengths")
    out_path = REPORT_FIG_DIR / "caption_length_hist.png"
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print("Saved:", out_path)


def plot_top_templates(df: pd.DataFrame, top_n: int = 20):
    counts = df["template_name"].value_counts().head(top_n)

    plt.figure()
    counts.plot(kind="bar")
    plt.xlabel("Meme template")
    plt.ylabel("Number of captions")
    plt.title(f"Top {top_n} meme templates by number of captions")
    plt.xticks(rotation=60, ha="right")
    plt.tight_layout()
    out_path = REPORT_FIG_DIR / "top_templates_bar.png"
    plt.savefig(out_path)
    plt.close()
    print("Saved:", out_path)


def plot_common_words(df: pd.DataFrame, top_n: int = 20):
    # simple word frequency analysis (lowercase, letters only)
    all_text = " ".join(df["caption"].astype(str).tolist()).lower()
    tokens = re.findall(r"[a-z']+", all_text)
    stopwords = {"the", "a", "an", "is", "are", "to", "and", "of", "for", "in", "on", "me", "you"}
    tokens = [t for t in tokens if t not in stopwords and len(t) > 2]

    counter = Counter(tokens)
    most_common = counter.most_common(top_n)
    words = [w for w, _ in most_common]
    freqs = [c for _, c in most_common]

    plt.figure()
    plt.bar(words, freqs)
    plt.xlabel("Word")
    plt.ylabel("Frequency")
    plt.title(f"Top {top_n} most frequent words in captions")
    plt.xticks(rotation=60, ha="right")
    plt.tight_layout()
    out_path = REPORT_FIG_DIR / "common_words_bar.png"
    plt.savefig(out_path)
    plt.close()
    print("Saved:", out_path)


def main():
    df = load_data()
    plot_caption_length(df)
    plot_top_templates(df)
    plot_common_words(df)


if __name__ == "__main__":
    main()
