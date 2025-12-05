import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

# ---------- CONFIG ----------
# Base data directory (ai-meme-creator/data)
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
RAW_FILE = DATA_DIR / "memegenerator.csv"

# Output files
PROCESSED_FILE = DATA_DIR / "memegenerator_processed.csv"
TRAIN_FILE = DATA_DIR / "train_captions.csv"
VAL_FILE = DATA_DIR / "val_captions.csv"
TEST_FILE = DATA_DIR / "test_captions.csv"


def load_raw_data():
    print(f"Loading raw data from: {RAW_FILE}")

    # Try reading as tab-separated UTF-16; if that fails, try normal CSV
    try:
        df = pd.read_csv(RAW_FILE, sep="\t", encoding="utf-16")
    except Exception as e:
        print("Tab/utf-16 read failed, trying default CSV (comma, utf-8). Error:", e)
        df = pd.read_csv(RAW_FILE)

    print("Columns:", df.columns.tolist())
    print("Total rows:", len(df))
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only useful columns and basic cleaning.
    """

    col_map = {
        "Base Meme Name": "template_name",
        "base_meme_name": "template_name",
        "Meme Name": "template_name",
        "meme_name": "template_name",
        "Alternate Text": "caption",
        "alternate_text": "caption",
        "Caption": "caption",
        "caption": "caption",
        "text": "caption",
    }

    # rename any matching columns to standard names
    df = df.rename(columns={c: col_map[c] for c in df.columns if c in col_map})

    # Now we expect at least 'caption'
    if "caption" not in df.columns:
        raise ValueError(
            f"'caption' column not found. Available columns: {df.columns.tolist()}"
        )

    # template_name is optional (we can still work with only captions)
    if "template_name" not in df.columns:
        print("Warning: 'template_name' not found, using only caption text.")
        df["template_name"] = "unknown"

    # keep only template + caption
    df = df[["template_name", "caption"]]

    # drop rows where caption is missing
    df = df.dropna(subset=["caption"])

    # clean caption
    df["caption"] = df["caption"].astype(str).str.strip()

    # remove very short captions
    df = df[df["caption"].str.len() > 5]

    # drop duplicates
    df = df.drop_duplicates(subset=["template_name", "caption"])

    print("After cleaning, rows:", len(df))
    return df


def split_and_save(df: pd.DataFrame):
    # Split into train/val/test
    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

    print("Train:", len(train_df), "Val:", len(val_df), "Test:", len(test_df))

    # Save processed datasets
    df.to_csv(PROCESSED_FILE, index=False)
    train_df.to_csv(TRAIN_FILE, index=False)
    val_df.to_csv(VAL_FILE, index=False)
    test_df.to_csv(TEST_FILE, index=False)

    print("Saved:")
    print("  ", PROCESSED_FILE)
    print("  ", TRAIN_FILE)
    print("  ", VAL_FILE)
    print("  ", TEST_FILE)


if __name__ == "__main__":
    df_raw = load_raw_data()
    df_clean = clean_data(df_raw)
    split_and_save(df_clean)