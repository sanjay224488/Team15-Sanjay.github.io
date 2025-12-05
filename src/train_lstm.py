import os
from pathlib import Path

import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.callbacks import ModelCheckpoint
import pickle

# ---------- PATHS ----------
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

TRAIN_FILE = DATA_DIR / "train_captions.csv"
VAL_FILE = DATA_DIR / "val_captions.csv"

TOKENIZER_FILE = MODELS_DIR / "tokenizer.pkl"
MODEL_FILE = MODELS_DIR / "meme_lstm.h5"

os.makedirs(MODELS_DIR, exist_ok=True)


def load_data(max_samples=5000):
    """Load a limited number of samples to keep training light."""
    print(f"Loading train data from: {TRAIN_FILE}")
    train_df = pd.read_csv(TRAIN_FILE)

    print(f"Loading val data from: {VAL_FILE}")
    val_df = pd.read_csv(VAL_FILE)

    train_texts = train_df["caption"].astype(str).tolist()[:max_samples]
    val_texts = val_df["caption"].astype(str).tolist()[: max_samples // 5]

    print("Using train captions:", len(train_texts))
    print("Using val captions:", len(val_texts))
    return train_texts, val_texts


def create_tokenizer(train_texts, num_words=8000):
    """Create a tokenizer with limited vocabulary size."""
    tokenizer = Tokenizer(num_words=num_words, oov_token="<OOV>")
    tokenizer.fit_on_texts(train_texts)
    vocab_size = min(len(tokenizer.word_index) + 1, num_words)
    print("Tokenizer created. Vocabulary size (including OOV):", vocab_size)
    return tokenizer, vocab_size


def create_sequences(texts, tokenizer, max_len=None, max_sequences=20000):
    """
    Create input-output sequences for next-word prediction.
    We DO NOT convert y to one-hot to save memory (use sparse loss instead).
    """
    sequences = []
    for line in texts:
        seq = tokenizer.texts_to_sequences([line])[0]
        if len(seq) < 2:
            continue
        for i in range(1, len(seq)):
            n_gram_seq = seq[: i + 1]
            sequences.append(n_gram_seq)
            if len(sequences) >= max_sequences:
                break
        if len(sequences) >= max_sequences:
            break

    print("Total n-gram sequences (limited):", len(sequences))

    if len(sequences) == 0:
        raise ValueError("No sequences created. Check your data / tokenizer.")

    if max_len is None:
        max_len = max(len(s) for s in sequences)
    print("Max sequence length:", max_len)

    sequences = pad_sequences(sequences, maxlen=max_len, padding="pre")

    X = sequences[:, :-1]
    y = sequences[:, -1]  # integer labels, NOT one-hot

    return X, y, max_len


def build_model(vocab_size, max_len, embedding_dim=64, lstm_units=64):
    model = Sequential()
    model.add(Embedding(vocab_size, embedding_dim, input_length=max_len - 1))
    model.add(LSTM(lstm_units))
    model.add(Dense(vocab_size, activation="softmax"))

    # IMPORTANT: sparse_categorical_crossentropy → no one-hot needed
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"],
    )
    model.summary()
    return model


def train():
    # 1. Load lighter data
    train_texts, val_texts = load_data(max_samples=10000)

    # 2. Tokenizer with limited vocab
    tokenizer, vocab_size = create_tokenizer(train_texts, num_words=20000)

    # 3. Sequences (train & val)
    X_train, y_train, max_len = create_sequences(
        train_texts, tokenizer, max_len=None, max_sequences=20000
    )
    X_val, y_val, _ = create_sequences(
        val_texts, tokenizer, max_len=max_len, max_sequences=5000
    )

    # 4. Build model
    model = build_model(vocab_size, max_len)

    # 5. Always save (not only best)
    checkpoint = ModelCheckpoint(
        filepath=str(MODEL_FILE),
        monitor="val_loss",
        save_best_only=False,
        verbose=1,
    )

    # 6. Train small number of epochs
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=10,
        batch_size=64,
        callbacks=[checkpoint],
    )

    # 7. Save tokenizer & max_len
    with open(TOKENIZER_FILE, "wb") as f:
        pickle.dump({"tokenizer": tokenizer, "max_len": max_len}, f)

    print("Training complete.")
    print("Model saved to:", MODEL_FILE)
    print("Tokenizer saved to:", TOKENIZER_FILE)


if __name__ == "__main__":
    train()
