import os
from pathlib import Path
import pickle
import textwrap
import random  # for professional templates

import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from PIL import Image, ImageDraw, ImageFont

# ---------- PATHS ----------
BASE_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = BASE_DIR / "models"
TEMPLATES_DIR = BASE_DIR / "templates"
GENERATED_DIR = BASE_DIR / "generated"

MODEL_FILE = MODELS_DIR / "meme_lstm.h5"
TOKENIZER_FILE = MODELS_DIR / "tokenizer.pkl"

os.makedirs(GENERATED_DIR, exist_ok=True)


# ---------- LOAD MODEL & TOKENIZER ----------
@st.cache_resource
def load_lstm_model():
    return load_model(MODEL_FILE)


@st.cache_resource
def load_tokenizer():
    with open(TOKENIZER_FILE, "rb") as f:
        data = pickle.load(f)
    tokenizer = data["tokenizer"]
    max_len = data["max_len"]
    return tokenizer, max_len


model = load_lstm_model()
tokenizer, max_len = load_tokenizer()


# ---------- CAPTION GENERATION (LSTM RAW) ----------
def lstm_extend(seed_text: str, num_words: int = 8) -> str:
    """
    Low-level function: ask LSTM to extend the seed_text.
    We will still post-process this output later.
    """
    seed_text = seed_text.strip()
    if not seed_text:
        return ""

    previous_word = None
    used_words = set(seed_text.lower().split())
    text = seed_text

    for _ in range(num_words):
        seq = tokenizer.texts_to_sequences([text])[0]
        if len(seq) == 0:
            break

        seq = pad_sequences([seq], maxlen=max_len - 1, padding="pre")
        preds = model.predict(seq, verbose=0)[0]
        next_index = preds.argmax()

        if next_index == 0:
            break

        next_word = tokenizer.index_word.get(next_index, None)
        if next_word is None:
            break

        if next_word == "<OOV>":
            break
        if next_word == previous_word:
            break
        if next_word.lower() in used_words:
            break

        text += " " + next_word
        previous_word = next_word
        used_words.add(next_word.lower())

    return text


# ---------- PROFESSIONAL TEMPLATE CAPTIONS ----------
def template_caption(topic: str) -> str:
    """Generate a clean, English meme-style caption using templates."""
    topic = topic.strip()
    if not topic:
        return ""

    templates = [
        f"When {topic} hits different 😅",
        f"Me vs {topic} – who wins? 🤔",
        f"Facing {topic} like a boss 👑",
        f"{topic} but I still survive 💀",
        f"{topic} be like… why me? 😭",
        f"POV: you’re dealing with {topic} again",
        f"{topic} level: legendary ⚡",
        f"Still standing after {topic} 💪",
    ]
    return random.choice(templates)


# ---------- HIGH-LEVEL AI CAPTION FUNCTION ----------
def generate_caption(seed_text: str, num_words: int = 8) -> str:
    """
    High-level AI caption generator:
    1. Try LSTM to extend the text.
    2. If LSTM output is not good (too short / weird / non-English),
       fall back to a professional English template.
    """
    seed_text = seed_text.strip()
    if not seed_text:
        return ""

    raw = lstm_extend(seed_text, num_words=num_words).strip()
    if not raw:
        return template_caption(seed_text)

    # Check if everything looks English-ish
    def looks_english(s: str) -> bool:
        # only ASCII characters
        if not all(ord(ch) < 128 for ch in s):
            return False
        return True

    # Split for analysis
    seed_words = seed_text.split()
    raw_words = raw.split()

    # Case 1: too short or same as input
    if len(raw_words) <= len(seed_words) or raw.lower() == seed_text.lower():
        return template_caption(seed_text)

    # Only consider extra words added by LSTM
    extra_words = raw_words[len(seed_words):]

    # filter out filler/bad small words
    banned_non_english = {
        "la", "mejor", "de", "el", "ella", "ellos", "ellas",
        "que", "por", "para", "con", "sin", "una", "uno",
        "unas", "unos", "al", "del"
    }
    filler_small = {"the", "a", "an", "of", "to", "in", "on"}

    good_extra = []
    for w in extra_words:
        wl = w.lower()
        if not w.isascii() or not w.isalpha():
            continue
        if wl in banned_non_english:
            continue
        if wl in filler_small:
            continue
        if len(wl) < 3:
            continue
        good_extra.append(w)

    # Case 2: LSTM added nothing useful or non-English → template
    if not good_extra or not looks_english(" ".join(raw_words)):
        return template_caption(seed_text)

    # Otherwise: seed + good extra words
    final_words = seed_words + good_extra
    return " ".join(final_words)


# ---------- TEMPLATE HANDLING ----------
def get_template_files():
    if not TEMPLATES_DIR.exists():
        return []
    return [
        f for f in TEMPLATES_DIR.iterdir()
        if f.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ]


# ---------- MEME IMAGE CREATION ----------
def create_meme_image(template_path: Path, caption: str) -> Path:
    """
    Draw the caption on the template image and save result into GENERATED_DIR.
    Ensures all text stays inside the image.
    """
    img = Image.open(template_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    width, height = img.size

    caption = caption.strip()
    if not caption:
        caption = "No caption entered 🤔"

    base_font_size = max(int(height / 18), 20)

    def layout_text(font_size):
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except Exception:
            font = ImageFont.load_default()

        approx_char_width = font_size * 0.55
        max_chars = max(8, int(width / approx_char_width))

        wrapped = textwrap.fill(caption, width=max_chars)
        lines = wrapped.split("\n")

        line_height = font_size + 4
        text_block_height = line_height * len(lines)
        return font, lines, line_height, text_block_height

    font_size = base_font_size
    while True:
        font, lines, line_height, text_block_height = layout_text(font_size)
        if text_block_height < height * 0.4 or font_size <= 14:
            break
        font_size -= 2

    y_start = height - text_block_height - 20
    if y_start < 10:
        y_start = 10

    outline_range = 2
    for i, line in enumerate(lines):
        if not line:
            continue

        try:
            w, h = font.getsize(line)
        except Exception:
            w, h = (len(line) * font_size * 0.6, line_height)

        x = (width - w) / 2
        y = y_start + i * line_height

        for dx in range(-outline_range, outline_range + 1):
            for dy in range(-outline_range, outline_range + 1):
                draw.text((x + dx, y + dy), line, font=font, fill="black")

        draw.text((x, y), line, font=font, fill="white")

    out_name = f"meme_{template_path.stem}.png"
    out_path = GENERATED_DIR / out_name
    img.save(out_path)

    return out_path


# ---------- STREAMLIT UI ----------
def main():
    st.title("🧠 AI Meme Creator")
    st.write(
        "Generate memes using AI-generated captions or your own text on any image. "
        "You can select from existing templates or upload a custom image."
    )

    image_source = st.radio(
        "Select image source:",
        ["Use template image", "Upload my own image"],
        index=0
    )

    selected_image_path = None
    uploaded_image = None

    if image_source == "Use template image":
        template_files = get_template_files()
        if not template_files:
            st.error("No templates found! Please add some images to the 'templates' folder.")
            return
        template_names = [f.name for f in template_files]
        template_choice = st.selectbox("Choose a template:", template_names)
        selected_image_path = TEMPLATES_DIR / template_choice
        st.image(str(selected_image_path), caption="Template Preview", use_column_width=True)
    else:
        uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
        if uploaded_image is not None:
            st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
        else:
            st.info("Please upload an image to continue.")

    caption_mode = st.radio(
        "How should the caption be generated?",
        ["Use my text", "Generate using AI (LSTM)"],
        index=0
    )

    topic = st.text_input("Enter meme topic / caption:")

    num_words = st.slider("Number of extra words (AI mode only):", 3, 10, 6)

    if st.button("Generate Meme"):
        if image_source == "Use template image":
            if selected_image_path is None:
                st.warning("Please select a template.")
                return
            base_image_path = selected_image_path
        else:
            if uploaded_image is None:
                st.warning("Please upload an image.")
                return
            tmp_path = GENERATED_DIR / "uploaded_image.png"
            img = Image.open(uploaded_image).convert("RGB")
            img.save(tmp_path)
            base_image_path = tmp_path

        if not topic.strip():
            st.warning("Please enter a topic / caption.")
            return

        if caption_mode == "Generate using AI (LSTM)":
            caption = generate_caption(topic, num_words=num_words)
        else:
            caption = topic.strip()

        if not caption:
            st.warning("Caption cannot be empty.")
            return

        meme_path = create_meme_image(base_image_path, caption)

        st.subheader("Generated Caption:")
        st.write(caption)

        st.subheader("Final Meme:")
        st.image(str(meme_path), use_column_width=True)

    st.markdown("---")
    st.caption("Prototype - AI Meme Creator (LSTM + Template-based Professional Captions)")


if __name__ == "__main__":
    main()
