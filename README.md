🧠 AI Meme Creator

An AI-powered meme generation system using LSTM, NLP, and Streamlit.
This project allows users to upload or choose a meme template, generate captions using AI, or enter custom captions, and download the final meme instantly.

📌 Project Overview

Meme creation is usually a manual process requiring creativity and effort.
This project automates meme generation using:

LSTM (Long Short-Term Memory) model for caption generation

PIL (Pillow) for image processing

Streamlit for user interface

Meme Generator dataset from Kaggle

Users can:

✔ Upload their own images
✔ Use built-in templates
✔ Generate captions using AI
✔ Write their own captions
✔ Download the final meme

🚀 Features
🎨 Meme Creation

Upload any image OR choose from built-in meme templates

Add text caption or generate using AI

Auto-fit caption inside image

High-quality export

🤖 AI Caption Generator

Uses an LSTM model trained on meme captions

Avoids random or meaningless words

Generates short, clean, relevant captions

🧰 Additional Functionalities

Clean UI built using Streamlit

Downloadable output

Template preview

Supports .jpg, .jpeg, .png

🏗️ Project Structure
AI-Meme-Creator/
│
├── app/
│   ├── app.py                # Main Streamlit UI
│
├── src/
│   ├── train_lstm.py         # Model training script
│
├── models/
│   ├── tokenizer.pkl         # Saved tokenizer
│   └── meme_lstm.h5          # (Ignored in GitHub)
│
├── templates/                # Meme template images
├── generated/                # Created memes (ignored)
│
├── data/
│   ├── train_captions.csv
│   ├── val_captions.csv
│
├── requirements.txt
├── README.md
└── .gitignore

🧩 Technologies Used
Technology	Purpose
Python	Main programming
TensorFlow / Keras	LSTM model
PIL (Pillow)	Image editing
Streamlit	Web UI
NumPy / Pandas	Data processing
Kaggle Dataset	Training data
🔧 How to Run Locally
1️⃣ Create virtual environment
python -m venv venv
venv\Scripts\activate

2️⃣ Install dependencies
pip install -r requirements.txt

3️⃣ Run the app
streamlit run app/app.py


Your app will open in browser automatically.

📊 Model Training

The LSTM model was trained using:

Tokenized captions

Sequence padding

10 epochs

Softmax classifier

Model generates short, meaningful captions.

🧪 Results

Caption generation time: <1 second

Validation accuracy: ~85%

Works on both template and uploaded images

Clean English sentences using improved logic
