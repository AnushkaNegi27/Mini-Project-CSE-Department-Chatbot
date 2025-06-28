# CSE Department Chatbot

A smart, interactive chatbot designed for the Computer Science & Engineering Department at Graphic Era Hill University. This chatbot answers department-related queries using NLP (Natural Language Processing) and provides a sleek web interface with avatars, timestamps, and a responsive UI.

---

## Description

This project uses PyTorch-based NLP techniques to classify user queries and respond accurately using a custom dataset. It is integrated with a Flask backend and a modern HTML/CSS frontend.

---

## Features

- 🔍 Query classification using bag-of-words and a neural network
- 💬 Dynamic, styled frontend with user and bot avatars
- ⏰ Timestamps and smooth chat experience
- 🌐 Web-based using Flask (no Tkinter!)
- 📦 Clean file structure, virtual environment isolated

---

## Installation & Run

```bash
git clone https://github.com/AnushkaNegi27/Mini-Project-CSE-Department-Chatbot.git
```
```bash
cd Mini-Project-CSE-Department-Chatbot
```

# Setup virtual environment (recommended)
```bash
python -m venv venv
```
```bash
venv\Scripts\activate     # On Windows
```

# Install dependencies
```bash
pip install -r requirements.txt
```

# Run the Flask app
```bash
python app.py
```

## Requirements

If requirements.txt is missing, you can manually install:
```bash
pip install flask nltk numpy torch
```

## Files and Directories

```nltk_utils.py``` : Utility functions for NLP tasks (tokenization, stemming, etc.)

```training.py``` : Script for training the chatbot model.

```cse_chatbot.py``` : Python file to load the model and get responses.

```app.py``` : Flask backend to connect model to UI.

```templates/index.html``` : Frontend UI with avatars, timestamp, and live interaction.

```static/style.css``` : CSS for styling the UI.

```cse_chatbot_dataset.json``` : Dataset for CSE Department Chatbot.

```cse_chatbot.pth``` : Trained model file.

```Research_Paper_CSE_Department_Chatbot.pdf``` : NLP research paper.

```Anushka_Negi_K1_2318473.pdf``` : Detailed project report.

```CSE_Chatbot_Presentation.pptx``` : Presentation of project goals, approach, and results.


