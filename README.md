# 🎯 GesturePilot-RAG

**Gesture-Controlled System with Real-Time RAG-Based Q&A**

## 🚀 Overview

GesturePilot-RAG is an AI-powered system that lets you **control your computer using hand gestures and facial expressions**, while simultaneously enabling **real-time question answering (RAG)** over your session activity.

It combines:

* 👁️ Computer Vision (Face + Hand Tracking)
* 🤖 AI Embeddings + Vector Search (FAISS)
* 💬 Retrieval-Augmented Generation (RAG)
* ⚡ Real-time interaction (multi-threaded system)

---

## ✨ Features

### 🎥 Gesture-Based Controls

* 😊 **Smile (hold)** → Opens YouTube
* ✋ **Open Palm** → Increase Volume
* ✊ **Closed Fist** → Decrease Volume
* ✌ **V-Sign (hold)** → Close tab + Exit

---

### 🧠 RAG-Based Q&A (Live)

Ask questions during or after your session:

```
> How many times did I raise the volume?
> When did YouTube open?
> What was my most used gesture?
```

✔ Uses FAISS vector search
✔ Retrieves relevant logs
✔ Generates answers using LLM (Claude, optional)

---

### 🔐 Face Lock System

* Detects and locks onto a single active user
* Ignores other faces in frame
* Prevents unintended control

---

### 📊 Session Logging

* All gestures are timestamped
* Stored in `gesture_log.txt`
* Used for retrieval + analytics

---

## 🏗️ Architecture

```
Webcam Input
     ↓
Face Recognition + Face Mesh
     ↓
Hand Tracking (MediaPipe)
     ↓
Gesture Detection Logic
     ↓
System Actions (Volume / Browser / Exit)
     ↓
Logging (gesture_log.txt)
     ↓
Embedding (SentenceTransformer)
     ↓
FAISS Vector Index
     ↓
User Query → Retrieval → Answer (RAG)
```

---

## 🛠️ Tech Stack

* OpenCV → Video processing
* MediaPipe → Hand & face tracking
* face_recognition → Face identification
* PyAutoGUI → System control
* FAISS → Vector similarity search
* Sentence Transformers → Embeddings
* Anthropic API → LLM-based answers (optional)
* NumPy → Data processing

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/dkaur1be24-sketch/GesturePilot-RAG.git
cd GesturePilot-RAG
```

---

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

---

### 3. Set API Key (optional for RAG answers)

```bash
export ANTHROPIC_API_KEY="your_api_key_here"
```

If not set → system still works, but returns raw log entries.

---

## ▶️ Usage

Run the main script:

```bash
python main.py
```

---

### 🎮 Controls

| Gesture       | Action          |
| ------------- | --------------- |
| Smile (hold)  | Open YouTube    |
| Open Palm     | Volume Up       |
| Closed Fist   | Volume Down     |
| V-Sign (hold) | Exit            |
| R key         | Reset face lock |
| Q key         | Quit            |

---

### 💬 Ask Questions (while running)

Type in terminal:

```
❓ Ask about your session: How many times did I increase volume?
```

---

## 📁 Project Structure

```
GesturePilot-RAG/
│
├── main.py                # Main application
├── requirements.txt      # Dependencies
├── gesture_log.txt       # Session logs (auto-generated)
├── README.md             # Documentation
```

---

## ⚠️ Notes

* Ensure webcam access is enabled
* Good lighting improves detection accuracy
* First run may download embedding model (~90MB)
* FAISS index updates every few seconds

---

## 🔮 Future Improvements

* Add more gestures (scroll, play/pause, etc.)
* GUI dashboard for analytics
* Multi-user recognition
* Voice + gesture hybrid control
* Cloud-based logging + dashboard

---

## 🤝 Contribution

Feel free to fork, improve, and submit pull requests.

---

## 📜 License

This project is open-source and available under the MIT License.

---

