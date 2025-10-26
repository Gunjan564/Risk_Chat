# 🧠 Mental Health Post Risk Detector

This project is a web application designed to analyze text (simulated social media posts) for mental health risk levels using a deep learning ensemble model. The system operates using a two-tier architecture: a Streamlit frontend and a dedicated Python API backend.

***

## 🛠️ System Architecture

The application is structured as a full-stack Python deployment with two distinct services:

| Service | Technology | Role | Port |
| :--- | :--- | :--- | :--- |
| **Frontend** (`app.py`) | Streamlit | Provides the web interface and handles user input. | `5000` |
| **Backend API** (`api_server.py`) | Flask/PyTorch/Transformers | Hosts the heavy `MentalHealthEnsemble` and performs predictions. | `5001` |
| **Database** (`risk_analysis_log.db`) | SQLite (via SQLAlchemy) | Logs every single analysis request for auditing purposes. | (File-based) |

***

## 🤖 Model Architecture

To ensure high accuracy and robust predictions, this system uses a **stacking ensemble** model. Instead of relying on a single model, it combines the "opinions" of three different Transformer models.

1.  **Base Models:** Three pre-trained Transformer models are used to individually analyze the text:
    * **XLNet**
    * **DistilBERT**
    * **Mental-RoBERTa** (A RoBERTa model fine-tuned on mental health-related text)

2.  **Meta-Model (Stacking):** The probability scores from all three base models are collected. These probabilities are then used as input features for a final, lightweight classifier (`LogisticRegression`, loaded from `meta_model.joblib`). This "meta-model" weighs the strengths of each base model to make the final decision.

This ensemble approach is more resilient to errors and typically outperforms any single model.

***

## ⚙️ How It Works (Prediction Pipeline)

When a user analyzes a post, the following sequence occurs:

1.  **Input:** A user pastes text into the **Streamlit Frontend** (port `5000`) and clicks "Analyze".
2.  **API Request:** The Streamlit app sends the raw text via an HTTP request to the **Flask Backend API** (port `5001`) at the `/predict_sentiment` endpoint.
3.  **Backend Analysis:**
    * The Flask server receives the text.
    * The text is preprocessed and fed into all three base models (XLNet, DistilBERT, RoBERTa) in parallel.
    * Each base model outputs a probability score for the four risk classes (`no risk`, `low`, `moderate`, `high`).
4.  **Meta-Prediction:**
    * The probability scores are stacked together into a new feature vector.
    * This vector is passed to the loaded `meta_model` (Logistic Regression), which makes the final prediction and calculates a confidence score.
5.  **API Response:** The Flask server sends the final result (e.g., `{"sentiment": "high", "confidence": 0.88}`) back to the Streamlit app.
6.  **Display:** The Streamlit app receives the JSON response and displays the risk badge, confidence score, and relevant crisis resources to the user.
7.  **Logging:** The original text, final prediction, and confidence are saved to the `risk_analysis_log.db` SQLite database for auditing.

***

## Setup and Installation

Follow these steps exactly to set up and run both the backend and frontend services.

### 1. Create and Activate a Virtual Environment

It’s **critical** to use a virtual environment to manage dependencies.

```bash
# Navigate to the project folder
cd C:\Users\hp\OneDrive\Desktop\Risk_Chat

# Create a new virtual environment
python -m venv venv_new

# Activate the environment
.\venv_new\Scripts\activate

# Run the Streamlit app
streamlit run app.py

```
---
## ⚠️ Troubleshooting Guide

| **Error Message** | **Cause** | **Fix** |
| :--- | :--- | :--- |
| **ConnectionError / Model service timed out** | The backend (`api_server.py`) is not running or the models took too long to load. | 1. Check **Terminal 1** – ensure the API server is running and shows:<br>`Running on http://0.0.0.0:5001/`.<br>2. Increase timeout: open `app.py`, find `APIModelClient.predict`, and set `timeout=60`. Restart both services. |
| **ModuleNotFoundError** | Dependencies installed in the wrong Python environment. | Recreate the virtual environment and reinstall all dependencies. Always ensure the environment is active before installing packages.<br>Alternatively, run:<br>`.\venv_new\Scripts\python.exe -m streamlit run app.py` |
| **ValueError: DATABASE_URL environment variable not set** | The required environment variable is missing. | Run:<br>`set DATABASE_URL=sqlite:///./risk_analysis_log.db`<br>before starting the app. |

---

## 📁 Project Structure
```
Risk_Chat/
│
├── app.py # Streamlit frontend
├── api_server.py # Flask backend API
├── model_utils.py # Model loading and prediction logic
├── models/ # Saved model and ensemble files
│ ├── meta_model.joblib
│ └── ensemble_metadata.pt
├── risk_analysis_log.db # SQLite database for logs
├── requirements.txt # Optional dependency list
└── README.md # Project documentation
```

---

## 💡 Notes

- Always start **`api_server.py`** before **`app.py`**, since the frontend depends on the backend API.  
- Use the same **virtual environment** for both terminals.  
- You can stop both services anytime using **Ctrl + C** in their respective terminals.

---

## 🧩 Future Improvements

- Add user authentication and session management  
- Integrate visualization dashboards for historical risk trends  
- Deploy the system using Docker or cloud services  
- Add multi-language text support  

---

**Author:** Gunjan  
**License:** MIT  
**Frameworks:** Streamlit · Flask · PyTorch · Transformers  
