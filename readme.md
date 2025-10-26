# 🧠 Mental Health Post Risk Detector

This project is a **web application** designed to analyze text (simulated social media posts) for **mental health risk levels** using a **deep learning ensemble model**.  
The system operates using a **two-tier architecture** — a Streamlit frontend and a dedicated Python API backend.

---

## System Overview

| Service | Technology | Role | Port |
| :--- | :--- | :--- | :--- |
| **Frontend** (`app.py`) | Streamlit | Provides the web interface and handles user input | `5000` |
| **Backend API** (`api_server.py`) | Flask / PyTorch / Transformers | Hosts the `MentalHealthEnsemble` model and performs predictions | `5001` |
| **Database** (`risk_analysis_log.db`) | SQLite (via SQLAlchemy) | Logs every analysis request for auditing and tracking | *(File-based)* |

---

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


---

## ⚠️ Troubleshooting Guide

| **Error Message** | **Cause** | **Fix** |
| :--- | :--- | :--- |
| **ConnectionError / Model service timed out** | The backend (`api_server.py`) is not running or the models took too long to load. | 1. Check **Terminal 1** – ensure the API server is running and shows:<br>`Running on http://0.0.0.0:5001/`.<br>2. Increase timeout: open `app.py`, find `APIModelClient.predict`, and set `timeout=60`. Restart both services. |
| **ModuleNotFoundError** | Dependencies installed in the wrong Python environment. | Recreate the virtual environment and reinstall all dependencies. Always ensure the environment is active before installing packages.<br>Alternatively, run:<br>`.\venv_new\Scripts\python.exe -m streamlit run app.py` |
| **ValueError: DATABASE_URL environment variable not set** | The required environment variable is missing. | Run:<br>`set DATABASE_URL=sqlite:///./risk_analysis_log.db`<br>before starting the app. |

---

## 📁 Project Structure

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
