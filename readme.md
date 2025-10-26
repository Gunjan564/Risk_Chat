# 🧠 Mental Health Post Risk Detector

This project is a web application designed to analyze text (simulated social media posts) for mental health risk levels using a deep learning ensemble model. The system operates using a two-tier architecture: a Streamlit frontend and a dedicated Python API backend.

| Service | Technology | Role | Port |
| :--- | :--- | :--- | :--- |
| **Frontend** (`app.py`) | Streamlit | Provides the web interface and handles user input. | `5000` |
| **Backend API** (`api_server.py`) | Flask/PyTorch/Transformers | Hosts the heavy `MentalHealthEnsemble` and performs predictions. | `5001` |
| **Database** (`risk_analysis_log.db`) | SQLite (via SQLAlchemy) | Logs every single analysis request for auditing purposes. | (File-based) |

***

## 🚀 Setup and Installation

Follow these steps exactly to set up and run both the backend and frontend services.

### 1. Create and Activate Virtual Environment

It's critical to use a virtual environment to manage dependencies.

```bash
# Navigate to the project folder (RISK_CHAT)
cd C:\Users\hp\OneDrive\Desktop\Risk_Chat

# Create a new environment
python -m venv venv_new

# Activate the environment
.\venv_new\Scripts\activate

2. Install Dependencies
Install all required packages in the active environment:

# Install core libraries and data science dependencies
pip install streamlit requests sqlalchemy pandas scikit-learn torch transformers joblib flask flask-cors


3. Setup Model Files
Ensure your cleaned model checkpoint files (without optimizer.pt, etc.) and the ensemble files (meta_model.joblib, ensemble_metadata.pt) are correctly placed inside the models folder, matching the paths defined in model_utils.py.

4. Set Environment Variable
The application requires an environment variable to locate the database file for logging. Set this in your terminal:

set DATABASE_URL=sqlite:///./risk_analysis_log.db

▶️ Running the Application

You must run the services in the order below, using two separate terminal windows.
A. Start the Backend Model Service (Terminal 1)

This service loads the large models and listens for requests on port 5001.

1. Open a FIRST terminal and activate your environment.
2. Run the server using the API file (api_server.py):Bash(venv_new)       
    C:\Users\hp\OneDrive\Desktop\Risk_Chat>python api_server.py
Wait until you see the confirmation messages (e.g., "✅ Ensemble system loaded successfully") and the server running on port 5001.
B. Start the Frontend Web App (Terminal 2)
This starts the Streamlit interface on port 5000.
1. Open a SECOND terminal and activate your environment.
2. Run the Streamlit app:
    Bash(venv_new) C:\Users\hp\OneDrive\Desktop\Risk_Chat> streamlit run app.py
The application will open in your browser at http://localhost:5000/.

⚠️ Troubleshooting

ConnectionError / Model service timed out.,The backend server (api_server.py) is either not running or took too long to load the large models.,"1. Check Terminal 1: Ensure api_server.py is running and says Running on http://0.0.0.0:5001/. 2. Increase Timeout: If models are slow, modify timeout in APIModelClient.predict (in app.py) to timeout=60 and restart both services."
ModuleNotFoundError,The dependency was installed in the wrong Python environment.,Recreate the virtual environment and ensure all pip install commands run while the environment is fully active. Use .\venv_new\Scripts\python.exe -m streamlit run app.py to guarantee the correct Python is used.
ValueError: DATABASE_URL environment variable not set,The mandatory variable was not set in the terminal.,Run set DATABASE_URL=sqlite:///./risk_analysis_log.db in the terminal before running the app.