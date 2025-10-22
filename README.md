# 🧠 Intent Classifier: A Full-Stack AI Application

Welcome to the **Intent Classifier** project! This repository hosts a full-stack application designed to predict user intent from text utterances and collect feedback to improve the model over time. It features a FastAPI backend for the AI logic and a dynamic React frontend for user interaction.

---

## ✨ Features

*   **Intent Prediction:** Submit a natural language utterance and receive a predicted intent from the AI model.
*   **User Feedback System:** Provide feedback on predictions (correct/incorrect) to help retrain and improve the model.
*   **Dynamic Intent Listing:** The frontend dynamically fetches all possible intent labels from the backend for feedback correction.
*   **Modern & Responsive UI:** A clean, intuitive, and responsive user interface built with React and styled using Tailwind CSS.
*   **CORS Enabled:** Seamless communication between the frontend and backend.

---

## 🚀 API Endpoint Specification (FastAPI Backend)

The backend is built with FastAPI, providing a robust and high-performance API for intent classification and feedback logging.

### 1. Get an Intent Prediction

*   **Endpoint:** `POST /predict`
*   **Description:** Takes a user's text utterance and returns the model's predicted intent.
*   **Request Body:**
    ```json
    {
      "utterance": "how much is in my savings account?"
    }
    ```
*   **Success Response (200 OK):**
    ```json
    {
      "utterance": "how much is in my savings account?",
      "predicted_intent": "check_balance"
    }
    ```

### 2. Submit User Feedback

*   **Endpoint:** `POST /feedback`
*   **Description:** Logs user feedback about a prediction to the database. This data is crucial for improving the model through retraining.
*   **Request Body:**
    ```json
    {
      "utterance": "where can I find my statements?",
      "predicted_intent": "check_balance",
      "is_correct": false,
      "correct_intent": "get_statements"
    }
    ```
*   **Success Response (201 Created):**
    ```json
    {
      "status": "Feedback received"
    }
    ```

### 3. Get All Possible Intents

*   **Endpoint:** `GET /intents`
*   **Description:** Returns a list of all unique intent labels the model currently recognizes. This is used to populate dropdowns in the UI for correcting predictions.
*   **Request Body:** None
*   **Success Response (200 OK):**
    ```json
    [
      "check_balance",
      "get_loan_info",
      "get_statements",
      "transfer_money"
    ]
    ```

---

## 🌐 Frontend (React Application)

The frontend is a dynamic Single-Page Application (SPA) built with React, providing an interactive interface for the intent classifier.

### Technologies

*   **React:** A JavaScript library for building user interfaces.
*   **Vite:** A fast build tool that provides an excellent development experience.
*   **Tailwind CSS:** A utility-first CSS framework for rapidly building custom designs.
*   **Axios:** A promise-based HTTP client for making API requests.

### Running the Frontend

1.  **Navigate to the frontend directory:**
    ```bash
    cd frontend
    ```
2.  **Install dependencies:**
    ```bash
    npm install
    ```
3.  **Start the development server:**
    ```bash
    npm run dev
    ```
    The application will typically open in your browser at `http://localhost:5173`.

### UI Components

The frontend is composed of two main views:

*   **`PredictionPage.jsx`**: This is the initial view where users can input an utterance and get an intent prediction. It also allows users to provide immediate feedback on the prediction.
*   **`AnswerPage.jsx`**: (Assumed to be the page where the correct intent is displayed or further actions are taken based on the intent). This page is navigated to when a prediction is confirmed as correct.

---

## ⚙️ Backend (FastAPI Application)

The backend handles the core logic of intent prediction and feedback logging.

### Technologies

*   **FastAPI:** A modern, fast (high-performance) web framework for building APIs with Python 3.7+ based on standard Python type hints.
*   **Pydantic:** Used by FastAPI for data validation and settings management.
*   **`dotenv`:** For loading environment variables.
*   **CORS Middleware:** Configured to allow cross-origin requests from the frontend.

### Running the Backend

1.  **Navigate to the project root directory:**
    ```bash
    cd C:\Mehul\ALLmyCODINGstuff\Hackathons\ASAPP
    ```
2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv .venv
    # On Windows:
    .venv\Scripts\activate
    # On macOS/Linux:
    source .venv/bin/activate
    ```
3.  **Install backend dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Run the FastAPI application:**
    ```bash
    uvicorn main:app --reload
    ```
    The API will be available at `http://127.0.0.1:8000`.

### Cross-Origin Resource Sharing (CORS)

CORS is a security feature implemented in web browsers that prevents web pages from making requests to a different domain than the one that served the web page. Since our React frontend runs on `http://localhost:5173` and our FastAPI backend runs on `http://127.0.0.1:8000`, they are considered different origins.

To allow the frontend to communicate with the backend, CORS must be explicitly enabled on the backend. This project's `main.py` includes the necessary FastAPI `CORSMiddleware` configuration to permit requests from `http://localhost:5173`.

```python
from fastapi.middleware.cors import CORSMiddleware

# ... (other imports and app initialization)

origins = [
    "http://localhost:5173", # Allow requests from your React frontend
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"], # Allows all methods (GET, POST, etc.)
    allow_headers=["*"], # Allows all headers
)
```

---

## 🛠️ Setup and Installation

Follow these steps to get the entire application running on your local machine.

### Prerequisites

*   Python 3.7+
*   Node.js (LTS recommended) & npm
*   Git

### Steps

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd ASAPP # Or whatever your project root directory is named
    ```
2.  **Backend Setup:**
    *   Follow the "Running the Backend" steps above. Ensure your virtual environment is activated and dependencies are installed.
3.  **Frontend Setup:**
    *   Follow the "Running the Frontend" steps above. Ensure you navigate into the `frontend` directory, install dependencies, and start the development server.

---

## 💡 Usage

1.  **Start both the backend and frontend servers.**
2.  Open your web browser and navigate to `http://localhost:5173`.
3.  Enter an utterance in the input field and click "Predict".
4.  Review the predicted intent.
5.  Provide feedback:
    *   Click "Yes, Correct" if the prediction is accurate.
    *   Click "No, Incorrect" if the prediction is wrong, then select the correct intent from the dropdown and click "Submit Correction".
6.  The feedback will be logged by the backend, contributing to future model improvements.

---

## 🤝 Feedback and Contribution

We welcome feedback and contributions to this project!

*   **Report Bugs:** If you find any issues, please open an issue on the GitHub repository.
*   **Suggest Features:** Have an idea for a new feature? Let us know by opening an issue.
*   **Contribute Code:** Feel free to fork the repository, make your changes, and submit a pull request.

---

## 📄 License

This project is licensed under the MIT License - see the `LICENSE` file for details.
