# Application Runbook: ASAPP

This document provides the standard operating procedures for running the ASAPP application, which consists of a Python backend and a React frontend.

## 1. Prerequisites

Ensure the following software is installed on your system:

*   Python (3.9 or higher)
*   Node.js (v14 or higher) and npm

## 2. Backend Setup and Execution

The backend server provides the core API and machine learning services. It must be running for the frontend to function correctly.

1.  **Open a new terminal session.**
2.  **Navigate to the backend directory:**
    ```bash
    cd C:\Mehul\ALLmyCODINGstuff\Hackathons\ASAPP\backend
    ```
3.  **Create and activate a Python virtual environment:**
    *   On Windows:
        ```bash
        python -m venv .venv
        .venv\Scripts\activate
        ```
    *   On macOS/Linux:
        ```bash
        python3 -m venv .venv
        source .venv/bin/activate
        ```
4.  **Install the required Python dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
5.  **Start the backend server:**
    ```bash
    uvicorn main:app --reload
    ```
    The backend is now running and accessible at `http://127.0.0.1:8000`. **Leave this terminal running.**

## 3. Frontend Setup and Execution

The frontend provides the user interface for the application.

1.  **Open a new, separate terminal session.**
2.  **Navigate to the frontend directory:**
    ```bash
    cd C:\Mehul\ALLmyCODINGstuff\Hackathons\ASAPP\frontend
    ```
3.  **Install the required Node.js dependencies:**
    ```bash
    npm install
    ```
4.  **Start the frontend application:**
    ```bash
    npm start
    ```
    A browser window should automatically open to `http://localhost:3000`. If it does not, open your web browser and navigate to this address manually.

## 4. Verification

*   The backend terminal (from Step 2) should show active logs for `uvicorn` without any critical errors.
*   The frontend terminal (from Step 3) should indicate that the React development server is running successfully.
*   The application should be fully interactive and usable in the browser at `http://localhost:3000`.

## 5. Shutdown Procedure

To stop the application, you must stop both the frontend and backend processes.

1.  **Stop the frontend:** In the frontend terminal, press `Ctrl + C`.
2.  **Stop the backend:** In the backend terminal, press `Ctrl + C`.
3.  **(Optional) Deactivate the backend virtual environment:**
    ```bash
    deactivate
    ```
