# ASAPP - AI-Powered Assistant

ASAPP is a full-stack application that provides an AI-powered assistant capable of answering questions and making predictions. It features a React-based frontend and a Python backend with a machine learning model and a Retrieval-Augmented Generation (RAG) pipeline.

## Features

*   **Question Answering:** Get answers to your questions through a user-friendly interface.
*   **Predictions:** Leverage a trained model to get predictions based on your input.
*   **RAG Pipeline:** The backend uses a RAG service to provide more accurate and context-aware answers.
*   **Model Training:** The project includes scripts to train your own intent model.
*   **Data Ingestion:** Scripts to ingest your own data for the RAG service.

## Technologies Used

*   **Frontend:**
    *   React
    *   Tailwind CSS
*   **Backend:**
    *   Python
    *   FastAPI
    *   PyTorch
    *   Transformers
    *   LangChain
    *   Jupyter Notebook

## Project Structure

The project is divided into two main parts:

*   `frontend/`: Contains the React application that provides the user interface.
*   `backend/`: Contains the Python backend, including the FastAPI server, machine learning model, and RAG pipeline.

## Backend Setup and Usage

The backend is a Python project that serves the AI/ML models and provides an API for the frontend.

### Setup

1.  **Navigate to the backend directory:**
    ```bash
    cd backend
    ```

2.  **Create a virtual environment:**
    ```bash
    python -m venv .venv
    ```

3.  **Activate the virtual environment:**
    *   On Windows:
        ```bash
        .venv\Scripts\activate
        ```
    *   On macOS/Linux:
        ```bash
        source .venv/bin/activate
        ```

4.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### Running the Backend Server

The backend server provides the API for the frontend.

1.  Make sure you are in the `backend` directory and the virtual environment is activated.
2.  **Run the FastAPI server:**
    ```bash
    uvicorn main:app --reload
    ```
    The server will be available at `http://127.0.0.1:8000`.

### Running the Jupyter Notebook

The Jupyter Notebook is used for tasks like dataset generation.

1.  **Install Jupyter Notebook:**
    If you haven't installed it during the initial setup:
    ```bash
    pip install jupyter
    ```

2.  **Start Jupyter Notebook:**
    Make sure you are in the `backend` directory and the virtual environment is activated.
    ```bash
    jupyter notebook
    ```

3.  **Open the notebook:**
    In the Jupyter interface that opens in your browser, navigate to the `notebooks/` directory and click on `dataset_generator.ipynb` to open it.

### Training the Model

The project includes a script to run the training pipeline for the intent model.

1.  Make sure you are in the `backend` directory and the virtual environment is activated.
2.  **Run the training pipeline:**
    ```bash
    python run_training_pipeline.py
    ```
    This will train the model and save the output in the `training_output/` directory.

## Frontend Setup and Usage

The frontend is a React application that provides the user interface for interacting with the AI assistant.

### Setup

1.  **Navigate to the frontend directory:**
    ```bash
    cd frontend
    ```

2.  **Install the required dependencies:**
    ```bash
    npm install
    ```

### Running the Frontend Development Server

1.  Make sure you are in the `frontend` directory.
2.  **Start the React development server:**
    ```bash
    npm start
    ```
    The application will open in your browser at `http://localhost:3000`.

## How to Use

1.  Start the backend server as described in the "Backend Setup and Usage" section.
2.  Start the frontend development server as described in the "Frontend Setup and Usage" section.
3.  Open your browser and navigate to `http://localhost:3000`.
4.  Use the interface to ask questions or get predictions.