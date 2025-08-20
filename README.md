# AI-Powered Emotion Detector

This project is a full-stack web application that analyzes the emotional sentiment of any given text. It uses a combination of local, lightweight machine learning models for initial analysis and a powerful cloud-based Large Language Model (LLM) to provide nuanced, human-readable explanations.

The frontend is built with React, offering a clean and responsive user interface, while the backend is a lightweight Flask server that orchestrates the AI and NLP processing.

## Features

  * **Sentiment Analysis**: Classifies the overall sentiment of the text as Positive, Negative, or Neutral.
  * **Multi-Label Emotion Detection**: Identifies a mix of up to 28 different emotions present in the text, not just a single one.
  * **Primary Emotion Highlighting**: Displays the top 4 detected emotions, with the most dominant one highlighted as the "Primary Emotion".
  * **Normalized Percentage Display**: The scores for the top 4 emotions are recalculated to show their relative strength, adding up to 100% for an intuitive user experience.
  * **AI-Generated Explanations**: Leverages the Llama 3 model via the Groq API to provide a simple summary and a detailed explanation for why each of the top emotions was detected.
  * **User-Friendly Interface**: A clean, modern UI built with React that includes loading states, error handling, and a character counter.

## Tech Stack

### Frontend

  * **React.js**: For building the user interface.
  * **CSS**: Custom styling for a modern look and feel.
  * **JavaScript (ES6+)**: Core language for frontend logic.

### Backend

  * **Python**: The core language for the server.
  * **Flask**: A lightweight web framework for creating the API.
  * **Transformers (Hugging Face)**: To run the local NLP models for emotion and sentiment classification.
  * **Groq API**: Provides high-speed access to the Llama 3 LLM for generating summaries and explanations.

## Local Setup

To run this project on your local machine, follow these steps.

### Prerequisites

  * **Python 3.8+** and `pip`
  * **Node.js v14+** and `npm`
  * A **Groq API Key**: You can get a free key from the [Groq Console](https://console.groq.com/keys).

### 1\. Clone the Project

First, clone the project from your repository:

```bash
git clone https://github.com/your-username/emotion_detector.git
cd emotion_detector
```

### 2\. Backend Setup

The backend server runs the AI models and the API.

1.  **Navigate to the backend directory**:

    ```bash
    cd backend
    ```

2.  **Create and activate a Python virtual environment**:

      * On macOS/Linux:
        ```bash
        python3 -m venv venv
        source venv/bin/activate
        ```
      * On Windows:
        ```bash
        python -m venv venv
        .\venv\Scripts\activate
        ```

3.  **Install the required Python packages**:

    ```bash
    pip install -r requirements.txt
    ```

4.  **Create the environment file**:
    Create a new file named `.env` inside the `backend` directory.

5.  **Add your Groq API Key** to the `.env` file:

    ```
    GROQ_API_KEY="your-groq-api-key-here"
    ```

6.  **Run the Flask server**:

    ```bash
    flask run
    ```

    The backend will start on `http://127.0.0.1:5000`. It may take a minute the first time to download the local transformer models.

### 3\. Frontend Setup

The frontend is a React application that provides the user interface.

1.  **Open a new terminal** and navigate to the `frontend` directory:

    ```bash
    cd frontend
    ```

2.  **Install the required npm packages**:

    ```bash
    npm install
    ```

3.  **Run the React development server**:

    ```bash
    npm start
    ```

    The frontend application will open automatically in your browser at `http://localhost:3000`.

## How to Use

1.  Make sure both the backend and frontend servers are running.
2.  Open your browser to `http://localhost:3000`.
3.  Type or paste any text into the text area.
4.  Click the "Analyze Emotions" button.
5.  View the AI-generated summary, sentiment, and the breakdown of the top 4 emotions with their respective explanations.
