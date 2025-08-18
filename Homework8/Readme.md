# Homework8-SOAR for Phishing Analysis

## overview 

This project is a prototype Security Orchestration, Automation, and Response (SOAR) application built with Python. It uses a machine learning model to predict if a URL is malicious and leverages Generative AI to prescribe a response plan. The entire application is containerized with Docker and orchestrated with Docker Compose for easy setup and deployment.

## Dual-Model Architecture

1. **Classification Model (RandomForest)**:

Input: URL features (SSL state, length, subdomains)
Output: Malicious/Benign verdict

2. **Clustering Model (K-Means)**:

Input: Malicious URL features
Output: Threat actor profile (APT, Cybercrime, Hacktivist)

## Features
-   **Supervised Learning**: Binary classification of URLs as malicious/benign
-   **Unsupervised Learning**: Threat actor attribution via clustering
-   **GenAI Integration**: Automated response plan generation
-   **Predictive Analytics**: Uses PyCaret to automatically train a model on a real-world phishing dataset.
-   **Prescriptive Analytics**: Integrates with Google Gemini, OpenAI, and Grok to generate detailed incident response plans.
-   **Interactive UI**: A user-friendly web interface built with the latest version of Streamlit.
-   **Containerized**: Fully containerized with Docker and managed with Docker Compose for a reproducible environment.
-   **Simplified Workflow**: A `Makefile` provides simple commands for building, running, and managing the application.

## Prerequisites

Before you begin, ensure you have the following installed on your system:
-   [Docker](https://www.docker.com/get-started)
-   [Docker Compose](https://docs.docker.com/compose/install/) (often included with Docker Desktop)
-   [Make](https://www.gnu.org/software/make/) (pre-installed on most Linux/macOS systems; Windows users can use WSL or Chocolatey).
-   API keys for at least one Generative AI service (Gemini, OpenAI, or Grok).

## Setup & Installation

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/djilanick1/SEAS-8414.git
    cd Homework8
    ```

2.  **Configure API Keys**
    This is the most important step. The application reads API keys from a `secrets.toml` file.

    -   Create the directory and file if doesn't exist; technically it should be there if git clone was performed
        ```bash
        mkdir -p .streamlit
        touch .streamlit/secrets.toml
        ```
    -   Open `.streamlit/secrets.toml` and add your API keys. Use the following template:
        ```toml
        # vim or vi .streamlit/secrets.toml
        OPENAI_API_KEY = "sk-..."
        GEMINI_API_KEY = "AIza..."
        GROK_API_KEY = "gsk_..."
        ```
        *You only need to provide a key for the service(s) you intend to use.*
    

## Running the Application

With the `Makefile`, running the application is simple.

-   **To build and start the application:**
    ```bash
    make up
    ```
    The first time you run this, it will download the necessary Docker images and build the application container. This may take a few minutes. Subsequent runs will be much faster.

-   Once it's running, open your web browser and go to:
    **[http://localhost:8501](http://localhost:9001)**

-   **To view the application logs:**
    ```bash
    make logs
    ```

-   **To stop the application:**
    ```bash
    make down
    ```

-   **To perform a full cleanup** (stops containers and removes generated model/data files):
    ```bash
    make clean
    ```


## Project Roadmap

1.  **Environment**: Define **Dockerfile**, **docker-compose.yml**, **Makefile**.
2.  **Predictive Engine** (train_model.py):
    * Create synthetic data generator.
    * Integrate PyCaret to train and save the model.
3.  **Prescriptive Engine** (genai_prescriptions.py):
    * Define a robust prompt for JSON output.
    * Implement API calls for GenAI services.
4.  **UI & Orchestration** (app.py):
    * Build input sidebar.
    * Orchestrate the prediction/prescription workflow.
    * Design output tabs for results.

***

## Project Structure

```bash
Homework8-SOAR/
├── .github/
│   └── workflows/
│       └── lint.yml             # For GitHub Actions
├── .streamlit/
│   └── secrets.toml
├── base_app/
│   ├── app.py                   # UI and main orchestrator
│   ├── genai_prescriptions.py   # Prescriptive engine
│   └── train_model.py           # Predictive engine
├── updated_app/
│   ├── updated_app.py                   # UI and main orchestrator
│   ├── updated_genai_prescriptions.py   # Prescriptive engine
│   └── updated_train_model.py           # Predictive engine
├── Dockerfile               # Container definition
├── docs/
│   ├── INSTALL.md               # Installation guide
│   └── TESTING.md               # How to test
├── models/
├── .dockerignore
├── .gitignore
├── docker-compose.yml           # Docker orchestration
├── Makefile                     # Command shortcuts
├── README.md                    # Project overview
└── requirements.txt             # Python libraries


