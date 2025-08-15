# Homework8-SOAR

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
├── app/
│   ├── app.py                   # UI and main orchestrator
│   ├── genai_prescriptions.py   # Prescriptive engine
│   └── train_model.py           # Predictive engine
├── docker/
│   └── Dockerfile               # Container definition
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
