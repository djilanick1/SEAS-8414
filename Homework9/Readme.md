# The Prescriptive DGA Detector

## Project Goal

The **Prescriptive DGA Detector** is an end-to-end Python application that demonstrates a modern, three-stage workflow for cybersecurity analysis: **Automated Machine Learning (AutoML) -> Explainable AI (XAI) -> Generative AI (GenAI)**.

This tool goes beyond a simple binary verdict of "malicious" or "legitimate." For a given domain name, it first classifies it as a potential **Domain Generation Algorithm (DGA)** domain. If a DGA is detected, the application provides a clear, human-readable explanation of **why** the model made that decision. Finally, it uses those findings to automatically generate a **prescriptive incident response playbook** for a security analyst, turning a complex finding into an actionable set of steps.

## Architecture and Workflow

The project's pipeline is divided into three distinct stages, implemented across two Python scripts:

### Stage 1: Model Training and Export

This stage is handled by `1_train_and_export.py`. It uses the **H2O AutoML** framework to rapidly train and select the best-performing model for DGA detection from a synthetic dataset. The winning model (the "leader") is then exported into a portable, production-ready **MOJO file**. This approach ensures that the detection model is both high-performance and easy to deploy.

### Stage 2: Prediction and Explanation

This is the core of the application, implemented in `2_analyze_domain.py`. This script loads the pre-trained MOJO model and takes a domain name as a command-line argument. It performs two key functions:

1. **Prediction:** It passes the domain's features (length and entropy) to the model to get a `legit` or `dga` classification.

2. **Explanation (XAI):** If a DGA is detected, the script generates a **SHAP (SHapley Additive exPlanations)** explanation for that specific prediction. This step reveals which features (e.g., high entropy or unusual length) were most influential in the model's decision.

### Stage 3: Prescription

The final stage, also in `2_analyze_domain.py`, acts as a bridge from explanation to action. The dynamically generated SHAP findings are programmatically formatted into a structured prompt. This prompt is then sent to a **Google Generative AI** model, which synthesizes the technical findings into a complete, easy-to-follow, and prescriptive incident response playbook for an analyst. This final output is printed to the console, completing the end-to-end workflow.

## Usage Instructions

### Prerequisites

* **Python 3.10 or 3.11:** The project is tested to work with these Python versions.

* **Google API Key:** You must have an API key for the Google Generative AI API.

* **Virtual Environment:** It is highly recommended to use a virtual environment to manage dependencies.

### Step 1: Environment Setup

First, create and activate a new virtual environment, and then install the required libraries.

```
# Create and activate the virtual environment
python3.11 -m venv dga_env
source dga_env/bin/activate

# Install dependencies
pip install h2o shap pandas aiohttp requests


```

### Step 2: Set Your API Key

Export your Google API key as an environment variable.

```
export GOOGLE_API_KEY="YOUR_API_KEY_HERE"


```

### Step 3: Train and Export the Model

Run the training script. This will generate a synthetic dataset and save the trained DGA detection model as `DGA_Leader.zip` in the `./model/` directory.

```
python 1_train_and_export.py


```

### Step 4: Analyze a Domain

Now you can use the main application to analyze a domain.

**Example 1: Analyzing a legitimate domain**

```
python 2_analyze_domain.py --domain example.com


```

**Example 2: Analyzing a DGA-like domain**

```
python 2_analyze_domain.py --domain d6w8x4z1s9q.net


```

The script will print the final playbook to your console if a DGA is detected.
