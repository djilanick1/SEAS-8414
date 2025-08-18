# Filename: 1_train_and_export.py

import h2o
import os
import random
import string
import shutil
from h2o.automl import H2OAutoML
from h2o.estimators.glm import H2OGeneralizedLinearEstimator

# --- 1. Initialize H2O and prepare data ---
print("Initializing H2O...")
h2o.init()

# Generate a small, synthetic DGA dataset for demonstration
def generate_dga_dataset():
    """Generates a synthetic dataset for DGA detection."""
    legit_domains = [
        "google.com", "github.com", "wikipedia.org", "openai.com", "h2o.ai",
        "youtube.com", "amazon.com", "twitter.com", "facebook.com", "linkedin.com"
    ]
    dga_domains = [
        "".join(random.choices(string.ascii_lowercase + string.digits, k=random.randint(10, 20))) + ".info"
        for _ in range(50)
    ]
    
    # Add some legitimate-looking but random domains to mix it up
    dga_domains.extend([
        "".join(random.choices(string.ascii_lowercase, k=random.randint(15, 25))) + ".net"
        for _ in range(50)
    ])

    # Calculate length and entropy for each domain
    def calculate_features(domain, is_dga):
        """Calculates length and Shannon entropy."""
        length = len(domain)
        
        # Calculate entropy
        prob_map = {char: domain.count(char) / length for char in set(domain)}
        entropy = -sum(p * h2o.log(p, 2) for p in prob_map.values())
        
        return [domain, length, entropy, is_dga]

    # Create the dataset
    data = []
    for domain in legit_domains:
        data.append(calculate_features(domain, 'legit'))
    for domain in dga_domains:
        data.append(calculate_features(domain, 'dga'))

    # Shuffle the data
    random.shuffle(data)

    # Save to a CSV file
    df_pandas = h2o.H2OFrame([['domain', 'length', 'entropy', 'label']] + data)
    df_pandas.as_data_frame().to_csv('dga_dataset_train.csv', index=False, header=False)
    
    print(" 'dga_dataset_train.csv' created successfully.")

# Generate the dataset first
generate_dga_dataset()
dga_data = h2o.import_file("dga_dataset_train.csv")
dga_data.columns = ['domain', 'length', 'entropy', 'label']

# Convert 'label' to a factor (categorical) for classification
dga_data['label'] = dga_data['label'].asfactor()

# Define features (x) and target (y)
x = ['length', 'entropy']
y = 'label'

# --- 2. Run H2O AutoML ---
print("Starting H2O AutoML process...")
aml = H2OAutoML(
    max_models=10,  # Limit the number of models for a quick demo
    seed=1
)
aml.train(x=x, y=y, training_frame=dga_data)

# --- 3. Identify and export the leader model ---
leader_model = aml.leader
print("\n-------------------------------------------")
print(f" AutoML Leader Model Found: {leader_model.model_id}")
print("-------------------------------------------")

# Create the /model directory if it doesn't exist
model_dir = "model"
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# Export the leader model as a MOJO file
mojo_path = h2o.save_model(model=leader_model, path=model_dir, force=True)

# MOJO files are compressed, so let's move it to a new location
# and rename it for clarity.
final_mojo_path = os.path.join(model_dir, "DGA_Leader.zip")
shutil.move(mojo_path, final_mojo_path)

print(f" Leader model exported as a MOJO file to: {final_mojo_path}")

# --- 4. Shutdown H2O instance ---
h2o.cluster().shutdown()
print("H2O cluster shut down.")