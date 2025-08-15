# train_model.py
import pandas as pd
import numpy as np
from pycaret.classification import *
from pycaret.clustering import setup as setup_cluster, create_model, save_model as save_cluster_model
import os
import matplotlib.pyplot as plt


def generate_synthetic_data(num_samples=500):
    """
    Generates a synthetic dataset of phishing and benign URL features
    with added features for threat actor profiling.
    """
    print("Generating synthetic dataset...")

    # The features used for both classification and clustering
    features = [
        'having_IP_Address', 'URL_Length', 'Shortining_Service',
        'having_At_Symbol', 'double_slash_redirecting', 'Prefix_Suffix',
        'having_Sub_Domain', 'SSLfinal_State', 'URL_of_Anchor', 'Links_in_tags',
        'SFH', 'Abnormal_URL', 'topical_keywords'
    ]

    num_phishing = num_samples // 2
    num_benign = num_samples - num_phishing

    # Data for malicious phishing URLs, designed to form three clusters
    phishing_data = {
        # State-Sponsored Profile: High sophistication
        'having_IP_Address': np.random.choice([1, -1], num_phishing, p=[0.05, 0.95]),
        'URL_Length': np.random.choice([1, 0, -1], num_phishing, p=[0.6, 0.3, 0.1]),
        'Shortining_Service': np.random.choice([1, -1], num_phishing, p=[0.1, 0.9]),
        'having_At_Symbol': np.random.choice([1, -1], num_phishing, p=[0.1, 0.9]),
        'double_slash_redirecting': np.random.choice([1, -1], num_phishing, p=[0.1, 0.9]),
        'Prefix_Suffix': np.random.choice([1, -1], num_phishing, p=[0.2, 0.8]),
        'having_Sub_Domain': np.random.choice([1, 0, -1], num_phishing, p=[0.7, 0.2, 0.1]),
        'SSLfinal_State': np.random.choice([-1, 0, 1], num_phishing, p=[0.1, 0.1, 0.8]),
        'URL_of_Anchor': np.random.choice([-1, 0, 1], num_phishing, p=[0.2, 0.1, 0.7]),
        'Links_in_tags': np.random.choice([-1, 0, 1], num_phishing, p=[0.2, 0.2, 0.6]),
        'SFH': np.random.choice([-1, 0, 1], num_phishing, p=[0.1, 0.1, 0.8]),
        'Abnormal_URL': np.random.choice([1, -1], num_phishing, p=[0.2, 0.8]),
        'topical_keywords': np.random.choice([0, 1], num_phishing, p=[0.9, 0.1])
    }
    
    # Data for benign URLs
    benign_data = {
        'having_IP_Address': np.random.choice([1, -1], num_benign, p=[0.05, 0.95]),
        'URL_Length': np.random.choice([1, 0, -1], num_benign, p=[0.1, 0.6, 0.3]),
        'Shortining_Service': np.random.choice([1, -1], num_benign, p=[0.1, 0.9]),
        'having_At_Symbol': np.random.choice([1, -1], num_benign, p=[0.05, 0.95]),
        'double_slash_redirecting': np.random.choice([1, -1], num_benign, p=[0.05, 0.95]),
        'Prefix_Suffix': np.random.choice([1, -1], num_benign, p=[0.1, 0.9]),
        'having_Sub_Domain': np.random.choice([1, 0, -1], num_benign, p=[0.1, 0.4, 0.5]),
        'SSLfinal_State': np.random.choice([-1, 0, 1], num_benign, p=[0.05, 0.15, 0.8]),
        'URL_of_Anchor': np.random.choice([-1, 0, 1], num_benign, p=[0.1, 0.2, 0.7]),
        'Links_in_tags': np.random.choice([-1, 0, 1], num_benign, p=[0.1, 0.2, 0.7]),
        'SFH': np.random.choice([-1, 0, 1], num_benign, p=[0.1, 0.1, 0.8]),
        'Abnormal_URL': np.random.choice([1, -1], num_benign, p=[0.1, 0.9]),
        'topical_keywords': np.random.choice([0, 1], num_benign, p=[0.95, 0.05])
    }

    # Combine into a single DataFrame
    df_phishing = pd.DataFrame(phishing_data)
    df_benign = pd.DataFrame(benign_data)

    df_phishing['label'] = 1
    df_benign['label'] = 0

    final_df = pd.concat([df_phishing, df_benign], ignore_index=True)
    return final_df.sample(frac=1).reset_index(drop=True)


def train():
    model_path = 'models/phishing_url_detector'
    cluster_model_path = 'models/threat_actor_cluster'
    plot_path = 'models/feature_importance.png'

    # Check if models already exist to prevent retraining
    if os.path.exists(model_path + '.pkl') and os.path.exists(cluster_model_path + '.pkl'):
        print("Models and plot already exist. Skipping training.")
        return

    data = generate_synthetic_data()
    os.makedirs('data', exist_ok=True)
    data.to_csv('data/phishing_synthetic.csv', index=False)

    # --- Train Classification Model ---
    print("Initializing PyCaret Classification Setup...")
    s = setup(data, target='label', session_id=42, verbose=False)

    print("Comparing models...")
    best_model = compare_models(n_select=1, include=['rf', 'et', 'lightgbm'])

    print("Finalizing classification model...")
    final_classification_model = finalize_model(best_model)

    print("Saving feature importance plot...")
    os.makedirs('models', exist_ok=True)
    plot_model(final_classification_model, plot='feature', save=True)
    os.rename('Feature Importance.png', plot_path)

    print("Saving classification model...")
    save_model(final_classification_model, model_path)
    
    # --- Train Clustering Model ---
    print("Initializing PyCaret Clustering Setup...")
    # Use only the features for clustering, and only on malicious data (where label is 1)
    # The actual malicious/benign classification happens with the first model.
    # The clustering model will be used only on malicious URLs.
    malicious_data = data[data['label'] == 1].drop(['label'], axis=1)
    s_cluster = setup_cluster(malicious_data, session_id=42, verbose=False)

    print("Creating K-Means clustering model...")
    kmeans_model = create_model('kmeans', num_clusters=3)

    print("Saving clustering model...")
    save_cluster_model(kmeans_model, cluster_model_path)
    
    print("Models and plot saved successfully.")


if __name__ == "__main__":
    train()
