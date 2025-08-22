import h2o
import os
import asyncio
import argparse
import tldextract
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from datetime import datetime

# Define global variables for models and tokenizer
generic_model = None
prescriptive_model = None
tokenizer = None

def connect_h2o():
    """
    Initializes and connects to a local H2O instance.
    This function handles the necessary JVM argument for MOJO model loading.
    """
    print("Checking whether there is an H2O instance running at http://localhost:54321.....", end=" ")
    try:
        # Use the environment variable to set the JVM argument.
        # This workaround bypasses the h2o.init() argument parsing issue.
        os.environ['H2O_JVM_ARGS'] = "-Dsys.ai.h2o.pojo.import.enabled=true"
        h2o.init()
        print("successful.")
    except h2o.exceptions.H2OConnectionError:
        print("not found.")
        print("Attempting to start a local H2O server...")
        # If not found, attempt to start a new local H2O instance
        h2o.init()
        print("Server is running at", h2o.connection().url)
    except Exception as e:
        print(f"Error connecting to H2O: {e}")
        exit(1)


def load_dga_models():
    """
    Loads the DGA detection models (generic and prescriptive) and the tokenizer.
    """
    global generic_model, prescriptive_model, tokenizer
    try:
        print("Loading DGA detection models and tokenizer...")
        generic_model = load_model("dga_generic_model.h5")
        
        # Using h2o.import_mojo() which is more robust for loading MOJO files from disk.
        prescriptive_model = h2o.import_mojo("dga_prescriptive_model.mojo")

        # Load the tokenizer
        # The 'character_tokenizer' function has been removed. Use Tokenizer with char_level=True.
        # This example assumes the original tokenizer was a simple character-to-index mapping.
        # If your model was trained with a Keras Tokenizer, you should load it from a JSON file.
        import json
        with open('char_tokenizer.json', 'r') as f:
            tokenizer_json = f.read()
            tokenizer = Tokenizer(char_level=True)
            tokenizer.word_index = json.loads(tokenizer_json)['word_index']
            
        print("Models and tokenizer loaded successfully.")
    except Exception as e:
        print(f"Error loading models or tokenizer: {e}")
        exit(1)

def featurize_domain(domain):
    """
    Extracts features from a domain for DGA detection.
    Args:
        domain (str): The domain name to featurize.
    Returns:
        dict: A dictionary of extracted features.
    """
    ext = tldextract.extract(domain)
    subdomain = ext.subdomain
    domain_name = ext.domain
    suffix = ext.suffix

    features = {
        'length': len(domain_name),
        'vowel_ratio': sum(1 for char in domain_name if char in 'aeiou') / len(domain_name) if len(domain_name) > 0 else 0,
        'consonant_ratio': sum(1 for char in domain_name if char not in 'aeiou' and char.isalpha()) / len(domain_name) if len(domain_name) > 0 else 0,
        'digit_count': sum(1 for char in domain_name if char.isdigit()),
        'entropy': calculate_entropy(domain_name),
        'has_hyphen': 1 if '-' in domain_name else 0,
        'num_subdomains': len(subdomain.split('.')) if subdomain else 0,
        'domain_name_str': domain_name # Keep original string for character-level model
    }
    return features

def calculate_entropy(s):
    """
    Calculates the Shannon entropy of a string.
    """
    if not s:
        return 0
    freq = {}
    for char in s:
        freq[char] = freq.get(char, 0) + 1
    entropy = 0.0
    total = len(s)
    for char in freq:
        prob = float(freq[char]) / total
        entropy -= prob * np.log2(prob)
    return entropy

def prepare_for_generic_model(domain_str, tokenizer, max_len=50):
    """
    Prepares a domain string for the generic (Keras LSTM) model.
    """
    # Convert domain string to a sequence of character indices using the new tokenizer
    tokenized_sequence = tokenizer.texts_to_sequences([domain_str])[0]
    
    # Pad the sequence to the fixed length
    padded_sequence = pad_sequences([tokenized_sequence], maxlen=max_len, padding='post')[0]
    return np.array([padded_sequence])

def analyze_domain(domain):
    """
    Analyzes a given domain using both generic and prescriptive DGA models.
    """
    global generic_model, prescriptive_model, tokenizer

    if generic_model is None or prescriptive_model is None or tokenizer is None:
        load_dga_models()

    print(f"Analyzing domain: {domain}")

    # Featurize the domain
    features = featurize_domain(domain)
    
    # Prepare for generic model (character-level LSTM)
    domain_sequence = prepare_for_generic_model(features['domain_name_str'], tokenizer)
    generic_prediction = generic_model.predict(domain_sequence)[0][0]

    print(f"\n--- Generic DGA Detector ---")
    print(f"Generic DGA Score: {generic_prediction:.4f}")
    if generic_prediction > 0.5: # Threshold can be tuned
        print("Verdict: Likely DGA (Generic)")
    else:
        print("Verdict: Legitimate (Generic)")

    # Prepare for prescriptive model (H2O MOJO model)
    # The MOJO model expects an H2OFrame as input
    prescriptive_data = {
        'length': [features['length']],
        'vowel_ratio': [features['vowel_ratio']],
        'consonant_ratio': [features['consonant_ratio']],
        'digit_count': [features['digit_count']],
        'entropy': [features['entropy']],
        'has_hyphen': [features['has_hyphen']],
        'num_subdomains': [features['num_subdomains']]
    }
    prescriptive_frame = h2o.H2OFrame(prescriptive_data)
    
    # Make prediction with prescriptive model
    prescriptive_prediction_frame = prescriptive_model.predict(prescriptive_frame)
    # Get the probability of being DGA (assuming column 2 is p0, column 3 is p1)
    prescriptive_prediction = prescriptive_prediction_frame['p1'][0,0] # Assuming p1 is for DGA

    print(f"\n--- Prescriptive DGA Detector ---")
    print(f"Prescriptive DGA Score: {prescriptive_prediction:.4f}")
    if prescriptive_prediction > 0.5: # Threshold can be tuned
        print("Verdict: Likely DGA (Prescriptive)")
    else:
        print("Verdict: Legitimate (Prescriptive)")

    # Provide a combined verdict
    print("\n--- Combined Verdict ---")
    if generic_prediction > 0.5 or prescriptive_prediction > 0.5:
        print("Overall Verdict: **Likely DGA** (Further investigation recommended)")
    else:
        print("Overall Verdict: **Legitimate**")

    # Disconnect H2O
    h2o.cluster().shutdown(prompt=False)


async def main(domain):
    """
    Main function to analyze a domain.
    """
    # 1. Connect to H2O instance
    connect_h2o()

    # 2. Analyze the domain
    analyze_domain(domain)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Analyze a domain for DGA characteristics.")
    parser.add_argument("--domain", type=str, required=True, help="The domain to analyze.")
    args = parser.parse_args()
    
    # The correct way to run the main function
    asyncio.run(main(args.domain))
