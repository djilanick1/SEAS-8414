# app.py
import streamlit as st
import pandas as pd
from pycaret.classification import load_model, predict_model
from pycaret.clustering import load_model as load_cluster_model, predict_model as predict_cluster_model
from updated_genai_prescriptions import generate_prescription
import os
import time


# --- Page Configuration ---
st.set_page_config(
    page_title="Homework8 GenAI-Powered Phishing SOAR",
    page_icon="",
    layout="wide"
)

# --- Threat Profile Descriptions ---
threat_profiles = {
    'Cluster 0': {'name': 'State-Sponsored', 'description': 'Highly sophisticated and targeted attacks, often using advanced techniques and aiming for long-term presence. Characterized by long URLs, trusted SSL certificates, and a focus on sub-domains.'},
    'Cluster 1': {'name': 'Organized Cybercrime', 'description': 'High-volume, financially motivated attacks. These campaigns often use disposable infrastructure like URL shortening services, IP addresses, and abnormal URL structures.'},
    'Cluster 2': {'name': 'Hacktivist', 'description': 'Attacks driven by social or political motives. These actors may use a mix of tactics and often include topical keywords in their URLs to draw attention.'}
}


# --- Load Models and Feature Plot ---
@st.cache_resource
def load_assets():
    model_path = 'models/phishing_url_detector'
    cluster_model_path = 'models/threat_actor_cluster'
    plot_path = 'models/feature_importance.png'

    model = None
    cluster_model = None
    plot = None

    if os.path.exists(model_path + '.pkl'):
        model = load_model(model_path)
    if os.path.exists(cluster_model_path + '.pkl'):
        cluster_model = load_cluster_model(cluster_model_path)
    if os.path.exists(plot_path):
        plot = plot_path
    return model, cluster_model, plot


model, cluster_model, feature_plot = load_assets()

if not model or not cluster_model:
    st.error(
        "Models not found. Please run the training script or wait for the initial training to complete. Check the container logs with `make logs` if the error persists.")
    st.stop()

# --- Sidebar for Inputs ---
with st.sidebar:
    st.title("URL Feature Input")
    st.write("Describe the characteristics of a suspicious URL below.")

    # Using a dictionary to hold form values
    form_values = {
        'url_length': st.select_slider("URL Length", options=['Short', 'Normal', 'Long'], value='Long'),
        'ssl_state': st.select_slider("SSL Certificate Status", options=['Trusted', 'Suspicious', 'None'],
                                      value='Suspicious'),
        'sub_domain': st.select_slider("Sub-domain Complexity", options=['None', 'One', 'Many'], value='One'),
        'prefix_suffix': st.checkbox("URL has a Prefix/Suffix (e.g.,'-')", value=True),
        'has_ip': st.checkbox("URL uses an IP Address", value=False),
        'short_service': st.checkbox("Is it a shortened URL", value=False),
        'at_symbol': st.checkbox("URL contains '@' symbol", value=False),
        'abnormal_url': st.checkbox("Is it an abnormal URL", value=True),
        'topical_keywords': st.checkbox("Contains topical/political keywords", value=False),
    }

    st.divider()
    genai_provider = st.selectbox("Select GenAI Provider", ["Gemini", "OpenAI", "Grok"])
    submitted = st.button("Analyze & Initiate Response", use_container_width=True, type="primary")

# --- Main Page ---
st.title("Homework8 GenAI-Powered SOAR for Phishing URL Analysis")

if not submitted:
    st.info("Please provide the URL features in the sidebar and click 'Analyze' to begin.")
    if feature_plot:
        st.subheader("Model Feature Importance")
        st.image(feature_plot,
                 caption="Feature importance from the trained RandomForest model. This shows which features the model weighs most heavily when making a prediction.")

else:
    # --- Data Preparation and Risk Scoring ---
    input_dict = {
        'having_IP_Address': 1 if form_values['has_ip'] else -1,
        'URL_Length': -1 if form_values['url_length'] == 'Short' else (
            0 if form_values['url_length'] == 'Normal' else 1),
        'Shortining_Service': 1 if form_values['short_service'] else -1,
        'having_At_Symbol': 1 if form_values['at_symbol'] else -1,
        'double_slash_redirecting': -1,
        'Prefix_Suffix': 1 if form_values['prefix_suffix'] else -1,
        'having_Sub_Domain': -1 if form_values['sub_domain'] == 'None' else (
            0 if form_values['sub_domain'] == 'One' else 1),
        'SSLfinal_State': -1 if form_values['ssl_state'] == 'None' else (
            0 if form_values['ssl_state'] == 'Suspicious' else 1),
        'Abnormal_URL': 1 if form_values['abnormal_url'] else -1,
        'URL_of_Anchor': 0, 'Links_in_tags': 0, 'SFH': 0,
        'topical_keywords': 1 if form_values['topical_keywords'] else 0,
    }
    input_data = pd.DataFrame([input_dict])

    # Simple risk contribution for visualization
    risk_scores = {
        "Bad SSL": 25 if input_dict['SSLfinal_State'] < 1 else 0,
        "Abnormal URL": 20 if input_dict['Abnormal_URL'] == 1 else 0,
        "Prefix/Suffix": 15 if input_dict['Prefix_Suffix'] == 1 else 0,
        "Shortened URL": 15 if input_dict['Shortining_Service'] == 1 else 0,
        "Complex Sub-domain": 10 if input_dict['having_Sub_Domain'] == 1 else 0,
        "Long URL": 10 if input_dict['URL_Length'] == 1 else 0,
        "Uses IP Address": 5 if input_dict['having_IP_Address'] == 1 else 0,
        "Topical Keywords": 10 if input_dict['topical_keywords'] == 1 else 0,
    }
    risk_df = pd.DataFrame(list(risk_scores.items()), columns=['Feature', 'Risk Contribution']).sort_values(
        'Risk Contribution', ascending=False)

    # --- Analysis Workflow ---
    with st.status("Executing SOAR playbook...", expanded=True) as status:
        st.write(" **Step 1: Predictive Analysis** - Running features through classification model.")
        time.sleep(1)
        prediction = predict_model(model, data=input_data)
        is_malicious = prediction['prediction_label'].iloc[0] == 1

        verdict = "MALICIOUS" if is_malicious else "BENIGN"
        st.write(f" **Step 2: Verdict Interpretation** - Model predicts **{verdict}**.")
        time.sleep(1)
        
        threat_profile_prediction = "None"
        if is_malicious:
            st.write(f" **Step 3: Threat Attribution** - Running features through clustering model.")
            time.sleep(1)
            # Make sure to drop the 'label' column if it exists and predict
            cluster_prediction = predict_cluster_model(cluster_model, data=input_data.drop(['double_slash_redirecting', 'URL_of_Anchor', 'Links_in_tags', 'SFH'], axis=1))
            threat_profile_prediction = f"Cluster {cluster_prediction['Cluster'].iloc[0]}"
            st.write(f" **Step 4: Prescriptive Analytics** - Engaging **{genai_provider}** for action plan.")
            try:
                # Add the threat profile to the details sent to the GenAI model
                alert_details = {**input_dict, 'threat_profile': threat_profiles[threat_profile_prediction]['name']}
                prescription = generate_prescription(genai_provider, alert_details)
                status.update(label=" SOAR Playbook Executed Successfully!", state="complete", expanded=False)
            except Exception as e:
                st.error(f"Failed to generate prescription: {e}")
                prescription = None
                status.update(label=" Error during GenAI prescription!", state="error")
        else:
            prescription = None
            status.update(label=" Analysis Complete. No threat found.", state="complete", expanded=False)

    # --- Tabs for Organized Output ---
    tab1, tab2, tab3, tab4 = st.tabs([" **Analysis Summary**", " **Visual Insights**", " **Prescriptive Plan**", " **Threat Attribution**"])

    with tab1:
        st.subheader("Verdict and Key Findings")
        if is_malicious:
            st.error("**Prediction: Malicious Phishing URL**", icon="")
        else:
            st.success("**Prediction: Benign URL**", icon="")

        st.metric("Malicious Confidence Score",
                  f"{prediction['prediction_score'].iloc[0]:.2%}" if is_malicious else f"{1 - prediction['prediction_score'].iloc[0]:.2%}")
        st.caption("This score represents the model's confidence in its prediction.")

    with tab2:
        st.subheader("Visual Analysis")
        st.write("#### Risk Contribution by Feature")
        st.bar_chart(risk_df.set_index('Feature'))
        st.caption("A simplified view of which input features contributed most to a higher risk score.")

        if feature_plot:
            st.write("#### Model Feature Importance (Global)")
            st.image(feature_plot,
                     caption="This plot shows which features the model found most important *overall* during its training.")

    with tab3:
        st.subheader("Actionable Response Plan")
        if prescription:
            st.success("A prescriptive response plan has been generated by the AI.", icon="")
            st.json(prescription, expanded=False)  # Show the raw JSON for transparency

            st.write("#### Recommended Actions (for Security Analyst)")
            for i, action in enumerate(prescription.get("recommended_actions", []), 1):
                st.markdown(f"**{i}.** {action}")

            st.write("#### Communication Draft (for End-User/Reporter)")
            st.text_area("Draft", prescription.get("communication_draft", ""), height=150)
        else:
            st.info("No prescriptive plan was generated because the URL was classified as benign.")

    with tab4:
        st.subheader("Predicted Threat Actor Profile")
        if is_malicious:
            profile = threat_profiles.get(threat_profile_prediction)
            if profile:
                st.info(f"The URL is likely associated with the **{profile['name']}** threat profile.", icon="")
                st.markdown(profile['description'])
            else:
                st.warning("Could not determine a threat actor profile.", icon="")
        else:
            st.info("Threat attribution is only performed on malicious URLs.")
