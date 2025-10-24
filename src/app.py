import streamlit as st
import pandas as pd
import joblib
from rdkit import Chem
from rdkit.Chem import Descriptors
import requests
from dotenv import load_dotenv
import os

# -----------------------------
# Load environment variables
# -----------------------------
load_dotenv()
API_KEY = os.getenv("NVIDIA_API_KEY")

# -----------------------------
# Load trained model
# -----------------------------
model = joblib.load('./models/esol_rf_model.pkl')
features = ['MolWt', 'NumHDonors', 'NumHAcceptors', 'NumRings', 'TPSA', 'NumRotatableBonds']

# -----------------------------
# RDKit descriptor function
# -----------------------------
def compute_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        return pd.Series({
            'MolWt': Descriptors.MolWt(mol),
            'NumHDonors': Descriptors.NumHDonors(mol),
            'NumHAcceptors': Descriptors.NumHAcceptors(mol),
            'NumRings': Descriptors.RingCount(mol),
            'TPSA': Descriptors.TPSA(mol),
            'NumRotatableBonds': Descriptors.NumRotatableBonds(mol)
        })
    else:
        return pd.Series([None]*6, index=features)

# -----------------------------
# NVIDIA LLM Function
# -----------------------------
def get_llm_insights(compound_name):
    if not API_KEY:
        return "NVIDIA API key not found in environment variables."
    
    invoke_url = "https://integrate.api.nvidia.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Accept": "application/json"
    }
    payload = {
        "model": "meta/llama-4-maverick-17b-128e-instruct",
        "messages": [
            {"role": "user", 
             "content": f"Provide a short description of the compound {compound_name} (1–2 lines) "
                        "and then explain its solubility and chemical properties in simple bullet points, "
                        "easy for a non-chemist to understand. Keep it concise and practical for researchers."}
        ],
        "max_tokens": 512,
        "temperature": 1.0,
        "top_p": 1.0,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
        "stream": False
    }

    response = requests.post(invoke_url, headers=headers, json=payload)
    if response.status_code == 200:
        result = response.json()
        try:
            return result['choices'][0]['message']['content']
        except:
            return "LLM returned unexpected response format."
    else:
        return f"Error calling LLM API: {response.status_code}"

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="ESOL Solubility Dashboard", layout="wide")
st.title("🔬 ESOL Solubility Prediction Dashboard")
st.markdown("""
Predict solubility of organic compounds and get detailed chemical insights using NVIDIA LLM.  
Supports batch processing and CSV download.
""")

# -----------------------------
# Input: Single or batch
# -----------------------------
st.sidebar.header("Input Options")
input_mode = st.sidebar.radio("Choose input type:", ["Single Compound", "Batch Upload (CSV)"])

if input_mode == "Single Compound":
    smiles_input = st.sidebar.text_input("Enter SMILES string:")
    compound_name_input = st.sidebar.text_input("Compound name (optional for LLM insights)")
    if st.sidebar.button("Predict & Get Insights"):
        if not smiles_input:
            st.sidebar.warning("Please enter a SMILES string.")
        else:
            descriptors = compute_descriptors(smiles_input)
            if descriptors.isnull().any():
                st.error("Invalid SMILES string.")
            else:
                pred = model.predict([descriptors[features]])[0]
                st.success(f"Predicted log solubility: **{pred:.3f} mol/L**")
                st.info("**Molecular Descriptors Used:**")
                st.dataframe(pd.DataFrame([descriptors]))
                
                if compound_name_input:
                    with st.spinner("Fetching chemical insights from NVIDIA LLM..."):
                        llm_output = get_llm_insights(compound_name_input)
                        st.subheader("🧪 NVIDIA LLM Insights")
                        st.markdown(llm_output.replace("\n", "  \n"))

else:  # Batch CSV
    st.sidebar.markdown("Upload CSV with columns: `SMILES`, `CompoundName` (optional)")
    uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        # Robust header normalization: strip whitespace, uppercase, remove spaces
        raw_cols = list(df.columns)
        def _norm(c):
            return str(c).strip().upper().replace(" ", "")
        # Map normalized -> original (if you later need original names)
        norm_map = { _norm(c): c for c in raw_cols }
        df.columns = [ _norm(c) for c in raw_cols ]

        if "SMILES" not in df.columns:
            st.error("CSV must contain 'SMILES' column.")
        else:
            st.info("Processing batch predictions...")
            results = []

            # Find sensible columns for compound name: look for headers containing one of these keywords
            name_keywords = ['NAME', 'COMPOUND', 'TITLE', 'ID']
            name_cols = [c for c in df.columns if any(k in c for k in name_keywords)]
            name_col = name_cols[0] if name_cols else None

            for idx, row in df.iterrows():
                smiles = row["SMILES"]
                # Get cleaned compound name if present. Try multiple candidate columns and pick first non-empty.
                name = ""
                if name_cols:
                    for nc in name_cols:
                        raw_name = row.get(nc, "")
                        if pd.isna(raw_name):
                            continue
                        candidate = str(raw_name).strip()
                        if candidate:
                            name = candidate
                            break

                desc = compute_descriptors(smiles)
                if desc.isnull().any():
                    pred = None
                else:
                    pred = model.predict([desc[features]])[0]
                llm_text = ""
                if name and API_KEY:
                    llm_text = get_llm_insights(name)
                results.append({
                    "SMILES": smiles,
                    "CompoundName": name,
                    "PredictedSolubility": pred,
                    "LLMInsights": llm_text
                })
            results_df = pd.DataFrame(results)
            # Show how many compound names were available vs missing to help debugging
            num_with_name = int(results_df['CompoundName'].astype(bool).sum()) if 'CompoundName' in results_df.columns else 0
            st.info(f"Compound names found: {num_with_name} / {len(results_df)} rows")
            
            # Color-coded solubility
            def color_solubility(val):
                if val is None:
                    return 'background-color: gray'
                elif val < -2:
                    return 'background-color: lightcoral'
                elif val < 0:
                    return 'background-color: lightyellow'
                else:
                    return 'background-color: lightgreen'
            
            st.subheader("Batch Predictions")
            st.dataframe(results_df.style.applymap(color_solubility, subset=['PredictedSolubility']))
            
            # Download CSV
            csv = results_df.to_csv(index=False).encode()
            st.download_button(
                label="📥 Download Results as CSV",
                data=csv,
                file_name='esol_predictions.csv',
                mime='text/csv'
            )
