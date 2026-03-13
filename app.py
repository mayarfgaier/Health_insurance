import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

st.set_page_config(page_title="Health-InsurTech", layout="wide")

MODEL_PATH = "insurance_model.pkl"
DATA_PATH = "insurance_data.csv"

SENSITIVE_COLUMNS = [
    "id_client", "nom", "prenom", "date_naissance", "email", "telephone",
    "numero_secu_sociale", "ville", "code_postal", "adresse_ip",
    "date_inscription", "sexe", "region_fr", "mutuelle_complementaire",
    "consentement_rgpd"
]

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    existing = [c for c in SENSITIVE_COLUMNS if c in df.columns]
    df = df.drop(columns=existing, errors="ignore").copy()

    for col in ["sex", "smoker", "region"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.lower()

    return df

def prepare_input_for_plain_model(user_input: pd.DataFrame, reference_df: pd.DataFrame, model):
    ref = clean_dataframe(reference_df)
    if "charges" in ref.columns:
        X_ref = ref.drop(columns=["charges"])
    else:
        X_ref = ref.copy()

    X_ref_encoded = pd.get_dummies(X_ref, drop_first=True)
    user_encoded = pd.get_dummies(user_input, drop_first=True)
    user_aligned = user_encoded.reindex(columns=X_ref_encoded.columns, fill_value=0)

    if hasattr(model, "feature_names_in_"):
        user_aligned = user_aligned.reindex(columns=model.feature_names_in_, fill_value=0)

    return user_aligned

def predict_cost(model, user_input: pd.DataFrame, raw_df: pd.DataFrame):
    try:
        pred = model.predict(user_input)
        return float(pred[0])
    except Exception:
        prepared = prepare_input_for_plain_model(user_input, raw_df, model)
        pred = model.predict(prepared)
        return float(pred[0])

def main():
    st.title("Health-InsurTech : Estimation des frais médicaux")
    st.write("Application de simulation des frais médicaux annuels.")

    model = load_model()
    raw_df = load_data()
    df = clean_dataframe(raw_df)

    col1, col2 = st.columns([1.2, 1])

    with col1:
        st.subheader("Visualisation : âge, IMC et frais médicaux")
        if all(col in df.columns for col in ["age", "bmi", "charges"]):
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.scatter(df["age"], df["charges"], s=df["bmi"] * 3, alpha=0.5)
            ax.set_xlabel("Âge")
            ax.set_ylabel("Frais médicaux")
            ax.set_title("Relation entre l'âge, l'IMC et les frais médicaux")
            st.pyplot(fig)
        else:
            st.warning("Les colonnes nécessaires au graphique ne sont pas disponibles.")

    with col2:
        st.subheader("Simuler vos frais médicaux")

        age = st.slider("Âge", 18, 100, 30)
        bmi = st.slider("IMC (BMI)", 10.0, 60.0, 25.0, step=0.1)
        children = st.slider("Nombre d'enfants", 0, 10, 0)
        sex = st.selectbox("Sexe", ["male", "female"])
        smoker = st.selectbox("Fumeur", ["yes", "no"])
        region = st.selectbox("Région", ["northeast", "northwest", "southeast", "southwest"])

        user_input = pd.DataFrame([{
            "age": age,
            "bmi": bmi,
            "children": children,
            "sex": sex,
            "smoker": smoker,
            "region": region
        }])

        if st.button("Estimer mes frais médicaux"):
            try:
                prediction = predict_cost(model, user_input, raw_df)
                st.success(f"Estimation des frais médicaux annuels : {prediction:,.2f} $".replace(",", " "))
            except Exception as e:
                st.error("Erreur lors de la prédiction.")
                st.exception(e)

    st.markdown("---")
    st.caption("Projet : modèle interprétable, visualisation et simulation en temps réel.")

if __name__ == "__main__":
    main()