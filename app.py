
import streamlit as st
import pandas as pd
import joblib
import altair as alt
from pathlib import Path

st.set_page_config(
    page_title="Health-InsurTech",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

MODEL_PATH = Path("insurance_model.pkl")
DATA_PATH = Path("insurance_data.csv")

SENSITIVE_COLUMNS = [
    "id_client", "nom", "prenom", "date_naissance", "email", "telephone",
    "numero_secu_sociale", "ville", "code_postal", "adresse_ip",
    "date_inscription", "sexe", "region_fr", "mutuelle_complementaire",
    "consentement_rgpd"
]

st.markdown(
    '''
    <style>
        .block-container {padding-top: 1.5rem; padding-bottom: 1.5rem;}
        .hero-card {
            background: linear-gradient(135deg, #0f172a 0%, #1d4ed8 100%);
            padding: 1.4rem 1.6rem;
            border-radius: 20px;
            color: white;
            margin-bottom: 1rem;
            box-shadow: 0 8px 24px rgba(0,0,0,0.12);
        }
        .soft-card {
            background: #f8fafc;
            padding: 1rem 1.1rem;
            border-radius: 16px;
            border: 1px solid #e2e8f0;
        }
        .metric-label {
            font-size: 0.95rem;
            color: #475569;
            margin-bottom: 0.2rem;
        }
        .metric-value {
            font-size: 1.6rem;
            font-weight: 700;
            color: #0f172a;
        }
    </style>
    ''',
    unsafe_allow_html=True
)

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_raw_data():
    return pd.read_csv(DATA_PATH)

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop(columns=[c for c in SENSITIVE_COLUMNS if c in df.columns], errors="ignore").copy()
    for col in ["sex", "smoker", "region"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.lower()
    return df

def prepare_input_for_plain_model(user_input: pd.DataFrame, reference_df: pd.DataFrame, model):
    ref = clean_dataframe(reference_df)
    X_ref = ref.drop(columns=["charges"]) if "charges" in ref.columns else ref.copy()
    X_ref_encoded = pd.get_dummies(X_ref, drop_first=True)
    user_encoded = pd.get_dummies(user_input, drop_first=True)
    user_aligned = user_encoded.reindex(columns=X_ref_encoded.columns, fill_value=0)
    if hasattr(model, "feature_names_in_"):
        user_aligned = user_aligned.reindex(columns=model.feature_names_in_, fill_value=0)
    return user_aligned

def predict_cost(model, user_input: pd.DataFrame, raw_df: pd.DataFrame) -> float:
    try:
        pred = model.predict(user_input)
        return float(pred[0])
    except Exception:
        prepared = prepare_input_for_plain_model(user_input, raw_df, model)
        pred = model.predict(prepared)
        return float(pred[0])

def rgpd_gate():
    st.info(
        "Notice RGPD : les données saisies sont utilisées uniquement pour la simulation. "
        "Aucune donnée directement identifiante n'est collectée dans cette interface."
    )
    accepted = st.checkbox("J'ai lu la notice et j'accepte le traitement pour la simulation.")
    if not accepted:
        st.stop()

def login_gate():
    with st.sidebar:
        st.markdown("### Accès application")
        username = st.text_input("Nom d'utilisateur")
        password = st.text_input("Mot de passe", type="password")
        login_clicked = st.button("Se connecter", use_container_width=True)
        if login_clicked:
            if username == "admin" and password == "admin123":
                st.session_state["auth_ok"] = True
            else:
                st.session_state["auth_ok"] = False
                st.error("Identifiants incorrects.")
    if not st.session_state.get("auth_ok", False):
        st.warning("Connectez-vous dans la barre latérale pour accéder à l'application.")
        st.stop()

def hero():
    st.markdown(
        '''
        <div class="hero-card">
            <h1 style="margin:0 0 0.3rem 0;">🩺 Health-InsurTech</h1>
            <div style="font-size:1.05rem; opacity:0.95;">
                Tableau de bord interactif et simulateur de frais médicaux annuels,
                avec une approche explicable, éthique et orientée RGPD.
            </div>
        </div>
        ''',
        unsafe_allow_html=True
    )

def build_scatter_chart(df: pd.DataFrame):
    return (
        alt.Chart(df)
        .mark_circle(opacity=0.75)
        .encode(
            x=alt.X("age:Q", title="Âge"),
            y=alt.Y("charges:Q", title="Frais médicaux"),
            size=alt.Size("bmi:Q", title="IMC"),
            color=alt.Color("smoker:N", title="Fumeur"),
            tooltip=["age", "bmi", "children", "sex", "smoker", "region", "charges"]
        )
        .interactive()
    )

def build_region_chart(df: pd.DataFrame):
    region_avg = df.groupby("region", as_index=False)["charges"].mean()
    return (
        alt.Chart(region_avg)
        .mark_bar()
        .encode(
            x=alt.X("region:N", title="Région"),
            y=alt.Y("charges:Q", title="Charges moyennes"),
            tooltip=["region", alt.Tooltip("charges:Q", format=",.2f")]
        )
        .properties(height=320)
    )

def prediction_badge(prediction: float) -> str:
    if prediction < 10000:
        return "Risque de coût faible à modéré"
    if prediction < 25000:
        return "Risque de coût modéré à élevé"
    return "Risque de coût élevé"

def add_prediction_to_history(user_input: pd.DataFrame, prediction: float):
    row = user_input.copy()
    row["predicted_charges"] = prediction
    if "history_df" not in st.session_state:
        st.session_state["history_df"] = row
    else:
        st.session_state["history_df"] = pd.concat([st.session_state["history_df"], row], ignore_index=True)

def sidebar_filters(df: pd.DataFrame):
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Filtres dashboard")
    smoker_options = sorted(df["smoker"].dropna().unique().tolist()) if "smoker" in df.columns else []
    region_options = sorted(df["region"].dropna().unique().tolist()) if "region" in df.columns else []
    selected_smoker = st.sidebar.multiselect("Statut fumeur", smoker_options, default=smoker_options)
    selected_regions = st.sidebar.multiselect("Régions", region_options, default=region_options)
    filtered = df.copy()
    if smoker_options:
        filtered = filtered[filtered["smoker"].isin(selected_smoker)]
    if region_options:
        filtered = filtered[filtered["region"].isin(selected_regions)]
    return filtered

def show_metrics(df: pd.DataFrame):
    c1, c2, c3, c4 = st.columns(4)
    metrics = [
        ("Nombre de profils", f"{len(df):,}".replace(",", " ")),
        ("Âge moyen", f"{df['age'].mean():.1f}" if "age" in df.columns else "-"),
        ("BMI moyen", f"{df['bmi'].mean():.1f}" if "bmi" in df.columns else "-"),
        ("Charges moyennes", f"{df['charges'].mean():,.0f} $".replace(",", " ") if "charges" in df.columns else "-"),
    ]
    for col, (label, value) in zip([c1, c2, c3, c4], metrics):
        with col:
            st.markdown(
                f'<div class="soft-card"><div class="metric-label">{label}</div><div class="metric-value">{value}</div></div>',
                unsafe_allow_html=True
            )

def show_dataset_preview(df: pd.DataFrame):
    st.dataframe(df.head(20), use_container_width=True, hide_index=True)

def simulation_panel(model, raw_df):
    st.markdown("### Simulateur personnalisé")
    with st.form("simulation_form", clear_on_submit=False):
        col1, col2 = st.columns(2)
        with col1:
            age = st.slider("Âge", 18, 100, 30)
            bmi = st.slider("IMC", 10.0, 60.0, 25.0, step=0.1)
            children = st.slider("Nombre d'enfants", 0, 10, 0)
        with col2:
            sex = st.selectbox("Sexe", ["female", "male"])
            smoker = st.selectbox("Fumeur", ["no", "yes"])
            region = st.selectbox("Région", ["northeast", "northwest", "southeast", "southwest"])
        submitted = st.form_submit_button("Estimer mes frais médicaux", use_container_width=True)

    if submitted:
        user_input = pd.DataFrame([{
            "age": age,
            "bmi": bmi,
            "children": children,
            "sex": sex,
            "smoker": smoker,
            "region": region
        }])
        prediction = predict_cost(model, user_input, raw_df)
        add_prediction_to_history(user_input, prediction)
        left, right = st.columns([1.1, 1])
        with left:
            st.success(f"Estimation annuelle : {prediction:,.2f} $".replace(",", " "))
            st.markdown(
                f'<div class="soft-card"><div class="metric-label">Niveau estimé</div><div class="metric-value" style="font-size:1.15rem;">{prediction_badge(prediction)}</div></div>',
                unsafe_allow_html=True
            )
        with right:
            st.markdown("#### Profil saisi")
            st.write({
                "age": age,
                "bmi": bmi,
                "children": children,
                "sex": sex,
                "smoker": smoker,
                "region": region,
            })
        st.caption(
            "Interprétation simple : le coût estimé dépend principalement de l'âge, de l'IMC, "
            "du nombre d'enfants, du statut fumeur et de la région."
        )

def history_panel():
    st.markdown("### Historique des simulations")
    hist = st.session_state.get("history_df")
    if hist is None or hist.empty:
        st.info("Aucune simulation enregistrée pour le moment.")
        return
    st.dataframe(hist, use_container_width=True, hide_index=True)
    csv_data = hist.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Télécharger l'historique en CSV",
        data=csv_data,
        file_name="simulation_history.csv",
        mime="text/csv",
        use_container_width=True
    )

def recommendation_panel(df: pd.DataFrame):
    st.markdown("### Observations utiles")
    tips = []
    if "smoker" in df.columns and "charges" in df.columns:
        avg = df.groupby("smoker")["charges"].mean().to_dict()
        if "yes" in avg and "no" in avg:
            tips.append(
                f"Les charges moyennes des fumeurs ({avg['yes']:,.0f} $) sont plus élevées que celles des non-fumeurs ({avg['no']:,.0f} $).".replace(",", " ")
            )
    if "bmi" in df.columns and "charges" in df.columns:
        tips.append(f"La corrélation BMI / charges est de {df['bmi'].corr(df['charges']):.2f}.")
    if "age" in df.columns and "charges" in df.columns:
        tips.append(f"La corrélation âge / charges est de {df['age'].corr(df['charges']):.2f}.")
    for tip in tips:
        st.markdown(f"- {tip}")

def main():
    if not MODEL_PATH.exists():
        st.error("Le fichier insurance_model.pkl est introuvable.")
        st.stop()
    if not DATA_PATH.exists():
        st.error("Le fichier insurance_data.csv est introuvable.")
        st.stop()

    hero()
    rgpd_gate()
    login_gate()

    model = load_model()
    raw_df = load_raw_data()
    df = clean_dataframe(raw_df)
    filtered_df = sidebar_filters(df)

    show_metrics(filtered_df)

    tab1, tab2, tab3, tab4 = st.tabs([
        "Dashboard interactif",
        "Simulation",
        "Historique",
        "Données & insights"
    ])

    with tab1:
        st.markdown("### Corrélation entre âge, IMC et frais médicaux")
        if all(col in filtered_df.columns for col in ["age", "bmi", "charges", "smoker"]):
            st.altair_chart(build_scatter_chart(filtered_df), use_container_width=True)
        else:
            st.warning("Certaines colonnes nécessaires au graphique sont absentes.")
        st.markdown("### Charges moyennes par région")
        if all(col in filtered_df.columns for col in ["region", "charges"]):
            st.altair_chart(build_region_chart(filtered_df), use_container_width=True)

    with tab2:
        simulation_panel(model, raw_df)

    with tab3:
        history_panel()

    with tab4:
        col_left, col_right = st.columns([1.25, 1])
        with col_left:
            st.markdown("### Aperçu des données")
            show_dataset_preview(filtered_df)
        with col_right:
            recommendation_panel(filtered_df)
            st.markdown("### Variables utilisées")
            st.write(["age", "bmi", "children", "sex", "smoker", "region"])

    st.markdown("---")
    st.caption(
        "Application Streamlit améliorée : dashboard interactif, simulateur, historique, "
        "notice RGPD, authentification simple et export CSV."
    )

if __name__ == "__main__":
    main()
