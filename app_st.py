import streamlit as st
import pandas as pd
import numpy as np
import os
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from keras.utils import get_custom_objects

# Résolution du problème lié à 'mse'
import keras.losses
get_custom_objects().update({"mse": keras.losses.mean_squared_error})

st.set_page_config(page_title="🔮 Prédiction Crypto - LSTM", layout="wide")
st.title("Prédiction Crypto - LSTM")
st.markdown("### Charger un fichier CSV de données récentes")

uploaded_file = st.file_uploader("ETH-USDT-USDT.csv", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    if "date" in df.columns:
        df = df.drop(columns=["date"])

    st.success("Données chargées et normalisées avec succès.")

    horizons = {
        "Court_Terme_-_1h": ("_CT", 24),
        "Court_Terme_-_2h": ("_CT", 24),
        "Court_Terme_-_4h": ("_CT", 24),
        "Moyen_Terme_-_12h": ("_MT", 30),
        "Moyen_Terme_-_1j": ("_MT", 14),
        "Moyen_Terme_-_1w": ("_MT", 4),
        "Long_Terme_-_1j": ("_LT", 90),
        "Long_Terme_-_1w": ("_LT", 12),
        "Long_Terme_-_1M": ("_LT", 3)
    }

    for model_name, (target_suffix, seq_len) in horizons.items():
        display_name = model_name.replace("_", " ")
        reg_col = f"Pourc_Price_Evol{target_suffix}"
        class_col = f"Label{target_suffix}"

        model_path = os.path.join("utilities", "modeles", f"model_{model_name}.h5")
        if not os.path.exists(model_path):
            st.warning(f"Modèle manquant pour {display_name}")
            continue

        if reg_col not in df.columns or class_col not in df.columns:
            st.warning(f"Colonnes manquantes pour {display_name} — ignoré.")
            continue

        df_clean = df.copy()
        df_clean = df_clean.drop(columns=[col for col in df.columns if col.endswith(('_CT', '_MT', '_LT')) and col not in [reg_col, class_col]])

        if df_clean.shape[0] < seq_len:
            st.warning(f"Pas assez de données pour {display_name} — requis: {seq_len}")
            continue

        features = df_clean.drop(columns=[reg_col, class_col]).values
        scaler = MinMaxScaler()
        features_scaled = scaler.fit_transform(features)

        def create_sequence(X, seq_len):
            return np.array([X[i - seq_len:i] for i in range(seq_len, len(X))])

        X_seq = create_sequence(features_scaled, seq_len)

        try:
            model = load_model(model_path)
        except Exception as e:
            st.error(f"Erreur de chargement du modèle pour {display_name} : {e}")
            continue

        preds = model.predict(X_seq)
        reg_pred = preds[0][-1][0] if isinstance(preds, list) else preds[-1][0]
        class_pred = preds[1][-1][0] if isinstance(preds, list) else None

        st.subheader(f" Prédiction {display_name}")
        st.metric(" % Évolution", f"{reg_pred:.2f}%")
        if class_pred is not None:
            tendance = "📉 Baisse probable" if class_pred < 0.5 else "📈 Hausse probable"
            st.write(f"Prédiction de tendance : **{tendance}**")
else:
    st.info("Veuillez charger un fichier CSV pour commencer.")