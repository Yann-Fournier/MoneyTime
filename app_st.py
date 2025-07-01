import streamlit as st
import pandas as pd
import numpy as np
import os
import requests
import datetime
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from keras.utils import get_custom_objects
import keras.losses
import plotly.graph_objects as go

# TA-lib
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import MACD, EMAIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator, AccDistIndexIndicator


get_custom_objects().update({"mse": keras.losses.mean_squared_error})

st.set_page_config(page_title="🔮 Prédiction Crypto - LSTM", layout="wide")
st.title("Prédiction Crypto - LSTM")


crypto_symbol = st.selectbox("Choisir une cryptomonnaie", ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"])
prediction_type = st.selectbox("Choisir l'horizon de prédiction", ["Court Terme", "Moyen Terme", "Long Terme"])
predict_button = st.button("Prédire")


horizons = {
    "Court Terme": {
        "model": "model_Court_Terme_-_1h.h5",
        "interval": "1h",
        "seq_len": 24,
        "suffix": "_CT"
    },
    "Moyen Terme": {
        "model": "model_Moyen_Terme_-_1j.h5",
        "interval": "1d",
        "seq_len": 14,
        "suffix": "_MT"
    },
    "Long Terme": {
        "model": "model_Long_Terme_-_1M.h5",
        "interval": "1d",
        "seq_len": 90,
        "suffix": "_LT"
    }
}



def fetch_binance_data(symbol, interval, limit=300):
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
    response = requests.get(url)
    data = response.json()
    df = pd.DataFrame(data, columns=[
        "date", "open", "high", "low", "close", "volume", "close_time",
        "quote_asset_volume", "number_of_trades", "taker_buy_base_volume",
        "taker_buy_quote_volume", "ignore"
    ])
    df = df[["date", "open", "high", "low", "close", "volume"]]
    df["date"] = pd.to_datetime(df["date"], unit='ms')
    df[["open", "high", "low", "close", "volume"]] = df[["open", "high", "low", "close", "volume"]].astype(float)
    return df



def enrich_features(df):
    df = df.copy()

    # Indicateurs techniques
    df['rsi'] = RSIIndicator(close=df['close']).rsi()
    df['macd'] = MACD(close=df['close']).macd_diff()

    stoch = StochasticOscillator(high=df['high'], low=df['low'], close=df['close'])
    df['stoch_%K'] = stoch.stoch()
    df['stoch_%D'] = stoch.stoch_signal()

    boll = BollingerBands(close=df['close'])
    df['lower_band'] = boll.bollinger_lband()
    df['higher_band'] = boll.bollinger_hband()

    df['ema_10'] = EMAIndicator(close=df['close'], window=10).ema_indicator()
    df['ema_20'] = EMAIndicator(close=df['close'], window=20).ema_indicator()
    df['atr'] = AverageTrueRange(high=df['high'], low=df['low'], close=df['close']).average_true_range()
    df['obv'] = OnBalanceVolumeIndicator(close=df['close'], volume=df['volume']).on_balance_volume()
    df['adi'] = AccDistIndexIndicator(high=df['high'], low=df['low'], close=df['close'],
                                      volume=df['volume']).acc_dist_index()

    df['hl_ratio'] = df['high'] / df['low']
    df['oc_ratio'] = df['open'] / df['close']
    df['vol_change'] = df['volume'].pct_change()
    df['price_change'] = df['close'].pct_change()

    df = df.dropna()

    feature_cols = ['open', 'high', 'low', 'close', 'volume', 'rsi', 'macd',
                    'stoch_%K', 'stoch_%D', 'lower_band', 'higher_band',
                    'ema_10', 'ema_20', 'atr', 'obv', 'adi',
                    'hl_ratio', 'oc_ratio', 'vol_change', 'price_change']

    return df[['date'] + feature_cols]



if predict_button:
    st.info(f"📡 Récupération des données pour {crypto_symbol}...")
    params = horizons[prediction_type]
    df = fetch_binance_data(crypto_symbol, params["interval"], 300)
    df = enrich_features(df)

    if df.shape[0] < params["seq_len"]:
        st.error("Pas assez de données pour générer une prédiction.")
    else:
        # Normalisation et séquencage
        features = df.drop(columns=["date"]).values
        scaler = MinMaxScaler()
        features_scaled = scaler.fit_transform(features)


        def create_sequence(X, seq_len):
            return np.array([X[i - seq_len:i] for i in range(seq_len, len(X))])


        X_seq = create_sequence(features_scaled, params["seq_len"])
        dates = df["date"].values[params["seq_len"]:]

        model_path = os.path.join("utilities", "modeles", params["model"])

        if not os.path.exists(model_path):
            st.error(f"Modèle introuvable : {model_path}")
        else:
            try:
                model = load_model(model_path)
                preds = model.predict(X_seq)

                reg_pred = preds[0][-1][0] if isinstance(preds, list) else preds[-1][0]
                class_pred = preds[1][-1][0] if isinstance(preds, list) and len(preds) > 1 else None

                st.success(f"Prédiction pour {crypto_symbol} ({prediction_type}) terminée.")
                st.metric("🔢 % Évolution prédite", f"{reg_pred:.2f}%")
                if class_pred is not None:
                    tendance = "📉 Baisse probable" if class_pred < 0.5 else "📈 Hausse probable"
                    st.write(f"Prédiction de tendance : **{tendance}**")




                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df["date"], y=df["close"], mode='lines', name='Prix réel'))
                last_price = df["close"].iloc[-1]
                predicted_price = last_price * (1 + reg_pred / 100)
                fig.add_trace(go.Scatter(
                    x=[df["date"].iloc[-1], df["date"].iloc[-1] + pd.Timedelta("1d")],
                    y=[last_price, predicted_price],
                    mode='lines+markers',
                    name='Prédiction',
                    line=dict(dash='dash', color='orange')
                ))
                fig.update_layout(title="Prix réel et prédiction",
                                  xaxis_title="Date", yaxis_title="Prix (USDT)", height=500)
                st.plotly_chart(fig, use_container_width=True)

                st.subheader("Dernières données enrichies utilisées")
                st.dataframe(df.tail(10).style.format("{:.4f}"), use_container_width=True)

                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button("Télécharger les données enrichies (CSV)", csv,
                                   file_name=f"{crypto_symbol}_features.csv",
                                   mime="text/csv")

            except Exception as e:
                st.error(f"Erreur lors de la prédiction : {e}")
