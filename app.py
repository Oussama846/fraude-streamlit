import streamlit as st
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

st.title("Détection de Fraudes Bancaires")

@st.cache_data
def load_data():
    df = pd.read_csv("data/creditcardfraud.csv")
    return df

df = load_data()
st.write("Aperçu des données :", df.head())

X = df.drop(columns=["Time", "Class"])
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = IsolationForest(contamination=0.001, random_state=42)
df["anomaly"] = model.fit_predict(X_scaled)
df["anomaly"] = df["anomaly"].map({1: 0, -1: 1})
df["risk_score"] = model.decision_function(X_scaled) * -1

fraudes = df[df["anomaly"] == 1]
st.metric("Nombre de fraudes détectées", len(fraudes))

st.subheader("Transactions frauduleuses")
st.dataframe(fraudes[["Amount", "risk_score"]].sort_values(by="risk_score", ascending=False))
