import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# --- Cargar datasets ---
train = pd.read_csv("customer_purchase_train_modify.csv")
test = pd.read_csv("customer_purchase_test_modify.csv")


# --- Separar features y label ---
X = train.drop(columns=["customer_id", "label"])
y = train["label"]
X_test = test.drop(columns=["customer_id"])

# --- Escalar datos ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

# --- Entrenar modelo ---
logreg = LogisticRegression(max_iter=2000, class_weight='balanced')
logreg.fit(X_scaled, y)

# --- Predecir sobre test ---
test["will_buy_pred"] = logreg.predict(X_test_scaled)

# --- Agrupar por cliente ---
agg_preds = (
    test.groupby("customer_id")["will_buy_pred"]
    .max()  # si algún producto predicho es 1, se considera comprador
    .reset_index()
    .rename(columns={"will_buy_pred": "pred"})
)
