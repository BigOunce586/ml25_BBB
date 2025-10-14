import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

train = pd.read_csv("customer_purchase_train_modify.csv")
test = pd.read_csv("customer_purchase_test_modify.csv")

if "label" not in train.columns:
    train["label"] = (train["purchases_in_category"] > 0).astype(int)

X = train.drop(columns=["customer_id", "label", "item_category"], errors="ignore")
y = train["label"]
X_test = test.drop(columns=["customer_id", "item_category"], errors="ignore")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    class_weight="balanced_subsample"
)
model.fit(X_scaled, y)


test["pred"] = model.predict(X_test_scaled)

agg = (
    test.groupby("customer_id")["pred"]
    .agg(["mean", "count"])  
    .reset_index()
)


agg["will_buy"] = (agg["mean"] > 0.25).astype(int)

agg[["customer_id", "will_buy"]].to_csv("predicciones_final.csv", index=False)

