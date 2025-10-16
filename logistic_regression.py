import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import joblib
import matplotlib.pyplot as plt

train = pd.read_csv("customer_purchase_train_modify.csv")
test = pd.read_csv("customer_purchase_test_modify.csv")

train['label'] = train['purchases_in_category'].apply(lambda x: 1 if x > 0 else 0)

cols_to_drop = [
    'purchases_in_category',
    'total_spent_in_category',
    'avg_spent_in_category'
]

feature_cols = [c for c in train.columns if c not in ['customer_id', 'label'] + cols_to_drop]
X = train[feature_cols]
y = train['label']

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

test_ids = test["customer_id"].copy()
missing_cols = [c for c in feature_cols if c not in test.columns]
for col in missing_cols:
    test[col] = 0
test_features = test[feature_cols]
test_scaled = scaler.transform(test_features)

model = LogisticRegression(
    max_iter=2000,
    solver='lbfgs',
    class_weight='balanced',
    random_state=42
)
model.fit(X_train_scaled, y_train)

preds_val = model.predict(X_val_scaled)
acc = accuracy_score(y_val, preds_val)
f1 = f1_score(y_val, preds_val)
cm = confusion_matrix(y_val, preds_val)

print("\n Evaluación en validación simple:")
print(f"Accuracy: {acc:.4f}")
print(f"F1-score: {f1:.4f}")
print("Matriz de confusión:\n", cm)

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
X_scaled_full = scaler.fit_transform(X)
acc_scores = cross_val_score(model, X_scaled_full, y, cv=kfold, scoring='accuracy')
f1_scores = cross_val_score(model, X_scaled_full, y, cv=kfold, scoring='f1')

print("\n Validación cruzada (5 folds):")
print(f"Accuracy promedio: {acc_scores.mean():.4f} ± {acc_scores.std():.4f}")
print(f"F1 promedio:       {f1_scores.mean():.4f} ± {f1_scores.std():.4f}")

model.fit(X_scaled_full, y)
coef_importance = pd.DataFrame({
    "feature": feature_cols,
    "importance": np.abs(model.coef_[0])
}).sort_values(by="importance", ascending=False)

print("\n Principales variables que influyen en la predicción:")
print(coef_importance.head(15))

plt.figure(figsize=(10,6))
top_features = coef_importance.head(15)
plt.barh(top_features["feature"], top_features["importance"], color="steelblue")
plt.xlabel("Importancia (|coef|)")
plt.ylabel("Variable")
plt.title("Importancia de variables - Regresión Logística")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

preds_test = model.predict(test_scaled)

pred_df = pd.DataFrame({
    "customer_id": test_ids,
    "prediction": preds_test
})
pred_df.to_csv("predicciones_logistic_final.csv", index=False)
joblib.dump(model, "best_model_logistic_final.pkl")
joblib.dump(scaler, "scaler_logistic_final.pkl")

print(f"Filas del test: {pred_df.shape[0]}")
print(pred_df.head())
