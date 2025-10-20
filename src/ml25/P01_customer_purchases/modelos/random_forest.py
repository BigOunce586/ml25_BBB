import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import numpy as np

train_df = pd.read_csv("train_final.csv")
test_df = pd.read_csv("test_final.csv")

y = train_df["label"]
X = train_df.drop(columns=["label"])

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    min_samples_split=5,
    min_samples_leaf=3,
    class_weight="balanced_subsample",
    random_state=42,
    n_jobs=-1
)
model.fit(X, y)
joblib.dump(model, "purchase_model.pkl")

y_pred_train = model.predict(X)
acc = accuracy_score(y, y_pred_train)
print("Entrenamiento Accuracy:", round(acc, 4))
print("Matriz de confusión:\n", confusion_matrix(y, y_pred_train))
print("Reporte de clasificación:\n", classification_report(y, y_pred_train))

preds = model.predict(test_df)

submission = pd.DataFrame({
    "ID": range(len(test_df)),
    "pred": preds
})
submission.to_csv("predictions.csv", index=False)
