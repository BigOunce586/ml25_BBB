import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

train = pd.read_csv("customer_purchases_train_modify.csv")
test = pd.read_csv("customer_purchases_test_modify.csv")

# Features
feature_cols = [c for c in train.columns if c not in ['customer_id','label']]
X_train = train[feature_cols]
y_train = train['label']

X_test = test[feature_cols]
y_test = test['label'] if 'label' in test.columns else None

# Modelo
logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)

# Resultado
if y_test is not None:
    print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred))
else:
    print("Predicciones generadas para test (sin label)")
