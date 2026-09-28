import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier

dados = joblib.load("dados_modelagem.pkl")
X_train, X_test = dados["X_train"], dados["X_test"]
y_train, y_test = dados["y_train"], dados["y_test"]

modelos = {
    "Regressão Logística": LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(class_weight="balanced", random_state=42),
    "XGBoost": XGBClassifier(
        scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
        random_state=42,
        eval_metric="logloss",
    ),
}

print("Comparação via validação cruzada (5 folds, métrica: ROC-AUC)\n")
resultados = {}
for nome, modelo in modelos.items():
    scores = cross_val_score(modelo, X_train, y_train, cv=5, scoring="roc_auc")
    resultados[nome] = scores
    print(f"{nome}: {scores.mean():.4f} (+/- {scores.std():.4f})")

# --- Treina o modelo com melhor desempenho médio na base completa de treino ---
melhor_nome = max(resultados, key=lambda n: resultados[n].mean())
melhor_modelo = modelos[melhor_nome]
melhor_modelo.fit(X_train, y_train)

print(f"\nMelhor modelo: {melhor_nome}")

joblib.dump(
    {"modelo": melhor_modelo, "nome": melhor_nome, "X_test": X_test, "y_test": y_test},
    "modelo_final.pkl",
)
print("Salvo em modelo_final.pkl")