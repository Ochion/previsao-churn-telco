import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, ConfusionMatrixDisplay

dados = joblib.load("modelo_final.pkl")
modelo = dados["modelo"]
nome = dados["nome"]
X_test, y_test = dados["X_test"], dados["y_test"]

y_pred = modelo.predict(X_test)
y_proba = modelo.predict_proba(X_test)[:, 1]

print(f"Modelo avaliado: {nome}\n")
print("Matriz de confusão:")
print(confusion_matrix(y_test, y_pred))

print("\nRelatório de classificação:")
print(classification_report(y_test, y_pred, target_names=["Não Churn", "Churn"]))

auc = roc_auc_score(y_test, y_proba)
print(f"AUC-ROC no teste: {auc:.4f}")

# --- Matriz de confusão em imagem ---
ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=["Não Churn", "Churn"], cmap="Blues")
plt.title(f"Matriz de Confusão — {nome}")
plt.tight_layout()
plt.savefig("matriz_confusao.png", dpi=150)
print("\nSalvo em matriz_confusao.png")

# --- Interpretação: coeficientes (Regressão Logística) ou importância de features (árvores) ---
if hasattr(modelo, "coef_"):
    coeficientes = pd.Series(modelo.coef_[0], index=X_test.columns).sort_values()

    print("\nTop 10 fatores que MAIS aumentam a chance de churn:")
    print(coeficientes.tail(10)[::-1])
    print("\nTop 10 fatores que MAIS reduzem a chance de churn:")
    print(coeficientes.head(10))

    top16 = pd.concat([coeficientes.head(8), coeficientes.tail(8)]).sort_values()
    cores = ["#dc2626" if v > 0 else "#2563eb" for v in top16]
    plt.figure(figsize=(8, 6))
    plt.barh(top16.index, top16.values, color=cores)
    plt.title("Fatores com maior peso na previsão de churn")
    plt.xlabel("Coeficiente (impacto na chance de churn)")
    plt.tight_layout()
    plt.savefig("importancia_features.png", dpi=150)
    print("Salvo em importancia_features.png")

elif hasattr(modelo, "feature_importances_"):
    importancias = pd.Series(modelo.feature_importances_, index=X_test.columns).sort_values(ascending=False)

    print("\nTop 10 features mais importantes:")
    print(importancias.head(10))

    plt.figure(figsize=(8, 6))
    importancias.head(15)[::-1].plot(kind="barh", color="#2563eb")
    plt.title("Importância das features")
    plt.tight_layout()
    plt.savefig("importancia_features.png", dpi=150)
    print("\nSalvo em importancia_features.png")