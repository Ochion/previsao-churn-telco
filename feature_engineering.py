import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

df = pd.read_csv("dados_tratados.csv")

# --- Separar target e remover colunas não preditivas ---
y = df["churn_binario"]
X = df.drop(columns=["customerID", "Churn", "churn_binario"])

# --- One-hot encoding das variáveis categóricas ---
colunas_categoricas = X.select_dtypes(include=["object", "str"]).columns.tolist()
X = pd.get_dummies(X, columns=colunas_categoricas, drop_first=True)

print(f"Formato após encoding: {X.shape}")

# --- Split treino/teste, estratificado pra preservar a proporção de churn (~27%) ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- Normalização das variáveis numéricas (fit só no treino, pra não vazar dado do teste) ---
colunas_numericas = ["tenure", "MonthlyCharges", "TotalCharges"]
scaler = StandardScaler()
X_train[colunas_numericas] = scaler.fit_transform(X_train[colunas_numericas])
X_test[colunas_numericas] = scaler.transform(X_test[colunas_numericas])

print(f"Treino: {X_train.shape[0]} registros | Teste: {X_test.shape[0]} registros")
print(f"Proporção de churn no treino: {y_train.mean():.3f} | no teste: {y_test.mean():.3f}")

joblib.dump(
    {"X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test, "scaler": scaler},
    "dados_modelagem.pkl",
)
print("\nSalvo em dados_modelagem.pkl")