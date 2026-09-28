import pandas as pd

pd.set_option("display.max_columns", None)

CAMINHO_DADOS = "Telco-Customer-Churn.csv"  # ajuste se o nome do arquivo baixado for diferente

df = pd.read_csv(CAMINHO_DADOS)

print("Formato do dataset:", df.shape)
print("\nTipos de dados:\n", df.dtypes)
print("\nValores nulos por coluna:\n", df.isnull().sum())

# --- Tratamento: TotalCharges vem como texto, com alguns registros em branco ---
df["TotalCharges"] = df["TotalCharges"].replace(" ", pd.NA)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"])

print(f"\nRegistros com TotalCharges ausente: {df['TotalCharges'].isnull().sum()}")
# são poucos, geralmente clientes com tenure=0 (ainda não faturados) — remover é seguro
df = df.dropna(subset=["TotalCharges"])

# --- Distribuição do target ---
distribuicao_churn = df["Churn"].value_counts(normalize=True) * 100
print("\nDistribuição de churn (%):\n", distribuicao_churn.round(2))

# --- Correlação entre variáveis numéricas e churn ---
df["churn_binario"] = df["Churn"].map({"Yes": 1, "No": 0})
numericas = ["tenure", "MonthlyCharges", "TotalCharges"]
correlacao = df[numericas + ["churn_binario"]].corr()["churn_binario"].sort_values(ascending=False)
print("\nCorrelação com churn (variáveis numéricas):\n", correlacao)

# --- Churn por variáveis categóricas relevantes ---
categoricas_chave = ["Contract", "InternetService", "PaymentMethod", "TechSupport"]
for coluna in categoricas_chave:
    print(f"\nTaxa de churn por {coluna} (%):")
    print((df.groupby(coluna)["churn_binario"].mean().sort_values(ascending=False) * 100).round(1))

df.to_csv("dados_tratados.csv", index=False)
print("\nSalvo em dados_tratados.csv")