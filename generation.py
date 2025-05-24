# Importiamo le librerie
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import random

# Leggiamo il dataset
dataset = pd.read_csv('dataset_traffic_accident_prediction1.csv', sep=';')

# Visualizziamo le prime righe
print(dataset.head())

# Selezioniamo la colonna "Road_Condition"
road_condition_col = 'Weather'
categories = dataset[road_condition_col].unique()

# Calcoliamo la distribuzione attuale
counts = dataset[road_condition_col].value_counts()
percents = dataset[road_condition_col].value_counts(normalize=True).mul(100).round(1)

distribution = pd.concat([counts, percents], axis=1)
distribution.columns = ["Count", "Percentage (%)"]

print("\nDistribuzione attuale di 'Road_Condition':\n")
print(distribution)

# Numero di righe totali desiderate
target_total = 5000
current_total = 841
n_to_generate = target_total - current_total

print(f"\nNumero di righe attuali: {current_total}")
print(f"Numero di righe da generare: {n_to_generate}")

# Prepariamo i pesi reali per la generazione basati sulla distribuzione attuale
weights = [percents.get(cat, 0) for cat in categories]

# Convertiamo le percentuali in pesi normalizzati (sommati a 1)
weights = [w / sum(weights) for w in weights]

# Generiamo nuovi valori in modo generalizzato
new_data = random.choices(
    population=list(categories),
    weights=weights,
    k=n_to_generate
)

# Salviamo i nuovi dati in un file
with open("generated_road_type.txt", "w") as f:
    for item in new_data:
        f.write(item + "\n")

print("\nNuovi dati generati e salvati in 'generated_road_conditions.txt'")
