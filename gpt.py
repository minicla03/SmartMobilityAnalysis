import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# Caricamento dataset
df = pd.read_csv("dt.csv")

# Conversione Timestamp
df['Timestamp'] = pd.to_datetime(df['Timestamp'])
df['Hour'] = df['Timestamp'].dt.hour
df['DayOfWeek'] = df['Timestamp'].dt.dayofweek

# Codifica ciclica
df['sin_hour'] = np.sin(2 * np.pi * df['Hour'] / 24)
df['cos_hour'] = np.cos(2 * np.pi * df['Hour'] / 24)
df['sin_dayofweek'] = np.sin(2 * np.pi * df['DayOfWeek'] / 7)
df['cos_dayofweek'] = np.cos(2 * np.pi * df['DayOfWeek'] / 7)

# Gestione valori nulli
df['Road_Condition'].fillna('Unknown', inplace=True)

# Feature engineering
df['occupancy_per_vehicle'] = df['Road_Occupancy_%'] / (df['Total'] + 1e-5)
df['public_transport_ratio'] = (df['BusCount'] + df['BikeCount']) / (df['Total'] + 1e-5)

# Encoding del target
le = LabelEncoder()
df['Traffic_Situation_Label'] = le.fit_transform(df['Traffic Situation'])

# One-hot encoding
df = pd.get_dummies(df, columns=['Traffic_Light_State', 'Weather_Condition', 'Road_Condition'], drop_first=True)

# Rimozione colonne non necessarie
df.drop(columns=['Timestamp', 'Latitude', 'Longitude', 'Traffic Situation'], inplace=True)

# Separazione feature e target
X = df.drop(columns=['Traffic_Situation_Label'])
y = df['Traffic_Situation_Label']

# Divisione train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Addestramento Random Forest
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Predizioni
y_pred = clf.predict(X_test)

# Valutazione
print("Classification Report:\n")
print(classification_report(y_test, y_pred, target_names=le.classes_))
print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))
print("Test Accuracy: {:.2f}%".format(clf.score(X_test, y_test) * 100))
print("Train Accuracy: {:.2f}%".format(clf.score(X_train, y_train) * 100))