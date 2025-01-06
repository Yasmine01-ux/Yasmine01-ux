# Importation des bibliothèques nécessaires
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error, accuracy_score

# --- Création de données fictives ---
np.random.seed(42)
data = {
    'Heures d\'étude': np.random.normal(6, 2, 200),  # Heures d'étude
    'Assiduité': np.random.normal(80, 10, 200),  # Pourcentage d'assiduité
    'Note précédente': np.random.normal(70, 15, 200),  # Notes précédentes
    'Note finale': np.random.normal(75, 10, 200)  # Notes finales
}
df = pd.DataFrame(data)

# --- Prétraitement des données ---
# Séparation des variables indépendantes (features) et dépendantes (target)
X = df[['Heures d\'étude', 'Assiduité', 'Note précédente']]
y_reg = df['Note finale']  # Variable cible pour la régression
y_class = (df['Note finale'] >= 70).astype(int)  # 1 pour réussite, 0 pour échec

# Séparation des données en ensembles d'entraînement et de test (80% train, 20% test)
X_train, X_test, y_train_reg, y_test_reg = train_test_split(X, y_reg, test_size=0.2, random_state=42)
X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(X, y_class, test_size=0.2, random_state=42)

# Normalisation des données (mise à l'échelle des features)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- Modèles et Entraînement ---
# 1. Régression linéaire (prédiction de la note finale)
regressor = LinearRegression()
regressor.fit(X_train_scaled, y_train_reg)
y_pred_reg = regressor.predict(X_test_scaled)

# 2. Classification avec Random Forest (prédiction de réussite ou échec)
classifier = RandomForestClassifier(random_state=42)
classifier.fit(X_train_scaled, y_train_class)
y_pred_class = classifier.predict(X_test_scaled)

# 3. Clustering avec KMeans (utilisé pour détecter des groupes dans les données)
kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(X)

# --- Évaluation des Modèles ---
mae = mean_absolute_error(y_test_reg, y_pred_reg)  # Erreur absolue moyenne pour la régression
accuracy = accuracy_score(y_test_class, y_pred_class)  # Précision pour la classification

# Affichage des résultats des évaluations
print(f"Erreur Absolue Moyenne (Régression) : {mae:.2f}")
print(f"Précision (Classification) : {accuracy:.2f}")

# --- Visualisation ---
# Visualisation des clusters obtenus par KMeans
sns.pairplot(df, hue='Cluster')

# Légendes en français pour la figure
plt.suptitle('Visualisation des Clusters (KMeans)', fontsize=16)
plt.show()
