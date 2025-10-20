# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import scipy.cluster.hierarchy as sc
import scipy.spatial.distance as sc_dist
import kneed

path = r"C:\Users\gbour\Downloads"
data = pd.read_excel(path + r"\UNDP_HDI.xlsx")

data.head()

# %%
data['Mean years of schooling (2022)'] = data['Mean years of schooling (2022)'].astype(float)

# %%
data.info()

# %%
data['Education Score (2022)'] = (data['Mean years of schooling (2022)'] + data['Expected years of schooling (2022)'])/2

# %%
data.head()

# %%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=2)
kmeans.fit(data.iloc[:, 1:])
labels = kmeans.labels_
fig = plt.figure(figsize = (15,15))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(data['Life expectancy at birth (2022)'], data['Education Score (2022)'], 
           data['Gross national income (GNI) per capita (2022)'], c=labels, cmap='viridis')
ax.set_xlabel("Life expectancy at birth")
ax.set_ylabel("Education Score")
ax.set_zlabel("GNI per capita")
plt.show()

# %%
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# Supposons que data2 contient votre jeu de données

# Sélection des colonnes pertinentes
selected_columns = ['Life expectancy at birth (2022)', 
                    'Gross national income (GNI) per capita (2022)', 
                    'Education Score (2022)']

# Extraction des données
data_bis = data[selected_columns]

# Normalisation des données
scaler = StandardScaler()
pipeline = make_pipeline(scaler)
data_normalized = pipeline.fit_transform(data_bis)

# Initialisation de la liste des inerties
inertias = []

# Nombre de clusters à tester
k_values = range(1, 11)

# Calcul de l'inertie pour chaque valeur de k
for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(data_normalized)
    inertias.append(kmeans.inertia_)

# Tracer le graphique de l'inertie en fonction du nombre de clusters
plt.figure(figsize=(10, 6))
plt.plot(k_values, inertias, marker='o')
plt.title('Elbow Method for Optimal Number of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.xticks(k_values)
plt.grid(True)
plt.show()

# %%
import pandas as pd
import plotly.graph_objs as go

# Assuming 'data' is your existing DataFrame with the required columns and country names
# If 'data' is already defined elsewhere in your code, you don't need to redefine it here

model = KMeans(n_clusters=2, init="k-means++", max_iter=300, n_init=10, random_state=0)
labels = model.fit_predict(data[['Life expectancy at birth (2022)', 'Education Score (2022)', 'Gross national income (GNI) per capita (2022)']])

# Create a Scatter3d trace
trace = go.Scatter3d(
    x=data['Life expectancy at birth (2022)'],
    y=data['Education Score (2022)'],
    z=data['Gross national income (GNI) per capita (2022)'],
    mode='markers',
    marker=dict(
        color=labels,
        size=10,
        line=dict(
            color='black',
            width=10
        )
    ),
    text=data['Country']  # Adding country names as text annotations
)

# Define layout
layout = go.Layout(
    margin=dict(l=0, r=0),
    scene=dict(
        xaxis=dict(title='Life expectancy at birth (2022)'),
        yaxis=dict(title='Education Score (2022)'),
        zaxis=dict(title='Gross national income (GNI) per capita (2022)')
    ),
    height=800,
    width=800
)

# Create the figure
fig = go.Figure(data=[trace], layout=layout)

# Show the figure
fig.show()

fig.write_html("interactive_graph.html")


# %%
data.head()

# %%
import pandas as pd
import plotly.graph_objs as go
from sklearn.cluster import KMeans

model = KMeans(n_clusters=2, init="k-means++", max_iter=300, n_init=10, random_state=0)
labels = model.fit_predict(data[['Life expectancy at birth (2022)', 'Education Score (2022)', 'Gross national income (GNI) per capita (2022)']])

cluster_1_indices = labels == 0
cluster_2_indices = labels == 1

trace_cluster_1 = go.Scatter3d(
    x=data.loc[cluster_1_indices, 'Life expectancy at birth (2022)'],
    y=data.loc[cluster_1_indices, 'Education Score (2022)'],
    z=data.loc[cluster_1_indices, 'Gross national income (GNI) per capita (2022)'],
    mode='markers',
    marker=dict(
        color='blue',  
        size=10,
        line=dict(
            color='black',
            width=0.5
        )
    ),
    name='Developing Countries',
    text=data.loc[cluster_1_indices, 'Country']
)

trace_cluster_2 = go.Scatter3d(
    x=data.loc[cluster_2_indices, 'Life expectancy at birth (2022)'],
    y=data.loc[cluster_2_indices, 'Education Score (2022)'],
    z=data.loc[cluster_2_indices, 'Gross national income (GNI) per capita (2022)'],
    mode='markers',
    marker=dict(
        color='yellow',
        size=10,
        line=dict(
            color='black',
            width=0.5
        )
    ),
    name='Developed Countries',
    text=data.loc[cluster_2_indices, 'Country']
)

traces = [trace_cluster_1, trace_cluster_2]

layout = go.Layout(
    margin=dict(l=0, r=0),
    scene=dict(
        xaxis=dict(title='Life expectancy at birth (2022)'),
        yaxis=dict(title='Education Score (2022)'),
        zaxis=dict(title='Gross national income (GNI) per capita (2022)')
    ),
    height=800,
    width=800,
    legend=dict(
        x=0.85,
        y=0.95
    )
)

fig = go.Figure(data=traces, layout=layout)
fig.show()

fig.write_html("interactive_graph.html")


# %%
data['Cluster'] = labels
cluster_names = {0: "Developed Countries", 1: "Developing Countries"}

for cluster_id, cluster_name in cluster_names.items():
    cluster_data = data[data['Cluster'] == cluster_id] 
    countries_in_cluster = cluster_data['Country'].tolist()
    print(f"{cluster_name}:")
    print(countries_in_cluster)

# %%
data2 = pd.read_excel(path + r"\HDI & IHDI.xlsx")

kmeans = KMeans(n_clusters=5)
kmeans.fit(data2.iloc[:, 1:])
labels = kmeans.labels_
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(data2['Carbon dioxide emissions (production) index -2021'],
           data2['Human Development Index (HDI) - 2022'], 
           data2['Gender Inequality Index (GII) - 2022 (formated)'], 
           c=labels, cmap='viridis')
ax.set_xlabel("CDEI")
ax.set_ylabel("HDI")
ax.set_zlabel("IDHI")
plt.show()

# %%
data2.head()

# %%
model2 = KMeans(n_clusters=4,init="k-means++", max_iter=300, n_init=10, random_state=0)
labels = model2.fit_predict(data2[['Gender Inequality Index (GII) - 2022 (formated)', 'Human Development Index (HDI) - 2022', 
                                 'Carbon dioxide emissions (production) index -2021']])

trace = go.Scatter3d(
    x=data2['Carbon dioxide emissions (production) index -2021'],
    y=data2['Human Development Index (HDI) - 2022'],
    z=data2['Gender Inequality Index (GII) - 2022 (formated)'], 
    mode='markers',
    marker=dict(
        color=labels,
        size=10,
        line=dict(
            color='black',
            width=10
        )
    ),
    text=data2['Country']
)

layout = go.Layout(
    margin=dict(l=0, r=0),
    scene=dict(
        xaxis=dict(title='Carbon dioxide emissions (production) index'),
        yaxis=dict(title='Human Development Index (HDI)'),
        zaxis=dict(title='Gender Inequality Index (GII) - 2022 (formated)')
    ),
    height=800,
    width=800
)

fig = go.Figure(data=[trace], layout=layout)
fig.show()

fig.write_html("interactive_graph_2.html")

# %%
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# Supposons que data2 contient votre jeu de données

# Sélection des colonnes pertinentes
selected_columns = ['Gender Inequality Index (GII) - 2022 (formated)', 
                    'Human Development Index (HDI) - 2022', 
                    'Carbon dioxide emissions (production) index -2021']

# Extraction des données
data = data2[selected_columns]

# Normalisation des données
scaler = StandardScaler()
pipeline = make_pipeline(scaler)
data_normalized = pipeline.fit_transform(data)

# Initialisation de la liste des inerties
inertias = []

# Nombre de clusters à tester
k_values = range(1, 11)

# Calcul de l'inertie pour chaque valeur de k
for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(data_normalized)
    inertias.append(kmeans.inertia_)

# Tracer le graphique de l'inertie en fonction du nombre de clusters
plt.figure(figsize=(10, 6))
plt.plot(k_values, inertias, marker='o')
plt.title('Elbow Method for Optimal Number of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.xticks(k_values)
plt.grid(True)
plt.show()

# %%
data2['Cluster'] = labels
cluster_names = {0: "Developing Countries", 1: "Developed Countries", 2: "Developing Countries", 3:"Developing Countries",
                4 : "Developed Countries"}

developing_countries = []
developed_countries = []

for cluster_id, cluster_name in cluster_names.items():
    cluster_data = data2[data2['Cluster'] == cluster_id] 
    countries_in_cluster = cluster_data['Country'].tolist()
    
    if cluster_name == "Developing Countries":
        developing_countries.extend(countries_in_cluster)
    elif cluster_name == "Developed Countries":
        developed_countries.extend(countries_in_cluster)

print("Developing Countries:")
print(developing_countries)

print("\nDeveloped Countries:")
print(developed_countries)

# %%
# Supposons que vous avez déjà défini model2, labels et data2 comme vous l'avez fait dans votre code

# Création d'une colonne dans le DataFrame pour les étiquettes prédites
data2['Cluster'] = labels

# Création des listes pour les pays développés et en développement
developed_countries = data2[data2['Cluster'].isin([0, 3])]['Country'].tolist()
developing_countries = data2[data2['Cluster'].isin([1, 2, 4])]['Country'].tolist()

# Affichage des résultats
print("Developed Countries:", developed_countries)
print("\nDeveloping Countries:", developing_countries)


# %%



