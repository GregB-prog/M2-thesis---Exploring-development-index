import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import plotly.graph_objs as go

# Data -- HDI
path = r"C:\Users\gbour\Downloads"
data = pd.read_excel(path + r"\UNDP_HDI.xlsx")

data['Mean years of schooling (2022)'] = data['Mean years of schooling (2022)'].astype(float)

# Education score
data['Education Score (2022)'] = (
    data['Mean years of schooling (2022)'] + data['Expected years of schooling (2022)']
) / 2

print(data.info())
print(data.head())

# Elbow Method
def plot_elbow(data, cols, k_max=10, title="Elbow Method"):
    scaler = StandardScaler()
    pipeline = make_pipeline(scaler)
    data_norm = pipeline.fit_transform(data[cols])

    inertias = []
    for k in range(1, k_max + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(data_norm)
        inertias.append(kmeans.inertia_)

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, k_max + 1), inertias, marker='o')
    plt.title(title)
    plt.xlabel("Nombre de clusters")
    plt.ylabel("Inertie")
    plt.grid(True)
    plt.show()

cols1 = ['Life expectancy at birth (2022)', 
         'Gross national income (GNI) per capita (2022)', 
         'Education Score (2022)']

plot_elbow(data, cols1, title="Elbow Method – UNDP HDI")

# Clustering & visualisation
def plot_3d_clusters(df, cols, n_clusters=2, filename="interactive_graph.html"):
    model = KMeans(n_clusters=n_clusters, init="k-means++", random_state=0)
    labels = model.fit_predict(df[cols])
    df['Cluster'] = labels

    trace = go.Scatter3d(
        x=df[cols[0]],
        y=df[cols[1]],
        z=df[cols[2]],
        mode='markers',
        marker=dict(color=labels, size=8, line=dict(color='black', width=0.5)),
        text=df['Country']
    )

    layout = go.Layout(
        margin=dict(l=0, r=0),
        scene=dict(
            xaxis=dict(title=cols[0]),
            yaxis=dict(title=cols[1]),
            zaxis=dict(title=cols[2])
        ),
        height=800,
        width=800
    )

    fig = go.Figure(data=[trace], layout=layout)
    fig.show()
    fig.write_html(filename)

    return labels

labels = plot_3d_clusters(data, cols1, n_clusters=2, filename="interactive_graph.html")

# Displaying countries
cluster_names = {0: "Developing Countries", 1: "Developed Countries"} 
data['Cluster'] = labels

for cluster_id, cluster_name in cluster_names.items():
    countries = data.loc[data['Cluster'] == cluster_id, 'Country'].tolist()
    print(f"\n{cluster_name}:")
    print(countries)

# Data -- Experimentation of a development index
data2 = pd.read_excel(path + r"\HDI & IHDI.xlsx")

cols2 = [
    'Gender Inequality Index (GII) - 2022 (formated)',
    'Human Development Index (HDI) - 2022',
    'Carbon dioxide emissions (production) index -2021'
]

plot_elbow(data2, cols2, title="Elbow Method – HDI & IHDI")

labels2 = plot_3d_clusters(data2, cols2, n_clusters=4, filename="interactive_graph_2.html")

data2['Cluster'] = labels2

developed = data2[data2['Cluster'].isin([0, 3])]['Country'].tolist()
developing = data2[data2['Cluster'].isin([1, 2])]['Country'].tolist()

print("\nDeveloped Countries:", developed)
print("\nDeveloping Countries:", developing)


