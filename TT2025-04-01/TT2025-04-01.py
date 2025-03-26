import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances

# Assume you have your Pokémon data in a DataFrame called 'df'
# with columns like 'pokemon', 'hp', 'attack', 'defense', 'special_attack', 'special_defense', 'speed'
df = pd.read_csv('https://raw.githubusercontent.com/rfordatascience/tidytuesday/main/data/2025/2025-04-01/pokemon_df.csv')

# Define the stat columns you want to use for similarity
stats_columns = ['hp', 'attack', 'defense', 'special_attack', 'special_defense', 'speed']

# Standardize the stats data
scaler = StandardScaler()
stats_scaled = scaler.fit_transform(df[stats_columns])

# Compute the pairwise Euclidean distance matrix
dist_matrix = euclidean_distances(stats_scaled)

# Create a NetworkX graph
G = nx.Graph()

# Add nodes with attributes (e.g., Pokémon name and other info)
for idx, row in df.iterrows():
    G.add_node(row['pokemon'], **row.to_dict())

# Define a threshold for connecting nodes (tune this value based on your data)
threshold = 1.5  # Example threshold

# Add edges for pairs with a distance below the threshold
n = len(df)
for i in range(n):
    for j in range(i + 1, n):
        if dist_matrix[i, j] < threshold:
            G.add_edge(df.iloc[i]['pokemon'], df.iloc[j]['pokemon'], weight=dist_matrix[i, j])

# Visualize the network graph using a spring layout
pos = nx.spring_layout(G, seed=42)  # Positions for nodes
plt.figure(figsize=(12, 8))
nx.draw(G, pos, with_labels=True, node_size=300, font_size=8)
plt.title("Pokémon Similarity Network Graph")
plt.show()
