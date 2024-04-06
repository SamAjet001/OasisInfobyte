# Import necessary libraries
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import folium

# Specify the file paths
file_path1 = os.path.join(os.path.expanduser("~"), "Desktop", "Unemployment in India.csv")
file_path2 = os.path.join(os.path.expanduser("~"), "Desktop", "Unemployment_Rate_upto_11_2020.csv")

# Load the data
dfs = [pd.read_csv(file_path) for file_path in [file_path1, file_path2]]
df = pd.concat(dfs, ignore_index=True)

# Explore the data
print(df.head())
print(df.info())
print(df.describe())

# Analyze the unemployment rate over time by region
plt.figure(figsize=(12, 6))
df.groupby('Region')[df.columns[3]].plot()
plt.title('Unemployment Rate Over Time')
plt.xlabel('Date')
plt.ylabel('Unemployment Rate (%)')
plt.legend()
plt.show()

# Analyze the unemployment rate by region
plt.figure(figsize=(12, 6))
sns.barplot(x='region', y='estimated unemployment rate', data=df)
plt.title('Unemployment Rate by Region')
plt.xlabel('Region')
plt.ylabel('Unemployment Rate (%)')
plt.xticks(rotation=90)
plt.show()

# Analyze the spatial distribution of unemployment
m = folium.Map(location=[20.5937, 78.9629], zoom_start=5)
folium.GeoJson('india_states.geojson').add_to(m)
df['color'] = pd.cut(df['estimated unemployment rate'], bins=5, labels=['#ffffb2', '#fed976', '#feb24c', '#fd8d3c', '#f03b20'])
folium.GeoJson('india_states.geojson', name='geojson').add_child(
    folium.GeoJsonTooltip(fields=['region', 'estimated unemployment rate'])).add_to(m)
m
