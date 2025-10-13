import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('models/pca_explained_variance.csv')
plt.plot(df['component'], df['cumulative_variance'], marker='o')
plt.xlabel('Principal component')
plt.ylabel('Cumulative explained variance')
plt.axhline(0.95, color='red', linestyle='--')
plt.grid(True)
plt.show()