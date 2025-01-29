import sys
import os
import numpy as np
import pandas as pd
from tensorflow.keras.datasets import fashion_mnist

# Add the parent folder (gsom/) to sys.path
sys.path.append(os.path.abspath('../../'))

from gsom import GSOM  # Import GSOM from the parent folder
from visualize import plot  # Import plot from the parent folder

# 1. Load Fashion MNIST dataset
print("Loading Fashion MNIST dataset...")
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
X_train_flattened = X_train.reshape(X_train.shape[0], -1)  # Flatten images to 1D
X_train_normalized = X_train_flattened / 255.0  # Normalize pixel values to range 0-1

# 2. Prepare DataFrame
print("Preparing data...")
fashion_data = pd.DataFrame(X_train_normalized)
fashion_data['label'] = y_train  # Add labels (clothing categories)
fashion_data['index'] = fashion_data.index  # Add indices for tracking

# 3. Initialize and train the GSOM
print("Initializing and training GSOM...")
gsom = GSOM(spred_factor=0.8, dimensions=784, max_radius=4)  # 784 features for 28x28 images
gsom.fit(fashion_data.iloc[:, :-2].to_numpy(), training_iterations=100, smooth_iterations=50)

# 4. Predict and cluster
print("Clustering data...")
output = gsom.predict(fashion_data, index_col="index", label_col="label")

# 5. Visualize the GSOM map
print("Visualizing results...")
os.makedirs("../outputs", exist_ok=True)  # Ensure the outputs directory exists
plot(output, index_col="index", gsom_map=gsom, file_name="../outputs/fashion_gsom", file_type=".pdf")

# 6. Save cluster assignments
print("Saving results...")
output.to_csv("../outputs/fashion_gsom_clusters.csv", index=False)

print("Cluster analysis complete!")
