import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# 1. Setup Mock Data 
print("Loading data...")
data = load_digits()
X = data.data
y = data.target

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Define the variance levels you want to test
variance_levels = [0.50, 0.75, 0.85, 0.90, 0.95, 0.99, 1.0]

results = []

print("\n--- Starting PCA Trade-off Analysis ---")

for var in variance_levels:
    if var == 1.0:
        # No PCA - Baseline
        X_train_pca = X_train
        X_test_pca = X_test
        n_components = X_train.shape[1]
    else:
        pca = PCA(n_components=var, random_state=42)
        
        X_train_pca = pca.fit_transform(X_train)
        X_test_pca = pca.transform(X_test)
        n_components = pca.n_components_

    # Train a baseline model
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train_pca, y_train)
    
    # Predict and calculate accuracy
    y_pred = knn.predict(X_test_pca)
    acc = accuracy_score(y_test, y_pred)
    
    # Save the results
    results.append((var, n_components, acc))
    
    # Print the step results
    print(f"Target Variance: {var*100:>5.1f}% | Components Kept: {n_components:>3} out of {X_train.shape[1]} | Model Accuracy: {acc*100:.2f}%")

# 4. Optional: Visualize the Trade-off for your team presentation
variances, components, accuracies = zip(*results)

plt.figure(figsize=(10, 5))

# Plot Accuracy vs Components
plt.plot(components, accuracies, marker='o', linestyle='-', color='b')
plt.title('PCA Compression vs. KNN Accuracy Trade-off')
plt.xlabel('Number of PCA Components (Data Size)')
plt.ylabel('Model Accuracy')
plt.grid(True)
plt.show()