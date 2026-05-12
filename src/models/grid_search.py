import os
import cv2
import time
import kagglehub
import numpy as np
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from feature_extraction import FeatureExtractor

def load_full_gtsrb_data(img_size=(32, 32)):
    """Downloads and loads the entire GTSRB training dataset."""
    print("Downloading GTSRB dataset...")
    path = kagglehub.dataset_download("meowmeowmeowmeowmeow/gtsrb-german-traffic-sign")
    data_path = os.path.join(path, 'Train')
    
    print(f"Loading images from {data_path}...")
    images = []
    labels = []
    
    for label in sorted(os.listdir(data_path)):
        folder = os.path.join(data_path, label)
        if not os.path.isdir(folder) or not label.isdigit():
            continue
            
        for img_name in os.listdir(folder):
            img_path = os.path.join(folder, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue
            img = cv2.resize(img, img_size)
            images.append(img)
            labels.append(int(label))
            
    return np.array(images), np.array(labels)

def main():
    # 1. Load the full dataset
    X_raw, y = load_full_gtsrb_data(img_size=(32, 32))
    print(f"Successfully loaded {len(X_raw)} images.")
    
    # 2. Extract features for the whole dataset
    print("Extracting features (HOG + Color Histogram). This will take a while...")
    start_time = time.time()
    fe = FeatureExtractor(resize=(32, 32))
    X_features = fe.extract_from_list(list(X_raw), which=('hog', 'hist'), verbose=True)
    print(f"Feature extraction completed in {(time.time() - start_time):.2f} seconds.")
    print(f"Extracted feature shape: {X_features.shape}")

    # 3. Create a training and testing split (GridSearchCV will use CV on the train set)
    X_train, X_test, y_train, y_test = train_test_split(X_features, y, test_size=0.2, random_state=42, stratify=y)

    # 4. Define pipelines and hyperparameter grids for models being tested
    models_to_search = {
        "KNN": {
            "pipeline": Pipeline([
                ('scaler', StandardScaler()),
                ('pca', PCA()),
                ('knn', KNeighborsClassifier())
            ]),
            "param_grid": {
                'pca__n_components': [95, 150], # Based on your previous EDA optimal variance
                'knn__n_neighbors': [3, 5, 7],
                'knn__weights': ['uniform', 'distance']
            }
        },
        "Random_Forest": {
            "pipeline": Pipeline([
                ('scaler', StandardScaler()),
                ('pca', PCA()),
                ('rf', RandomForestClassifier(random_state=42))
            ]),
            "param_grid": {
                'pca__n_components': [95], 
                'rf__n_estimators': [100, 200],
                'rf__max_depth': [None, 15, 30]
            }
        },
        "Gradient_Boosting": {
            "pipeline": Pipeline([
                ('scaler', StandardScaler()),
                ('pca', PCA()),
                ('gb', GradientBoostingClassifier(random_state=42))
            ]),
            "param_grid": {
                'pca__n_components': [95],
                'gb__n_estimators': [50, 100],
                'gb__learning_rate': [0.1, 0.2]
            }
        }
    }

    # 5. Run Grid Search
    print("\\nStarting Grid Search Optimization...")
    best_models = {}
    
    for model_name, config in models_to_search.items():
        print(f"\\n--- Optimizing {model_name} ---")
        grid_search = GridSearchCV(
            estimator=config["pipeline"],
            param_grid=config["param_grid"],
            cv=3,              # 3-fold cross-validation to save time on the large dataset
            scoring='f1_macro', # Optimizing for macro F1-score as used in your EDA
            n_jobs=-1,         # Use all available CPU cores
            verbose=2
        )
        
        search_start = time.time()
        grid_search.fit(X_train, y_train)
        
        print(f"Optimization for {model_name} finished in {(time.time() - search_start):.2f} seconds.")
        print(f"Best Parameters: {grid_search.best_params_}")
        print(f"Best Cross-Validation F1-Score: {grid_search.best_score_:.4f}")
        
        best_models[model_name] = grid_search.best_estimator_

    print("\\nGrid Search Complete. You can now use best_models dictionary to evaluate on X_test.")

if __name__ == '__main__':
    main()