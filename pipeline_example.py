"""
This file is an example of how the pipeline should be used after assembling the rest of the parts.
"""
from feature_extraction import FeatureExtractor
from baselines import BaselineModels
import numpy as np
from sklearn.model_selection import train_test_split


def demo_with_synthetic_data():
    X = np.random.randn(200, 128)
    y = np.random.randint(0, 2, size=(200,))
    return X, y


def run_pipeline(images=None, labels=None, which_features=('hog', 'hist')):
    if images is None:
        X, y = demo_with_synthetic_data()
    else:
        fe = FeatureExtractor(resize=(128, 128))
        X = fe.extract_from_list(images, which=which_features)
        y = np.asarray(labels)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    model = BaselineModels(knn_k=5, use_scaler=True, use_pca=False)
    model.fit(X_train, y_train)

    metrics_knn = model.evaluate(X_test, y_test, model='knn')
    metrics_nb = model.evaluate(X_test, y_test, model='nb')

    print('KNN metrics:')
    for k, v in metrics_knn.items():
        print(f'  {k}: {v}')

    print('\nNaive Bayes metrics:')
    for k, v in metrics_nb.items():
        print(f'  {k}: {v}')


if __name__ == '__main__':
    run_pipeline()
