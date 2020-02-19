from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import validation_curve, learning_curve
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import pandas as pd
import numpy as np
import argparse

# Creates a sample learning curve and validation curve for the sample data generated with create_sample_data.py

def createValidationDiagram():
    df_train = pd.read_csv('./sample_data.csv', sep='\t')

    X_train = df_train.drop(['LABEL'], axis=1).values
    y_train = df_train['LABEL'].values

    param_range = np.arange(3, 1000, step=2)
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('knn', KNeighborsClassifier())
    ])

    train_scores, valid_scores = validation_curve(pipeline, X_train, y_train, param_name="knn__n_neighbors", param_range=param_range, cv=5, n_jobs=6, verbose=5)

    results = {
        'n_neighbors': param_range,
        'train_scores_mean': list(np.mean(train_scores, axis=1)),
        'train_scores_std': list(np.std(train_scores, axis=1)),
        'valid_scores_mean': list(np.mean(valid_scores, axis=1)),
        'valid_scores_std': list(np.std(valid_scores, axis=1))
    }

    output_df = pd.DataFrame(results)

    print(output_df)

    output_df.to_csv('./knn_sample_validation_curve.csv', index=False)

def createLearningDiagram():
    df_train = pd.read_csv('./sample_data.csv', sep='\t')

    X_train = df_train.drop(['LABEL'], axis=1).values
    y_train = df_train['LABEL'].values

    train_sizes = np.linspace(0.1, 1.0, 20)

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('knn', KNeighborsClassifier(n_neighbors=157))
    ])

    train_sizes_abs, train_scores, valid_scores = learning_curve(pipeline, X_train, y_train, train_sizes=train_sizes, cv=5, n_jobs=6, verbose=5)

    results = {
        'train_sizes': train_sizes_abs,
        'train_scores_mean': list(np.mean(train_scores, axis=1)),
        'train_scores_std': list(np.std(train_scores, axis=1)),
        'valid_scores_mean': list(np.mean(valid_scores, axis=1)),
        'valid_scores_std': list(np.std(valid_scores, axis=1))
    }

    output_df = pd.DataFrame(results)

    print(output_df)

    output_df.to_csv('./knn_sample_learning_curve.csv', index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process which model should be selected for the learning curve.')
    parser.add_argument('--diagram', dest='diagram', type=str)
    args = parser.parse_args()
    if args.diagram == 'learning':
        createLearningDiagram()
    else:
        createValidationDiagram()