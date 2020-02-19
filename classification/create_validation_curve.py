import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import validation_curve
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
import argparse


# Creates validation curves for either KNN or SVM

def createValidationCurveKNN(feature_vector_dimension):

    df_train = pd.read_csv('./data/report_features_std_train.csv', sep='\t')

    X_train = df_train.drop(['CIK', 'Ticker', 'Company', 'Filing_Date', 'Form_Type', 'Change_Ratio', 'Change_Nominal', 'File_Path'], axis=1).values[:, :feature_vector_dimension]
    _, y_train = np.unique(df_train['Change_Nominal'].values, return_inverse=True)

    param_range = np.arange(3, 350, step=2)
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('knn', KNeighborsClassifier())
    ])

    train_scores, valid_scores = validation_curve(pipeline, X_train, y_train, param_name="knn__n_neighbors", param_range=param_range, cv=5, n_jobs=20, verbose=5)

    results = {
        'n_neighbors': param_range,
        'train_scores_mean': list(np.mean(train_scores, axis=1)),
        'train_scores_std': list(np.std(train_scores, axis=1)),
        'valid_scores_mean': list(np.mean(valid_scores, axis=1)),
        'valid_scores_std': list(np.std(valid_scores, axis=1))
    }

    output_df = pd.DataFrame(results)

    print(output_df)

    output_df.to_csv('./data/knn_validation_curve.csv', index=False)

def createValidationCurveSVM(feature_vector_dimension):

    df_train = pd.read_csv('./data/report_features_std_train.csv', sep='\t')

    X_train = df_train.drop(['CIK', 'Ticker', 'Company', 'Filing_Date', 'Form_Type', 'Change_Ratio', 'Change_Nominal', 'File_Path'], axis=1).values[:, :feature_vector_dimension]
    _, y_train = np.unique(df_train['Change_Nominal'].values, return_inverse=True)

    param_range = [1e-7, 1e-6, 1e-5, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100]
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(kernel='rbf', C=C))
    ])

    train_scores, valid_scores = validation_curve(pipeline, X_train, y_train, param_name="svm__gamma", param_range=param_range, cv=5, n_jobs=20, verbose=5)

    results = {
        'gamma': param_range,
        'train_scores_mean': list(np.mean(train_scores, axis=1)),
        'train_scores_std': list(np.std(train_scores, axis=1)),
        'valid_scores_mean': list(np.mean(valid_scores, axis=1)),
        'valid_scores_std': list(np.std(valid_scores, axis=1))
    }

    output_df = pd.DataFrame(results)

    print(output_df)

    output_df.to_csv(f'./data/svm_validation_curve_C_{C}.csv', index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process which model should be selected for the validation curve.')
    parser.add_argument('--model', dest='model', type=str)
    parser.add_argument('--mode', dest='mode', type=str)
    parser.add_argument('--C', dest='C', type=float)
    args = parser.parse_args()
    feature_dimension = None
    if args.mode == 'AVG_ONLY':
        feature_dimension = 768
    elif args.mode == 'USE_STD':
        feature_dimension = 1536
    else:
        print('Parameter --mode not set, use AVG_ONLY or USE_STD')

    if args.model == 'SVM':
        createValidationCurveSVM(feature_vector_dimension, args.C)
    elif args.model == 'KNN':
        createValidationCurveKNN(feature_vector_dimension)
