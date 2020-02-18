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

def createValidationCurveKNN():

    df_train = pd.read_csv('./data/report_features_std_train.csv', sep='\t')

    X_train = df_train.drop(['CIK', 'Ticker', 'Company', 'Filing_Date', 'Form_Type', 'Change_Ratio', 'Change_Nominal', 'File_Path'], axis=1).values[:, :768]
    _, y_train = np.unique(df_train['Change_Nominal'].values, return_inverse=True)

    # param_range = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271]
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

def createValidationCurveSVM(C):

    df_train = pd.read_csv('./data/report_features_std_train.csv', sep='\t')
    # C = 0.1

    X_train = df_train.drop(['CIK', 'Ticker', 'Company', 'Filing_Date', 'Form_Type', 'Change_Ratio', 'Change_Nominal', 'File_Path'], axis=1).values[:, :768]
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

def createValidationCurveDecisionTree():

    df_train = pd.read_csv('./data/report_features_train.csv', sep='\t')

    X_train = df_train.drop(['CIK', 'Ticker', 'Company', 'Filing_Date', 'Form_Type', 'Change_Ratio', 'Change_Nominal', 'File_Path'], axis=1).values
    _, y_train = np.unique(df_train['Change_Nominal'].values, return_inverse=True)

    param_range = np.arange(1, 101, 1)
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('tree', DecisionTreeClassifier(criterion='entropy'))
    ])

    train_scores, valid_scores = validation_curve(pipeline, X_train, y_train, param_name="tree__max_depth", param_range=param_range, cv=5, n_jobs=20, verbose=5)

    results = {
        'max_depth': param_range,
        'train_scores_mean': list(np.mean(train_scores, axis=1)),
        'train_scores_std': list(np.std(train_scores, axis=1)),
        'valid_scores_mean': list(np.mean(valid_scores, axis=1)),
        'valid_scores_std': list(np.std(valid_scores, axis=1))
    }

    output_df = pd.DataFrame(results)

    print(output_df)

    output_df.to_csv('./data/decision_tree_validation_curve.csv', index=False)

def createValidationCurveRandomForest():

    df_train = pd.read_csv('./data/report_features_train.csv', sep='\t')

    X_train = df_train.drop(['CIK', 'Ticker', 'Company', 'Filing_Date', 'Form_Type', 'Change_Ratio', 'Change_Nominal', 'File_Path'], axis=1).values
    _, y_train = np.unique(df_train['Change_Nominal'].values, return_inverse=True)

    param_range = np.arange(1, 251, 1)
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('rdf', RandomForestClassifier(criterion='gini', max_depth=3))
    ])

    train_scores, valid_scores = validation_curve(pipeline, X_train, y_train, param_name="rdf__n_estimators", param_range=param_range, cv=5, n_jobs=20, verbose=5)

    results = {
        'n_estimators': param_range,
        'train_scores_mean': list(np.mean(train_scores, axis=1)),
        'train_scores_std': list(np.std(train_scores, axis=1)),
        'valid_scores_mean': list(np.mean(valid_scores, axis=1)),
        'valid_scores_std': list(np.std(valid_scores, axis=1))
    }

    output_df = pd.DataFrame(results)

    print(output_df)

    output_df.to_csv('./data/random_forest_validation_curve.csv', index=False)

def createValidationCurveLogisticRegression():

    df_train = pd.read_csv('./data/report_features_train.csv', sep='\t')

    X_train = df_train.drop(['CIK', 'Ticker', 'Company', 'Filing_Date', 'Form_Type', 'Change_Ratio', 'Change_Nominal', 'File_Path'], axis=1).values
    _, y_train = np.unique(df_train['Change_Nominal'].values, return_inverse=True)

    param_range = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('lr', LogisticRegression())
    ])

    train_scores, valid_scores = validation_curve(pipeline, X_train, y_train, param_name="lr__C", param_range=param_range, cv=5, n_jobs=20, verbose=5)

    results = {
        'C': param_range,
        'train_scores_mean': list(np.mean(train_scores, axis=1)),
        'train_scores_std': list(np.std(train_scores, axis=1)),
        'valid_scores_mean': list(np.mean(valid_scores, axis=1)),
        'valid_scores_std': list(np.std(valid_scores, axis=1))
    }

    output_df = pd.DataFrame(results)

    print(output_df)

    output_df.to_csv('./data/logistic_regression_validation_curve.csv', index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process which model should be selected for the validation curve.')
    parser.add_argument('--model', dest='model', type=str)
    parser.add_argument('--C', dest='C', type=float)
    args = parser.parse_args()
    if args.model == 'Logistic_Regression':
        createValidationCurveLogisticRegression()
    elif args.model == 'Random_Forest':
        createValidationCurveRandomForest()
    elif args.model == 'Decision_Tree':
        createValidationCurveDecisionTree()
    elif args.model == 'SVM':
        createValidationCurveSVM(args.C)
    elif args.model == 'KNN':
        createValidationCurveKNN()
