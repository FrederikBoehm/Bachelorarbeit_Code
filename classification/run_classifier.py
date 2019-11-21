import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

def optimizeNaiveBayes():
    df_train = pd.read_csv('./data/report_features_train.csv', sep='\t')
    df_test = pd.read_csv('./data/report_features_test.csv', sep='\t')

    X_train = df_train.drop(['CIK', 'Ticker', 'Company', 'Filing_Date', 'Form_Type', 'Change_Ratio', 'Change_Nominal', 'File_Path'], axis=1).values
    X_test = df_test.drop(['CIK', 'Ticker', 'Company', 'Filing_Date', 'Form_Type', 'Change_Ratio', 'Change_Nominal', 'File_Path'], axis=1).values
    _, y_train = np.unique(df_train['Change_Nominal'].values, return_inverse=True)
    _, y_test = np.unique(df_test['Change_Nominal'].values, return_inverse=True)

    pipeline = Pipeline([
        ('nb', GaussianNB())
    ])

    print('Running grid search for Naive Bayes')
    clf = GridSearchCV(pipeline, cv=5, n_jobs=-8, param_grid={})
    clf.fit(X_train, y_train)

    print('-----------RESULTS Naive Bayes-----------')
    print(f'CV score Naive Bayes: {clf.best_score_}')
    print(f'Test score Naive Bayes: {clf.score(X_test, y_test)}')

def optimizeKNN():
    df_train = pd.read_csv('./data/report_features_train.csv', sep='\t')
    df_test = pd.read_csv('./data/report_features_test.csv', sep='\t')

    X_train = df_train.drop(['CIK', 'Ticker', 'Company', 'Filing_Date', 'Form_Type', 'Change_Ratio', 'Change_Nominal', 'File_Path'], axis=1).values
    X_test = df_test.drop(['CIK', 'Ticker', 'Company', 'Filing_Date', 'Form_Type', 'Change_Ratio', 'Change_Nominal', 'File_Path'], axis=1).values
    _, y_train = np.unique(df_train['Change_Nominal'].values, return_inverse=True)
    _, y_test = np.unique(df_test['Change_Nominal'].values, return_inverse=True)

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('knn', KNeighborsClassifier())
    ])

    print('Running grid search for KNN')
    clf = GridSearchCV(pipeline, cv=5, n_jobs=-8, param_grid={
        'knn__n_neighbors': [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271]
    })
    clf.fit(X_train, y_train)

    print('-----------RESULTS KNN-----------')
    print(clf.best_params_)
    print(f'CV score KNN: {clf.best_score_}')
    print(f'Test score KNN: {clf.score(X_test, y_test)}')

    results_df = pd.DataFrame(clf.cv_results_)
    results_df.to_csv('./data/knn_results.csv', index=False)

def optimizeSVM():
    df_train = pd.read_csv('./data/report_features_train.csv', sep='\t')
    df_test = pd.read_csv('./data/report_features_test.csv', sep='\t')

    X_train = df_train.drop(['CIK', 'Ticker', 'Company', 'Filing_Date', 'Form_Type', 'Change_Ratio', 'Change_Nominal', 'File_Path'], axis=1).values
    X_test = df_test.drop(['CIK', 'Ticker', 'Company', 'Filing_Date', 'Form_Type', 'Change_Ratio', 'Change_Nominal', 'File_Path'], axis=1).values
    _, y_train = np.unique(df_train['Change_Nominal'].values, return_inverse=True)
    _, y_test = np.unique(df_test['Change_Nominal'].values, return_inverse=True)

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC())
    ])

    print('Running grid search for SVM')
    clf = GridSearchCV(pipeline, cv=5, n_jobs=-8, param_grid=[
        {
            'svm__kernel': ['linear'],
            'svm__C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]
        },
        {
            'svm__kernel': ['poly'],
            'svm__C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
            'svm__degree': [2, 3],
            'svm__coef0': [0, 1]
        },
        {
            'svm__kernel': ['rbf'],
            'svm__C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
            'svm__gamma': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]
        }

    ])
    
    clf.fit(X_train, y_train)

    print('-----------RESULTS SVM-----------')
    print(clf.best_params_)
    print(f'CV score KNN: {clf.best_score_}')
    print(f'Test score KNN: {clf.score(X_test, y_test)}')

if __name__ == "__main__":
    # optimizeNaiveBayes()
    # optimizeKNN()
    optimizeSVM()
