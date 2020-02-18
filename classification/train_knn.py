import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import pickle
import numpy as np
import argparse

# Train model on full training set after best hyperparameters were selected with validation curve -> Create final test accuracy

def trainKNN(train_on_average_features, k):
    print('Reading training data')
    df_train = pd.read_csv('./data/report_features_std_train.csv', sep='\t')

    X_train = None
    if train_on_average_features:
        X_train = df_train.drop(['CIK', 'Ticker', 'Company', 'Filing_Date', 'Form_Type', 'Change_Ratio', 'Change_Nominal', 'File_Path'], axis=1).values[:, :768]
    else:
        X_train = df_train.drop(['CIK', 'Ticker', 'Company', 'Filing_Date', 'Form_Type', 'Change_Ratio', 'Change_Nominal', 'File_Path'], axis=1).values

    _, y_train = np.unique(df_train['Change_Nominal'].values, return_inverse=True)

    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)

    print('Fitting on data...')
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)

    print('Saving model...')
    if train_on_average_features:
        with open('./data/knn_avg.pkl', 'wb') as fid:
            pickle.dump(model, fid)
    else:
        with open('./data/knn_std.pkl', 'wb') as fid:
            pickle.dump(model, fid)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Select model path and name.')
    parser.add_argument('--train_on_average_features', dest='train_on_average_features', action='store_true')
    parser.add_argument('--k', dest='k', type=int)
    args = parser.parse_args()
    trainKNN(args.train_on_average_features, args.k)

