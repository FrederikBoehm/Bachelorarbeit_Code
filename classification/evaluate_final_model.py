import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, classification_report
import os
import argparse

def evaluateFinalModel(model_path, model_name, evaluate_average):
    df_train = pd.read_csv('./data/report_features_std_train.csv', sep='\t')
    X_train = None
    if evaluate_average:
        X_train = df_train.drop(['CIK', 'Ticker', 'Company', 'Filing_Date', 'Form_Type', 'Change_Ratio', 'Change_Nominal', 'File_Path'], axis=1).values[:, :768]
    else:
        X_train = df_train.drop(['CIK', 'Ticker', 'Company', 'Filing_Date', 'Form_Type', 'Change_Ratio', 'Change_Nominal', 'File_Path'], axis=1).values
        
    _, y_train = np.unique(df_train['Change_Nominal'].values, return_inverse=True)

    df_test = pd.read_csv('./data/report_features_std_test.csv', sep='\t')
    X_test = None
    if evaluate_average:
        X_test = df_test.drop(['CIK', 'Ticker', 'Company', 'Filing_Date', 'Form_Type', 'Change_Ratio', 'Change_Nominal', 'File_Path'], axis=1).values[:, :768]
    else:
        X_test = df_test.drop(['CIK', 'Ticker', 'Company', 'Filing_Date', 'Form_Type', 'Change_Ratio', 'Change_Nominal', 'File_Path'], axis=1).values
    _, y_test = np.unique(df_test['Change_Nominal'].values, return_inverse=True)

    scaler = StandardScaler()
    scaler.fit(X_train)
    X_test = scaler.transform(X_test)

    fid = open(model_path, 'rb')
    model = pickle.load(fid)

    if not os.path.exists('./data/model_evaluation'):
        os.makedirs('./data/model_evaluation')

    # Creating classification report
    print('Creating classification report...')
    y_test_pred = model.predict(X_test)
    report = classification_report(y_test, y_test_pred, digits=4)
    classification_report_file = open(f'./data/model_evaluation/{model_name}_classification_report.txt', 'w+')
    classification_report_file.write(str(report))
    classification_report_file.close()

    # Creating confusion matrix
    print('Create confusion matrix...')
    matrix = confusion_matrix(y_test, y_test_pred)
    confusion_matrix_file = open(f'./data/model_evaluation/{model_name}_confusion_matrix.txt', 'w+')
    confusion_matrix_file.write(str(list(matrix)))
    confusion_matrix_file.close()

    # Creating ROC-Curve data
    print('Create data for ROC-Curve...')
    y_test_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_test_proba)

    roc_curve_output = {
        'fpr': fpr,
        'tpr': tpr,
        'thresholds': thresholds
    }

    roc_curve_output_df = pd.DataFrame(roc_curve_output)
    roc_curve_output_df.to_csv(f'./data/model_evaluation/{model_name}_roc_curve.csv')

    auc_score = roc_auc_score(y_test, y_test_proba)
    auc_file = open(f'./data/model_evaluation/{model_name}_auc.txt', 'w+')
    auc_file.write(str(auc_score))
    auc_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Select model path and name.')
    parser.add_argument('--model_path', dest='model_path', type=str)
    parser.add_argument('--model_name', dest='model_name', type=str)
    parser.add_argument('--evaluate_average', dest='evaluate_average', action='store_true')
    args = parser.parse_args()
    evaluateFinalModel(args.model_path, args.model_name, args.evaluate_average)
