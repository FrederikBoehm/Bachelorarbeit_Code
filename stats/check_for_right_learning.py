import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
import numpy as np

# Should check whether the classifier bases its decision on the filing company
# Background: Some companies might always perform good in the training dataset,
# therefore the classifier learns that and predicts during test for that company always "positive"

def checkForRightLearning():
    print('Reading train file...')
    df_train = pd.read_csv('./data/report_features_std_train.csv', sep='\t')
    train_data = df_train.to_dict('records')
    
    print('Reading test file...')
    df_test = pd.read_csv('./data/report_features_std_test.csv', sep='\t')
    test_data = df_test.to_dict('records')
    
    changes_per_company = _initChangesPerCompany(train_data, test_data)
    changes_per_company = _getTrainingChanges(train_data, changes_per_company)
    changes_per_company = _getTestChanges(test_data, changes_per_company)
    changes_per_company = _getPredictions(df_train, df_test, changes_per_company, 'nb', './data/naive_bayes_avg.pkl')
    changes_per_company = _getPredictions(df_train, df_test, changes_per_company, 'knn', './data/knn_avg.pkl')
    changes_per_company = _getPredictions(df_train, df_test, changes_per_company, 'svm', './data/svm_avg.pkl')
    output_df = _convertToDataframe(changes_per_company)

    print('Writing to output file')
    output_df.to_csv('./data/changes_per_company.csv', sep='\t', index=False)

def _initChangesPerCompany(train_data, test_data):

    init_value = lambda entry: {
        'ticker': entry['Ticker'],
        'name': entry['Company'],
        'train_positives': 0,
        'train_negatives': 0,
        'test_positives': 0,
        'test_negatives': 0,
        'nb_test_positives': 0,
        'nb_test_negatives': 0,
        'nb_test_pred_positives': 0,
        'nb_test_pred_negatives': 0,
        'knn_test_positives': 0,
        'knn_test_negatives': 0,
        'knn_test_pred_positives': 0,
        'knn_test_pred_negatives': 0,
        'svm_test_positives': 0,
        'svm_test_negatives': 0,
        'svm_test_pred_positives': 0,
        'svm_test_pred_negatives': 0
    }

    changes_per_company = {}
    for entry in train_data:
        cik = str(entry['CIK'])
        if not cik in changes_per_company:
            changes_per_company[f'{cik}'] = init_value(entry)

    for entry in test_data:
        cik = str(entry['CIK'])
        if not cik in changes_per_company:
            changes_per_company[f'{cik}'] = init_value(entry)

    return changes_per_company

def _getTrainingChanges(train_data, changes_per_company):

    # X_train = df_train.drop(['CIK', 'Ticker', 'Company', 'Filing_Date', 'Form_Type', 'Change_Ratio', 'Change_Nominal', 'File_Path'], axis=1).values
    print('Counting positives and negatives in train dataset...')
    for entry in train_data:
        cik = str(entry['CIK'])
        if entry['Change_Nominal'] == 'positive':
            changes_per_company[f'{cik}']['train_positives'] += 1
        else:
            changes_per_company[f'{cik}']['train_negatives'] += 1

    return changes_per_company

def _getTestChanges(test_data, changes_per_company):

    # X_train = df_train.drop(['CIK', 'Ticker', 'Company', 'Filing_Date', 'Form_Type', 'Change_Ratio', 'Change_Nominal', 'File_Path'], axis=1).values
    print('Counting positives and negatives in test dataset...')
    for entry in test_data:
        cik = str(entry['CIK'])
        if entry['Change_Nominal'] == 'positive':
            changes_per_company[f'{cik}']['test_positives'] += 1
        else:
            changes_per_company[f'{cik}']['test_negatives'] += 1

    return changes_per_company

def _getPredictions(df_train, df_test, changes_per_company, model_type, model_path):
    print(f'Getting predictions for {model_type} at {model_path}...')

    X_train = df_train.drop(['CIK', 'Ticker', 'Company', 'Filing_Date', 'Form_Type', 'Change_Ratio', 'Change_Nominal', 'File_Path'], axis=1).values[:, :768]
    X_test = df_test.drop(['CIK', 'Ticker', 'Company', 'Filing_Date', 'Form_Type', 'Change_Ratio', 'Change_Nominal', 'File_Path'], axis=1).values[:, :768]
    ciks = df_test['CIK'].values
    labels = df_test['Change_Nominal'].values

    scaler = StandardScaler()
    scaler.fit(X_train)
    X_test = scaler.transform(X_test)
    
    fid = open(model_path, 'rb')
    model = pickle.load(fid)

    accuracy_per_company = {}
    for i, cik in enumerate(ciks):
        cik = str(cik)
        label = labels[i]
        X = X_test[i]
        prediction = model.predict(np.reshape(X, (1, -1)))[0]

        if prediction == 1:
            changes_per_company[f'{cik}'][f'{model_type}_test_pred_positives'] += 1
        else:
            changes_per_company[f'{cik}'][f'{model_type}_test_pred_negatives'] += 1

        if not cik in accuracy_per_company:
            accuracy_per_company[f'{cik}'] = {
                'correct': 0,
                'wrong': 0
            }

        if prediction == 1 and label == 'positive':
            accuracy_per_company[f'{cik}']['correct'] += 1
        else:
            accuracy_per_company[f'{cik}']['wrong'] += 1

    for cik in accuracy_per_company.keys():
        accuracy = accuracy_per_company[f'{cik}']['correct'] / (accuracy_per_company[f'{cik}']['correct'] + accuracy_per_company[f'{cik}']['wrong'])
        changes_per_company[f'{cik}'][f'{model_type}_accuracy'] = accuracy

    return changes_per_company

def _convertToDataframe(changes_per_company):
    print('Converting back to dataframe...')
    changes_per_company_list = []
    for cik in changes_per_company.keys():
        company = changes_per_company[f'{cik}']
        train_positive_ratio = company['train_positives'] / (company['train_positives'] + company['train_negatives'])
        train = np.array([company['train_positives'], company['train_negatives']])
        test = np.array([company['test_positives'], company['test_negatives']])
        norm_train = np.linalg.norm(train)
        norm_test = np.linalg.norm(test)

        cosine_similarity = None
        if norm_train != 0 and norm_test != 0:
            cosine_similarity = np.dot(train, test) / (norm_train * norm_test)
        changes_per_company_list.append({
            'cik': cik,
            'ticker': company['ticker'],
            'name': company['name'],
            'train_positives': company['train_positives'],
            'train_negatives': company['train_negatives'],
            'train_positive_ratio': train_positive_ratio,
            'test_positives': company['test_positives'],
            'test_negatives': company['test_negatives'],
            'cosine_similarity_train_test': cosine_similarity,
            'nb_test_pred_positives': company['nb_test_pred_positives'],
            'nb_test_pred_negatives': company['nb_test_pred_negatives'],
            'nb_accuracy': company['nb_accuracy'] if 'nb_accuracy' in company else None,
            'knn_test_pred_positives': company['knn_test_pred_positives'],
            'knn_test_pred_negatives': company['knn_test_pred_negatives'],
            'knn_accuracy': company['knn_accuracy'] if 'knn_accuracy' in company else None,
            'svm_test_pred_positives': company['svm_test_pred_positives'],
            'svm_test_pred_negatives': company['svm_test_pred_negatives'],
            'svm_accuracy': company['svm_accuracy'] if 'svm_accuracy' in company else None
        })
    changes_per_company_list.sort(key=lambda company_entry: company_entry['train_positive_ratio'], reverse=True)
    output_df = pd.DataFrame(changes_per_company_list, columns=[
        'cik',
        'ticker',
        'name',
        'train_positives',
        'train_negatives',
        'train_positive_ratio',
        'test_positives',
        'test_negatives',
        'cosine_similarity_train_test',
        'nb_test_pred_positives',
        'nb_test_pred_negatives',
        'nb_accuracy',
        'knn_test_pred_positives',
        'knn_test_pred_negatives',
        'knn_accuracy',
        'svm_test_pred_positives',
        'svm_test_pred_negatives',
        'svm_accuracy'
    ])

    return output_df


if __name__ == "__main__":
    checkForRightLearning()