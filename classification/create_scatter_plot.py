import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import NeighborhoodComponentsAnalysis
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import argparse


# Creates a scatter plot for either the sequence embeddings or the document embeddings
# For both a PCA is applied first

def createSequenceScatterPlot(algorithm):
    df_index = pd.read_csv('./data/multiline_report_features_index.csv', sep="\t")

    index_list = df_index.to_dict('records')

    all_report_features_labeled = []
    for index_entry in index_list:
        print(f"Reading in {index_entry['File_Path']}...")
        features = list(pd.read_csv(index_entry["File_Path"], sep="\t").values)
        single_report_features_labeled = [{"label": index_entry["Change_Nominal"], "sequence_features": sequence_features } for sequence_features in features]
        all_report_features_labeled.extend(single_report_features_labeled)

    X = np.array(list(map(lambda report: report["sequence_features"], all_report_features_labeled)))
    y = np.array(list(map(lambda report: report["label"], all_report_features_labeled)))

    print("Scaling points")
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_transformed = []
    if algorithm == 'PCA':
        print("Running PCA")
        p = PCA(n_components = 2)
        p.fit(X)
        X_transformed = p.transform(X)
    else:
        print("Running NCA")
        l = NeighborhoodComponentsAnalysis(n_components = 2)
        _, y_int = np.unique(y, return_inverse=True)
        l.fit(X, y_int)
        X_transformed = l.transform(X)


    matplotlib.rcParams.update({'font.size': 16})   
    for activity in np.unique(y):
        X_transformed_filtered = X_transformed[y == activity, :]
        plt.scatter(X_transformed_filtered[:, 0], X_transformed_filtered[:, 1], label = activity, s = 2)

    num_sequences = len(list(y))

    f = open('./data/sequence_count.txt', 'w+')
    f.write(str(num_sequences))
    f.close()
    
    plt.legend()

    plt.savefig(f'./data/{algorithm}_scatter_sequence_level.png', dpi=500, bbox_inches='tight')

def createDocumentScatterPlot(algorithm):
    df_features = pd.read_csv('./data/report_features_with_std.csv', sep="\t")

    X = df_features.drop(['CIK', 'Ticker', 'Company', 'Filing_Date', 'Form_Type', 'Change_Ratio', 'Change_Nominal', 'File_Path'], axis=1).values[:, :768]
    y = df_features['Change_Nominal'].values

    print("Scaling points")
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_transformed = []
    if algorithm == 'PCA':
        print("Running PCA")
        p = PCA(n_components = 2)
        p.fit(X)
        X_transformed = p.transform(X)
    else:
        print("Running NCA")
        l = NeighborhoodComponentsAnalysis(n_components = 2)
        _, y_int = np.unique(y, return_inverse=True)
        l.fit(X, y)
        X_transformed = l.transform(X)


    matplotlib.rcParams.update({'font.size': 16})   
    for activity in np.unique(y):
        X_transformed_filtered = X_transformed[y == activity, :]
        plt.scatter(X_transformed_filtered[:, 0], X_transformed_filtered[:, 1], label = activity, s = 2)

    plt.legend()

    plt.savefig(f'./data/{algorithm}_scatter_document_level.png', dpi=500, bbox_inches='tight')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Settings for the scatter plot.')
    parser.add_argument('--level', dest='level', type=str)
    parser.add_argument('--algorithm', dest='algorithm', type=str)
    args = parser.parse_args()
    if args.level == 'sequence':
        createSequenceScatterPlot(args.algorithm)
    elif args.level == "document":
        createDocumentScatterPlot(args.algorithm)




