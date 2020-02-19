import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

# Plots a scatter plot for the data generated with check_for_right_learning
# Shows for each company the cossine similarity between train and test data and the achieved accuracy

def plotCosSimilarityAccuracy():
    df = pd.read_csv('./data/changes_per_company.csv', sep="\t")

    df = df.dropna()

    matplotlib.rcParams.update({'font.size': 14})

    plt.scatter(df['cosine_similarity_train_test'], df['svm_accuracy'], s=4, label="Classification accuracy\nper train/test similarity")


    plt.xlabel("Cosinus similarity train/test per company")
    plt.ylabel("Classification accuracy")


    plt.legend()
    plt.savefig(f'./data/accuracy_per_cos_similarity.png', dpi=500, bbox_inches='tight')

if __name__ == "__main__":
    plotCosSimilarityAccuracy()