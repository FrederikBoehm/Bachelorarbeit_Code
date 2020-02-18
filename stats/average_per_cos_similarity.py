import pandas as pd
import numpy as np

def averagePerCosSimilarity():
    df = pd.read_csv('./data/changes_per_company.csv', sep='\t')

    cos_similarities = {}
    for entry in df.to_dict('records'):
        cos_similarity = round(entry["cosine_similarity_train_test"], 3)
        accuracy_nb = entry["nb_accuracy"]
        accuracy_svm = entry["svm_accuracy"]
        accuracy_knn = entry["knn_accuracy"]

        if not cos_similarity in cos_similarities:
            cos_similarities[cos_similarity] = {
                "accuracies_nb": [],
                "accuracies_svm": [],
                "accuracies_knn": []
            }

        cos_similarities[cos_similarity]["accuracies_nb"].append(accuracy_nb)
        cos_similarities[cos_similarity]["accuracies_svm"].append(accuracy_svm)
        cos_similarities[cos_similarity]["accuracies_knn"].append(accuracy_knn)

    output = {
        "cos_similarity": [],
        "accuracy_nb_avg": [],
        "accuracy_nb_std": [],
        "accuracy_svm_avg": [],
        "accuracy_svm_std": [],
        "accuracy_knn_avg": [],
        "accuracy_knn_std": []
    }
    for cos_similarity, accuracies in cos_similarities.items():
        output["cos_similarity"].append(cos_similarity)
        output["accuracy_nb_avg"].append(np.average(accuracies["accuracies_nb"]))
        output["accuracy_nb_std"].append(np.std(accuracies["accuracies_nb"]))
        output["accuracy_svm_avg"].append(np.average(accuracies["accuracies_svm"]))
        output["accuracy_svm_std"].append(np.std(accuracies["accuracies_svm"]))
        output["accuracy_knn_avg"].append(np.average(accuracies["accuracies_knn"]))
        output["accuracy_knn_std"].append(np.std(accuracies["accuracies_knn"]))

    output_df = pd.DataFrame(output)
    output_df.to_csv('./data/accuracy_per_cos_similarity.csv', sep="\t", index=False)

if __name__ == "__main__":
    averagePerCosSimilarity()