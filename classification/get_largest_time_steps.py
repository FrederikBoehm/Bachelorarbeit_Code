import pandas as pd

def getLargestTimeSteps():
    index_df = pd.read_csv('./data/multiline_report_features_train.csv', sep='\t')

    file_paths = index_df["File_Path"]

    max_length = 0
    max_length_path = ""

    lengths = {}

    for file_path in file_paths:
        report_df = pd.read_csv(file_path, sep="\t")
        if len(report_df.index) > max_length:
            max_length = len(report_df.index)
            max_length_path = file_path

            print(f"New max length of {max_length} found in {max_length_path}")

        if not len(report_df.index) in lengths:
            lengths[len(report_df.index)] = 1
        else:
            lengths[len(report_df.index)] += 1

    lengths = list(lengths.items())
    lengths = {
        "length": list(map(lambda dict_entry: dict_entry[0], lengths)),
        "occurences": list(map(lambda dict_entry: dict_entry[1], lengths))
    }
    output_df = pd.DataFrame(lengths)
    output_df.to_csv("./data/timesteps_distribution.csv")

    output_file = open("./data/max_timesteps.txt", "w+")
    output_file.write(f"Max length: {max_length}, Path: {max_length_path}")
    output_file.close()

if __name__ == "__main__":
    getLargestTimeSteps()

    