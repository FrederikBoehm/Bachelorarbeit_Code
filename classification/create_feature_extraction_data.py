try:
    from create_finetuning_data import concatToMaxSequenceLength
except ImportError:
    from finetuning.create_finetuning_data import concatToMaxSequenceLength

import pandas as pd
import os

def createFeatureExtractionData():

    max_sequence_length = 512

    df_multiline_report_index = pd.read_csv("./data/multiline_report_index.csv", sep="\t")
    multiline_report_index = df_multiline_report_index.to_dict('records')

    output_dir = f'./data/multiline_reports_{max_sequence_length}/'

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    for index_entry in multiline_report_index:
        file_path = index_entry["File_Path"]

        f = open(file_path)
        lines = [line.rstrip('\n') for line in f]
        f.close()

        lines = concatToMaxSequenceLength(lines, max_sequence_length)

        splitted_file_path = file_path.split('/')
        file_name = splitted_file_path[len(splitted_file_path) - 1]

        output_file_path = output_dir + file_name
        index_entry["File_Path"] = output_file_path


        f = open(output_file_path, 'w+')
        f.write("\n".join(lines))
        f.close()
        print(f'Wrote file to {output_file_path}')

    output_df = pd.DataFrame(multiline_report_index)

    output_df.to_csv('./data/multiline_report_index_feature_extraction.csv', sep='\t', index=False)

if __name__ == "__main__":
    createFeatureExtractionData()



