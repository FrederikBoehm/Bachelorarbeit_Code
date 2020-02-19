try:
    from concat_to_max_sequence_length import concatToMaxSequenceLength
except ImportError:
    from shared.concat_to_max_sequence_length import concatToMaxSequenceLength

import pandas as pd
import os
import argparse

# Prepares the data for feature extraction, by concatenating sentences until the max_sequence_length is reached

def createFeatureExtractionData(max_sequence_length):

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
    parser = argparse.ArgumentParser(description='Parameters to createFeatureExtractionData.')
    parser.add_argument('--max_sequence_length', dest='max_sequence_length', type=int)
    args = parser.parse_args()
    max_sequence_length = args.max_sequence_length
    if max_sequence_length:
        createFeatureExtractionData(max_sequence_length)
    else:
        print('Provide a value for max_sequence_length, e.g. 128 or 512')



