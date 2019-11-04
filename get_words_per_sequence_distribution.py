import glob
import re
import pandas as pd
import os

def getWordsPerSequenceDistribution():

    file_paths = glob.glob('./data/multiline_reports/multiline_report*')
    distribution = {}
    sequences_in_class = {}
    smaller_than_six = []
    larger_than_128 = []
    for file_path in file_paths:
        print(f'Processing {file_path}')
        with open(file_path) as f:
            lines = [line.rstrip('\n') for line in f]
            for line in lines:
                number_of_words = len(re.findall(r'\w+', line))
                if number_of_words in distribution:
                    current_number = distribution[number_of_words]
                    distribution[number_of_words] = current_number + 1
                    sequences_in_class[number_of_words].append(line)
                else:
                    distribution[number_of_words] = 1
                    sequences_in_class[number_of_words] = [line]

                

                # if number_of_words < 6:
                #     smaller_than_six.append(line)
                # elif number_of_words > 128:
                #     larger_than_128.append(line)

    distribution = list(distribution.items())
    distribution.sort(key=lambda item: item[0])
    distribution_df = pd.DataFrame(distribution, columns=['words_in_sequence', 'appearances'])

    distribution_df.to_csv('./data/words_per_sequence_distribution.csv', sep='\t', index=False)

    # smaller_than_six_file = open('./data/smaller_than_six.txt', 'w')
    # smaller_than_six_file.write('\n'.join(smaller_than_six))
    # smaller_than_six_file.close()

    # larger_than_128_file = open('./data/larger_than_128.txt', 'w')
    # larger_than_128_file.write('\n'.join(larger_than_128))
    # larger_than_128_file.close()

    if not os.path.exists('./data/words_per_sequence_distribution'):
        os.makedirs('./data/words_per_sequence_distribution')

    for key in sequences_in_class.keys():
        file_content = '\n'.join(sequences_in_class[key])
        f = open(f'./data/words_per_sequence_distribution/seq_{key}.txt', 'w')
        f.write(file_content)
        f.close()

if __name__ == '__main__':
    getWordsPerSequenceDistribution()
    


        

