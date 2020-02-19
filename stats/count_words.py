import re
import glob

# Count number of words in dataset

def countWords():

    reports_path = './data/multiline_reports/multiline_report*'
    file_paths = glob.glob(reports_path)

    word_count = 0
    for file_path in file_paths:
        f = open(file_path, 'r')
        report = f.read()
        f.close()
        regex = r"\w+"
        words_in_report = re.findall(regex, report)
        word_count += len(words_in_report)
        print(f'{file_path} has {len(words_in_report)} words')

    print(f'Total words: {word_count}')

    f = open('./data/word_count.txt', 'w')
    f.write(str(word_count))
    f.close()

if __name__ == '__main__':
    countWords()