import re

def concatToMaxSequenceLength(lines, max_seq_length):

    max_line_length = int(max_seq_length * 0.9) # We only concat to about 90% of the max_seq_length, so that we don't get longer sequences than max_seq_length after tokenization
    line_index = 0
    concatenated_lines = []
    concat_index = 0
    while line_index < len(lines) - 1:

        word_regex = r"\w+"
        words_in_current_line = len(re.findall(word_regex, lines[line_index]))
        next_line_to_concat_index = line_index + 1
        words_in_next_line = len(re.findall(word_regex, lines[next_line_to_concat_index]))

        concatenated_lines.append(lines[line_index])
        while (next_line_to_concat_index < len(lines) - 1) and (words_in_current_line + words_in_next_line < max_line_length):
            concatenated_lines[concat_index] = concatenated_lines[concat_index] + ' ' + lines[next_line_to_concat_index]
            words_in_current_line = words_in_current_line + words_in_next_line

            next_line_to_concat_index += 1
            words_in_next_line = len(re.findall(word_regex, lines[next_line_to_concat_index]))
        
        line_index = next_line_to_concat_index + 1
        concat_index += 1

    return concatenated_lines