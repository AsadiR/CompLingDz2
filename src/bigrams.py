from unigrams import fname
from unigrams import read_file
from unigrams import split_proportionally
from unigrams import count_word_types
from unigrams import write_list_in_file
from unigrams import find_the_best_param
from unigrams import get_prob_dict
from unigrams import calculate_perplexity

out_file = 'results/bigram_pr_output'
def get_dict_for_bigrams(words):
    bigram_dict = {}
    for i in range(len(words) - 1):
        if (words[i], words[i + 1]) in bigram_dict:
            bigram_dict[(words[i], words[i + 1])] += 1
        else:
            bigram_dict[(words[i], words[i + 1])] = 1
    return bigram_dict

def get_bigrams_list(words):
    bigram_list = []
    for i in range(len(words) - 1):
        bigram_list.append((words[i], words[i + 1]))
    return bigram_list

if __name__ == "__main__":
    '''
    Считываем данные о произведении из файла
    '''

    words = read_file(fname)
    parts = split_proportionally(words, 80, 10, 10)
    d = get_dict_for_bigrams(words)
    d_for_learning = get_dict_for_bigrams(parts[0])
    d_hold_out = get_dict_for_bigrams(parts[1])
    d_for_test = get_dict_for_bigrams(parts[2])
    number_of_types = count_word_types(d)

    held_out_bigrams = get_bigrams_list(parts[1])
    param = find_the_best_param(d, d_for_learning, held_out_bigrams, number_of_types, 2, 3, 0.01)
    d_of_p = get_prob_dict(d, d_for_learning, param, number_of_types)

    sorted_list = sorted(d_of_p.items(), key=lambda x: x[1])[::-1]
    write_list_in_file(sorted_list, out_file)

    test_words = parts[2]
    print('types = ', number_of_types)
    print('best_param = ', param)
    print('perplexity = ', calculate_perplexity(get_bigrams_list(test_words), d_of_p))




