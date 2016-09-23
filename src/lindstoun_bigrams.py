from lindstoun_unigrams import fname
from lindstoun_unigrams import read_file
from lindstoun_unigrams import split_proportionally
from lindstoun_unigrams import count_ngram_types
from lindstoun_unigrams import write_dict_in_file
from lindstoun_unigrams import find_the_best_param
from lindstoun_unigrams import get_prob_dict
from lindstoun_unigrams import calculate_perplexity
from lindstoun_unigrams import learning_cs
from lindstoun_unigrams import hold_out_cs
from lindstoun_unigrams import test_cs

out_file = 'results/bigrams/lindstoun_bigram_pr_output'


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
    words = read_file(fname)
    parts = split_proportionally(words, learning_cs, hold_out_cs, test_cs)
    d = get_dict_for_bigrams(words)
    d_for_learning = get_dict_for_bigrams(parts[0])
    d_hold_out = get_dict_for_bigrams(parts[1])
    d_for_test = get_dict_for_bigrams(parts[2])
    number_of_types = count_ngram_types(d)

    held_out_bigrams = get_bigrams_list(parts[1])
    start = 0.1
    step = 0.1
    end = 1 + step
    #param = find_the_best_param(d, d_for_learning, held_out_bigrams, start, end, step)
    param = 1
    d_of_p = get_prob_dict(d, d_for_learning, param)

    write_dict_in_file(d_of_p, out_file)

    test_words = parts[2]
    print('types = ', number_of_types)
    print('best_param = ', param)
    print('perplexity = ', calculate_perplexity(get_bigrams_list(test_words), d_of_p))




