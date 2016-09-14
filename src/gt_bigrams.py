from unigrams import read_file
from unigrams import split_proportionally
from unigrams import count_ngram_types
from unigrams import calculate_perplexity
from bigrams import get_dict_for_bigrams
from unigrams import fname
from unigrams import write_dict_in_file
from gt_unigrams import get_gt_prob_dict
from bigrams import get_bigrams_list


out_file = 'results/gt_bigram_pr_output'

if __name__ == "__main__":
    words = read_file(fname)
    parts = split_proportionally(words, 60, 20, 20)
    d = get_dict_for_bigrams(words)
    d_for_learning = get_dict_for_bigrams(parts[0])
    d_hold_out = get_dict_for_bigrams(parts[1])
    d_for_test = get_dict_for_bigrams(parts[2])
    number_of_types = count_ngram_types(d)

    '''
    Необходимо посчитать кол-во слов, которые встречаются k раз.
    '''
    d_of_p = get_gt_prob_dict(d_for_learning, d)
    write_dict_in_file(d_of_p, out_file)

    test_words = parts[2]
    print('types = ', number_of_types)
    print('perplexity = ', calculate_perplexity(get_bigrams_list(test_words), d_of_p))


