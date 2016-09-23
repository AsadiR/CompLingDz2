from lindstoun_unigrams import read_file
from lindstoun_unigrams import split_proportionally
from lindstoun_unigrams import count_ngram_types
from lindstoun_unigrams import calculate_perplexity
from lindstoun_bigrams import get_dict_for_bigrams
from lindstoun_unigrams import fname
from lindstoun_unigrams import write_dict_in_file
from gt_unigrams import get_gt_prob_dict
from lindstoun_bigrams import get_bigrams_list
from lindstoun_unigrams import learning_cs
from lindstoun_unigrams import hold_out_cs
from lindstoun_unigrams import test_cs


out_file = 'results/bigrams/gt_bigram_pr_output'

if __name__ == "__main__":
    words = read_file(fname)
    parts = split_proportionally(words, learning_cs, hold_out_cs, test_cs)
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


