from lindstoun_unigrams import fname
from lindstoun_unigrams import read_file
from lindstoun_unigrams import split_proportionally
from lindstoun_unigrams import count_ngram_types
from lindstoun_unigrams import count_all_ngrams
from lindstoun_unigrams import get_dict
from lindstoun_unigrams import write_dict_in_file
from lindstoun_unigrams import calculate_perplexity
from lindstoun_unigrams import learning_cs
from lindstoun_unigrams import hold_out_cs
from lindstoun_unigrams import test_cs


out_file = 'results/unigrams/gt_unigram_pr_output'

def get_fr_dict(d):
    fr_dict = {}
    max_fr = 0
    for key in d:
        if d[key] in fr_dict:
            fr_dict[d[key]] += 1
        else:
            fr_dict[d[key]] = 1
        if max_fr < d[key]: max_fr = d[key]
    return fr_dict, max_fr


def get_nc(fr_dict, fr, max_fr):
    if fr in fr_dict:
        return fr_dict[fr]
    else:
        while fr < max_fr:
            fr += 1
            if fr in fr_dict:
                return fr_dict[fr]
        return 0.7



def get_gt_prob_dict(d_for_learning, d):
    fr_dict, max_fr = get_fr_dict(d_for_learning)
    d_of_p = {}
    number_of_types_in_d = count_ngram_types(d)
    N = count_all_ngrams(d_for_learning)
    dif = count_ngram_types(d) - count_ngram_types(d_for_learning)
    if dif == 0: dif = 1
    for_new = get_nc(fr_dict, 1, max_fr) / (dif*N)
    #d_of_p["#new"] = for_new
    for key in d:
        if key in d_for_learning:
            fr = d_for_learning[key]
            n_cur = get_nc(fr_dict, fr, max_fr)
            n_next = get_nc(fr_dict, fr + 1, max_fr)
            d_of_p[key] = (fr + 1) * n_next / (n_cur * N)
        else:
            d_of_p[key] = for_new

    return d_of_p

if __name__ == "__main__":
    words = read_file(fname)
    parts = split_proportionally(words, learning_cs, hold_out_cs, test_cs)
    d = get_dict(words)
    d_for_learning = get_dict(parts[0])
    d_hold_out = get_dict(parts[1])
    d_for_test = get_dict(parts[2])
    number_of_types = count_ngram_types(d)

    '''
    Необходимо посчитать кол-во слов, которые встречаются k раз.
    '''
    d_of_p = get_gt_prob_dict(d_for_learning, d)
    write_dict_in_file(d_of_p, out_file)

    test_words = parts[2]
    print('types = ', number_of_types)
    print('perplexity = ', calculate_perplexity(test_words, d_of_p))


