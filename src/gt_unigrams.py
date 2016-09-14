from unigrams import fname
from unigrams import read_file
from unigrams import split_proportionally
from unigrams import count_ngram_types
from unigrams import count_all_ngrams
from unigrams import get_dict
from unigrams import write_dict_in_file
from unigrams import calculate_perplexity

out_file = 'results/gt_unigram_pr_output'

def get_fr_dict(d):
    fr_dict = {}
    for key in d:
        if d[key] in fr_dict:
            fr_dict[d[key]] += 1
        else:
            fr_dict[d[key]] = 1
    return fr_dict


def get_elem_from_fr_dict(key, fr_dict):
    if key in fr_dict:
        return fr_dict[key]
    else:
        before = find_fr_before(key, fr_dict)
        after = find_fr_after(key, fr_dict)
        return (before + after)/2


def find_fr_before(key, d):
    while key > 1 and key not in d:
        key -= 1
    if key not in d: return 1
    else: return d[key]


def find_fr_after(key, d):
    while key < len(d) and key not in d:
        key += 1
    if key not in d: return 1
    else: return d[key]


def get_gt_prob_dict(d_for_learning, d):
    fr_dict = get_fr_dict(d_for_learning)
    d_of_p = {}
    N = count_all_ngrams(d_for_learning)
    for key in d:
        if key in d_for_learning:
            fr = d_for_learning[key]
        else:
            fr = 0
        n_cur = get_elem_from_fr_dict(fr, fr_dict)
        n_next = get_elem_from_fr_dict(fr + 1, fr_dict)
        d_of_p[key] = (fr+1)*n_next/(n_cur*N)
    return d_of_p

if __name__ == "__main__":
    words = read_file(fname)
    parts = split_proportionally(words, 80, 10, 10)
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


