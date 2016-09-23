from lindstoun_unigrams import fname
from lindstoun_unigrams import read_file
from lindstoun_unigrams import split_proportionally
from lindstoun_unigrams import count_all_ngrams
from lindstoun_bigrams import get_dict_for_bigrams
from lindstoun_unigrams import write_list_in_file
from lindstoun_unigrams import learning_cs
from lindstoun_unigrams import hold_out_cs
from lindstoun_unigrams import test_cs

out_file_test = 'results/bigrams/real_test_bigram_pr_output'
out_file_learning = 'results/bigrams/real_learning_bigram_pr_output'

words = read_file(fname)
parts = split_proportionally(words, learning_cs, hold_out_cs, test_cs)
d_for_test = get_dict_for_bigrams(parts[2])
d_for_learning = get_dict_for_bigrams(parts[0])


N = count_all_ngrams(d_for_learning)
f_for_pr = (lambda t: (t[0], t[1]/N))
temp = sorted(d_for_learning.items(), key=lambda x: x[1])[::-1]
sorted_list = map(f_for_pr, temp)
write_list_in_file(sorted_list, out_file_learning)

N = count_all_ngrams(d_for_test)
f_for_pr = (lambda t: (t[0], t[1]/N))
sorted_list = map(f_for_pr, sorted(d_for_test.items(), key=lambda x: x[1])[::-1])
write_list_in_file(sorted_list, out_file_test)