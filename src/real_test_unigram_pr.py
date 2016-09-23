from lindstoun_unigrams import fname
from lindstoun_unigrams import read_file
from lindstoun_unigrams import split_proportionally
from lindstoun_unigrams import count_all_ngrams
from lindstoun_unigrams import get_dict
from lindstoun_unigrams import write_list_in_file
from lindstoun_unigrams import learning_cs
from lindstoun_unigrams import hold_out_cs
from lindstoun_unigrams import test_cs

out_file_test = 'results/unigrams/real_test_unigram_pr_output'
out_file_learning = 'results/unigrams/real_learning_unigram_pr_output'

words = read_file(fname)
parts = split_proportionally(words, learning_cs, hold_out_cs, test_cs)
d = get_dict(words)
d_for_learning = get_dict(parts[0])
d_for_test = get_dict(parts[2])

N = count_all_ngrams(d_for_learning)
f_for_pr = (lambda t: (t[0], t[1]/N))
temp = sorted(d_for_learning.items(), key=lambda x: x[1])[::-1]
sorted_list = map(f_for_pr, temp)
write_list_in_file(sorted_list, out_file_learning)

N = count_all_ngrams(d_for_test)
f_for_pr = (lambda t: (t[0], t[1]/N))
sorted_list = map(f_for_pr, sorted(d_for_test.items(), key=lambda x: x[1])[::-1])
write_list_in_file(sorted_list, out_file_test)

print('type', len(d))
print('N', len(words))
