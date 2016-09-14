from unigrams import fname
from unigrams import read_file
from unigrams import split_proportionally
from unigrams import count_all_ngrams
from bigrams import get_dict_for_bigrams
from unigrams import write_list_in_file

out_file = 'results/real_bigram_pr_output'

words = read_file(fname)
parts = split_proportionally(words, 60, 20, 20)
d_for_test = get_dict_for_bigrams(parts[2])
all_words_number = count_all_ngrams(d_for_test)

f_for_pr = (lambda t: (t[0], t[1]/all_words_number))
sorted_list = map(f_for_pr, sorted(d_for_test.items(), key=lambda x: x[1])[::-1])
write_list_in_file(sorted_list, out_file)