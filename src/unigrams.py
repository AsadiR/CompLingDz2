from functools import reduce
import math


fname = 'CrimeAndPunishment.txt'
out_file = 'results/unigram_pr_output'


def frange(start, end=None, inc=None):
    if end == None:
        end = start + 0.0
        start = 0.0
    if inc == None:
        inc = 1.0
    L = []
    while 1:
        next = start + len(L) * inc
        if inc > 0 and next >= end:
            break
        elif inc < 0 and next <= end:
            break
        L.append(next)
    return L


def read_file(name):
    with open(name) as f:
        content = f.readlines()

    text = ' '.join(content)
    text = text.lower()

    sym_to_delete = ':,.-?!;№#()[]«»„“…<>—\n'
    for sym in sym_to_delete:
        text = text.replace(sym, '')

    words = text.split(' ')
    words = list(filter(''.__ne__, words))
    return words


def split_proportionally(words, *args):
    s = sum(args)
    if s != 100:
        print("Sum of args must be 100!")
    list = []
    n = len(words)
    beginning = 0
    for p in args:
        ending = beginning+round(n*p/100)
        list.append(words[beginning:ending])
        beginning = ending

    return list


def get_dict(words):
    d = {}
    for word in words:
        if word in d:
            d[word] += 1
        else:
            d[word] = 1
    return d


def count_all_words(dict):
    sum = 0
    for key in dict:
        sum += dict[key]
    return sum


def count_word_types(dict):
    return len(dict)


def calculate_perplexity(test_words, d):
    N = len(test_words)
    # s = reduce((lambda x, y: x+math.log2(d[y])), [0] + test_words)
    s = 0
    for w in test_words:
        s += math.log2(d[w])
    return 2**(-s/N)

def write_list_in_file(l, name):
    with open(name, 'w') as f:
        for elem in l:
            f.write(str(elem[0]) + ' = ' + str(elem[1]) + '\n')


'''
Для каждой униграммы из обучающего корпуса нужно вычислить вероятность по правилу Линдстоуна.
При этом будем полагать что словарь униграмм имеет типы слов, которые встречаются во всем произведении,
т.е. во всех трех корпусах. По этой причине при вычислении вероятностей могут встретиться униграммы с нулевой частотой,
однако ввиду сглаживания вероятность их появления не будет нулевой.
'''
def get_prob_dict(d, d_for_learning, param_lambda, number_of_types):
    """
    :param d: словарь частот для всех слов из произведения
    :param d_for_learning: словарь частот для обучения
    :param param_lambda: параметр лямбда
    :param number_of_types: количество типов n-грамм
    :return: словарь вероятностей
    """

    N = len(d_for_learning)
    d_of_p = {}
    for key in d:
        if key in d_for_learning:
            f = d_for_learning[key]
        else:
            f = 0
        p = (f + param_lambda) / (N + number_of_types * param_lambda)
        d_of_p[key] = p
    return d_of_p


def find_the_best_param(d, d_for_learning, held_out_words, number_of_types, rb, re, st):
    cur_param_max = 0
    cur_max = -math.inf
    N = len(held_out_words)
    for param_lambda in frange(rb, re, st):
        d_of_p = get_prob_dict(d, d_for_learning, param_lambda, number_of_types)
        sum_of_logs = 0
        for w in held_out_words:
            sum_of_logs += math.log2(d_of_p[w])
        # print(sum_of_logs, ' ', param_lambda)
        if cur_max < sum_of_logs:
            cur_max = sum_of_logs
            cur_param_max = param_lambda

    return cur_param_max


if __name__ == "__main__":
    '''
    Считываем данные о произведении из файла
    '''

    words = read_file(fname)
    parts = split_proportionally(words, 80, 10, 10)
    d = get_dict(words)
    d_for_learning = get_dict(parts[0])
    d_hold_out = get_dict(parts[1])
    d_for_test = get_dict(parts[2])
    number_of_types = count_word_types(d)

    '''
    Теперь подберем параметр так на специально выделенном корпусе.
    '''
    held_out_words = parts[1]
    param = find_the_best_param(d, d_for_learning, held_out_words, number_of_types, 0.01, 0.3, 0.01)
    d_of_p =  get_prob_dict(d, d_for_learning, param, number_of_types)

    '''
    Записываем словарь вероятностей по Линдстоуну в файл
    '''

    sorted_list = sorted(d_of_p.items(), key=lambda x: x[1])[::-1]
    write_list_in_file(sorted_list, out_file)

    '''
    Теперь необходимо вычислить перплексию.
    Для каждой униграммы из текстового корпуса находим ее вероятность в словаре и подставляем в формулу.
    '''

    test_words = parts[2]
    print('types = ', number_of_types)
    print('best_param = ', param)
    print('perplexity = ', calculate_perplexity(test_words, d_of_p))











