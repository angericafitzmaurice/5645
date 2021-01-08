import sys
import argparse
from time import time
import string
import re

import argparse as argparse
from pyspark import SparkContext
from operator import add


def count_sentences(rdd):
    """ Count the sentences in a file.

    Input:
    - rdd: an RDD containing the contents of a file, with one sentence in each element.

    Return value: The total number of sentences in the file.
    """
    # read the data and split into sentences
    sentencerdd = rdd.flatMap(lambda line: line.split('.'))
    # filter out the empty elements
    new_senrdd = sentencerdd.filter(lambda element: len(element) > 0)

    return new_senrdd.count()


def count_words(rdd):
    """ Count the number of words in a file.

    Input:
    - rdd: an RDD containing the contents of a file, with one sentence in each element.

    
    Return value: The total number of words in the file.
    """
    # read the data and split into words
    word = rdd.flatMap(lambda w: w.split(' '))
    # filter out the empty elements
    new_wordrdd = word.filter(lambda element: len(element) > 0)

    return new_wordrdd.count()


def compute_counts(rdd, numPartitions=10):
    """ Produce an rdd that contains the number of occurences of each word in a file.

    Each word in the file is converted to lowercase and then stripped of leading and trailing non-alphabetic
    characters before its occurences are counted.

    Input:
    - rdd: an RDD containing the contents of a file, with one sentence in each element.


    Return value: an RDD containing pairs of the form (word,count), where word is is a lowercase string,
    without leading or trailing non-alphabetic characters, and count is the number of times it appears
    in the file. The returned RDD should have a number of partitions given by numPartitions.

    """
    #get number partition
    numPartitions = args.N
    print(numPartitions)

    # converted to lowercase and remove leading or trailing non-alphabetic chars
    converted_rdd = rdd.flatMap(lambda strip: strip_non_alpha(strip.lower().split(' ')))
    # filter out the empty elements
    newrdd = converted_rdd.filter(lambda element: len(element) > 0)
    # create a new rdd that contains pairs of (word, value)
    countrdd = newrdd.map(lambda w: (w, 1)).reduceByKey(add)

    return countrdd


def count_difficult_words(counts, easy_list):
    """ Count the number of difficult words in a file.

    Input:
    - counts: an RDD containing pairs of the form (word,count), where word is a lowercase string, 
    without leading or trailing non-alphabetic characters, and count is the number of times this word appears
    in the file.
    - easy_list: a list of words deemed 'easy'.


    Return value: the total number of 'difficult' words in the file represented by RDD counts. 

    A word should be considered difficult if is not the 'same' as a word in easy_list. Two words are the same
    if one is the inflection of the other, when ignoring cases and leading/trailing non-alphabetic characters. 
    """
    # contains one word each element
    split_easyrdd = easy_list.flatMap(lambda easy: easy.lower().split(' '))
    # filter out the empty elements
    ezrdd = split_easyrdd.filter(lambda element: len(element) > 0)
    newezrdd = set(ezrdd.collect())
    dffrdd = counts.filter(lambda w: w[0] not in newezrdd).count()

    return dffrdd


def strip_non_alpha(alpha):
    """ Remove non-alphabetic characters from the beginning and end of a string.

    E.g. ',1what?!"' should become 'what'. Non-alphabetic characters in the middle
    of the string should not be removed. E.g. "haven't" should remain unaltered."""

    global beg_str, end_str
    if not alpha:
        return alpha
    for beg_str, element in enumerate(alpha):
        if element.isalpha():
            break
    for end_str, element in enumerate(alpha[::-1]):
        if element.isalpha():
            break

    return alpha[beg_str:len(alpha) - end_str]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Text Analysis via the Dale Chall Formula',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('mode', help='Mode of operation', choices=['SEN', 'WRD', 'UNQ', 'TOP20', 'DFF', 'DCF'])
    parser.add_argument('input',
                        help='Text file to be processed. This file contains text over several lines, with each line '
                             'corresponding to a different sentence.')
    parser.add_argument('--master', default="local[20]", help="Spark Master")
    parser.add_argument('--N', type=int, default=20,
                        help="Number of partitions to be used in RDDs containing word counts.")
    parser.add_argument('--simple_words', default="DaleChallEasyWordList.txt",
                        help="File containing Dale Chall simple word list. Each word appears in one line.")
    args = parser.parse_args()

    sc = SparkContext(args.master, 'Text Analysis')
    sc.setLogLevel('warn')

    start = time()

    # Add your code here
    # create rdd and read a file
    myrdd = sc.textFile('Data/Books/WarAndPeace.txt')
    # easy list
    easyrdd = sc.textFile('DaleChallEasyWordList.txt')

    if args.mode == 'SEN':
        sen = count_sentences(myrdd)
        print(f'{sen} sentences')

    elif args.mode == 'WRD':
        wrd = count_words(myrdd)
        print(f'{wrd} words')

    elif args.mode == 'UNQ':
        unqrdd = compute_counts(myrdd, args.N)
        print(unqrdd.collect())

    elif args.mode == 'TOP20':
        countrdd = compute_counts(myrdd, args.N)
        # print top 20 most frequent words
        unqrdd20 = countrdd.map(lambda w: (w[1], w[0])).sortByKey(False).take(20)
        print(unqrdd20)

    elif args.mode == 'DFF':
        total_unq = compute_counts(myrdd, args.N)
        diffrdd = count_difficult_words(total_unq, easyrdd)
        print(f'{diffrdd}  difficult words')

    elif args.mode == 'DCF':
        # calculates the Dale-Chall formula
        total_words = count_words(myrdd)
        total_sen = count_sentences(myrdd)
        count_uniq = compute_counts(myrdd, args.N)
        total_diff_words = count_difficult_words(count_uniq, easyrdd)

        words_per_sen = total_words / total_sen
        percent_dff = (total_diff_words / total_words) * 100
        raw_score = 0.1579 * percent_dff + 0.0496 * words_per_sen
        print(f'The Dale-Chall Formula is {raw_score}')

    end = time()
    print('Total execution time:', str(end - start) + 'sec')
