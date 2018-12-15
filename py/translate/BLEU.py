
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2017--2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You may not
# use this file except in compliance with the License. A copy of the License
# is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is distributed on
# an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.


import argparse
import gzip
import logging
import io
import os
import re
import sys
from collections import Counter, namedtuple
from itertools import zip_longest
from typing import List, Iterable, Tuple

import math
import unicodedata
NGRAM_ORDER = 4

logger = logging.getLogger(__name__)
logger.setLevel(level = logging.INFO)
info_handler = logging.FileHandler("BLEU.log")
info_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
info_handler.setFormatter(formatter)
logger.addHandler(info_handler)

#%%my log
def my_log(num):
    """
    Floors the log function
    :param num: the number
    :return: log(num) floored to a very low number
    """

    if num == 0.0:
        return -9999999999
    return math.log(num)

#%% tokenize functions, return a string sentence
def tokenize_13a(line):
    """
    Tokenizes an input line using a relatively minimal tokenization that is however equivalent to mteval-v13a, used by WMT.
    :param line: a segment to tokenize
    :return: the tokenized line
    """
    norm = line
    # language-independent part:
    norm = norm.replace('<skipped>', '')
    norm = norm.replace('-\n', '')
    norm = norm.replace('\n', ' ')
    norm = norm.replace('&quot;', '"')
    norm = norm.replace('&amp;', '&')
    norm = norm.replace('&lt;', '<')
    norm = norm.replace('&gt;', '>')

    # language-dependent part (assuming Western languages):
    norm = " {} ".format(norm)
    norm = re.sub(r'([\{-\~\[-\` -\&\(-\+\:-\@\/])', ' \\1 ', norm)
    norm = re.sub(r'([^0-9])([\.,])', '\\1 \\2 ', norm)  # tokenize period and comma unless preceded by a digit
    norm = re.sub(r'([\.,])([^0-9])', ' \\1 \\2', norm)  # tokenize period and comma unless followed by a digit
    norm = re.sub(r'([0-9])(-)', '\\1 \\2 ', norm)  # tokenize dash when preceded by a digit
    norm = re.sub(r'\s+', ' ', norm)  # one space only between words
    norm = re.sub(r'^\s+', '', norm)  # no leading space
    norm = re.sub(r'\s+$', '', norm)  # no trailing space
    return norm

def tokenize_v14_international(string):
    r"""Tokenize a string following the official BLEU implementation.
    See https://github.com/moses-smt/mosesdecoder/blob/master/scripts/generic/mteval-v14.pl#L954-L983
    In our case, the input string is expected to be just one line
    and no HTML entities de-escaping is needed.
    So we just tokenize on punctuation and symbols,
    except when a punctuation is preceded and followed by a digit
    (e.g. a comma/dot as a thousand/decimal separator).
    Note that a number (e.g., a year) followed by a dot at the end of sentence is NOT tokenized,
    i.e. the dot stays with the number because `s/(\p{P})(\P{N})/ $1 $2/g`
    does not match this case (unless we add a space after each sentence).
    However, this error is already in the original mteval-v14.pl
    and we want to be consistent with it.
    The error is not present in the non-international version,
    which uses `$norm_text = " $norm_text "` (or `norm = " {} ".format(norm)` in Python).
    :param string: the input string
    :return: a list of tokens
    """
    string = UnicodeRegex.nondigit_punct_re.sub(r'\1 \2 ', string)
    string = UnicodeRegex.punct_nondigit_re.sub(r' \1 \2', string)
    string = UnicodeRegex.symbol_re.sub(r' \1 ', string)
    return string.strip()

def tokenize_zh(sentence):
    """MIT License
    Copyright (c) 2017 - Shujian Huang <huangsj@nju.edu.cn>
    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:
    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.
    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
    The tokenization of Chinese text in this script contains two steps: separate each Chinese
    characters (by utf-8 encoding); tokenize the non Chinese part (following the mteval script).
    Author: Shujian Huang huangsj@nju.edu.cn
    :param sentence: input sentence
    :return: tokenized sentence
    """

    def is_chinese_char(uchar):
        """
        :param uchar: input char in unicode
        :return: whether the input char is a Chinese character.
        """
        if uchar >= u'\u3400' and uchar <= u'\u4db5':  # CJK Unified Ideographs Extension A, release 3.0
            return True
        elif uchar >= u'\u4e00' and uchar <= u'\u9fa5':  # CJK Unified Ideographs, release 1.1
            return True
        elif uchar >= u'\u9fa6' and uchar <= u'\u9fbb':  # CJK Unified Ideographs, release 4.1
            return True
        elif uchar >= u'\uf900' and uchar <= u'\ufa2d':  # CJK Compatibility Ideographs, release 1.1
            return True
        elif uchar >= u'\ufa30' and uchar <= u'\ufa6a':  # CJK Compatibility Ideographs, release 3.2
            return True
        elif uchar >= u'\ufa70' and uchar <= u'\ufad9':  # CJK Compatibility Ideographs, release 4.1
            return True
        elif uchar >= u'\u20000' and uchar <= u'\u2a6d6':  # CJK Unified Ideographs Extension B, release 3.1
            return True
        elif uchar >= u'\u2f800' and uchar <= u'\u2fa1d':  # CJK Compatibility Supplement, release 3.1
            return True
        elif uchar >= u'\uff00' and uchar <= u'\uffef':  # Full width ASCII, full width of English punctuation, half width Katakana, half wide half width kana, Korean alphabet
            return True
        elif uchar >= u'\u2e80' and uchar <= u'\u2eff':  # CJK Radicals Supplement
            return True
        elif uchar >= u'\u3000' and uchar <= u'\u303f':  # CJK punctuation mark
            return True
        elif uchar >= u'\u31c0' and uchar <= u'\u31ef':  # CJK stroke
            return True
        elif uchar >= u'\u2f00' and uchar <= u'\u2fdf':  # Kangxi Radicals
            return True
        elif uchar >= u'\u2ff0' and uchar <= u'\u2fff':  # Chinese character structure
            return True
        elif uchar >= u'\u3100' and uchar <= u'\u312f':  # Phonetic symbols
            return True
        elif uchar >= u'\u31a0' and uchar <= u'\u31bf':  # Phonetic symbols (Taiwanese and Hakka expansion)
            return True
        elif uchar >= u'\ufe10' and uchar <= u'\ufe1f':
            return True
        elif uchar >= u'\ufe30' and uchar <= u'\ufe4f':
            return True
        elif uchar >= u'\u2600' and uchar <= u'\u26ff':
            return True
        elif uchar >= u'\u2700' and uchar <= u'\u27bf':
            return True
        elif uchar >= u'\u3200' and uchar <= u'\u32ff':
            return True
        elif uchar >= u'\u3300' and uchar <= u'\u33ff':
            return True

        return False

    sentence = sentence.strip()
    sentence_in_chars = ""
    for char in sentence:
        if is_chinese_char(char):
            sentence_in_chars += " "
            sentence_in_chars += char
            sentence_in_chars += " "
        else:
            sentence_in_chars += char
    sentence = sentence_in_chars

    # tokenize punctuation
    sentence = re.sub(r'([\{-\~\[-\` -\&\(-\+\:-\@\/])', r' \1 ', sentence)

    # tokenize period and comma unless preceded by a digit
    sentence = re.sub(r'([^0-9])([\.,])', r'\1 \2 ', sentence)

    # tokenize period and comma unless followed by a digit
    sentence = re.sub(r'([\.,])([^0-9])', r' \1 \2', sentence)

    # tokenize dash when preceded by a digit
    sentence = re.sub(r'([0-9])(-)', r'\1 \2 ', sentence)

    # one space only between words
    sentence = re.sub(r'\s+', r' ', sentence)

    # no leading space
    sentence = re.sub(r'^\s+', r'', sentence)

    # no trailing space
    sentence = re.sub(r'\s+$', r'', sentence)

    return sentence

TOKENIZERS = {
    '13a': tokenize_13a,
    'intl': tokenize_v14_international,
    'zh': tokenize_zh,
    'none': lambda x: x,
}
DEFAULT_TOKENIZER = '13a'


#%%extract ngrams, return ONE dictionary containing ngrams and counts 

def extract_ngrams(line, min_order=1, max_order=NGRAM_ORDER) -> Counter:
    """Extracts all the ngrams (1 <= n <= NGRAM_ORDER) from a sequence of tokens.
    :param line: a segment containing a sequence of words
    :param max_order: collect n-grams from 1<=n<=max
    :return: a dictionary containing ngrams and counts
    """

    ngrams = Counter()
    tokens = line.split()
    for n in range(min_order, max_order + 1):
        for i in range(0, len(tokens) - n + 1):
            ngram = ' '.join(tokens[i: i + n])
            ngrams[ngram] += 1

    return ngrams

#%%find best ref, useless here
def ref_stats(output, refs):
    ngrams = Counter()
    closest_diff = None
    closest_len = None
    for ref in refs: #each sentence
        tokens = ref.split()
        reflen = len(tokens)
        diff = abs(len(output.split()) - reflen)
        if closest_diff is None or diff < closest_diff:
            closest_diff = diff
            closest_len = reflen
        elif diff == closest_diff:
            if reflen < closest_len:
                closest_len = reflen

        ngrams_ref = extract_ngrams(ref)
        for ngram in ngrams_ref.keys():
            ngrams[ngram] = max(ngrams[ngram], ngrams_ref[ngram])

    return ngrams, closest_diff, closest_len

#%%compute bleu
BLEU = namedtuple('BLEU', 'score, counts, totals, precisions, bp, sys_len, ref_len')

def compute_bleu(correct: List[int], total: List[int], sys_len: int, ref_len: int, smooth = 'none', smooth_floor = 0.01,
                 use_effective_order = False) -> BLEU:
    """Computes BLEU score from its sufficient statistics. Adds smoothing.
    :param correct: List of counts of correct ngrams, 1 <= n <= NGRAM_ORDER
    :param total: List of counts of total ngrams, 1 <= n <= NGRAM_ORDER
    :param sys_len: The cumulative system length
    :param ref_len: The cumulative reference length
    :param smooth: The smoothing method to use
    :param smooth_floor: The smoothing value added, if smooth method 'floor' is used
    :param use_effective_order: Use effective order.
    :return: A BLEU object with the score (100-based) and other statistics.
    """

    precisions = [0 for x in range(NGRAM_ORDER)]

    smooth_mteval = 1.
    effective_order = NGRAM_ORDER
    for n in range(NGRAM_ORDER):
        if total[n] == 0:
            break

        if use_effective_order:
            effective_order = n + 1

        if correct[n] == 0:
            if smooth == 'exp':
                smooth_mteval *= 2
                precisions[n] = 100. / (smooth_mteval * total[n])
            elif smooth == 'floor':
                precisions[n] = 100. * smooth_floor / total[n]
        else:
            precisions[n] = 100. * correct[n] / total[n]

    # If the system guesses no i-grams, 1 <= i <= NGRAM_ORDER, the BLEU score is 0 (technically undefined).
    # This is a problem for sentence-level BLEU or a corpus of short sentences, where systems will get no credit
    # if sentence lengths fall under the NGRAM_ORDER threshold. This fix scales NGRAM_ORDER to the observed
    # maximum order. It is only available through the API and off by default

    brevity_penalty = 1.0
    if sys_len < ref_len:
        brevity_penalty = math.exp(1 - ref_len / sys_len) if sys_len > 0 else 0.0

    bleu = brevity_penalty * math.exp(sum(map(my_log, precisions[:effective_order])) / effective_order)

    return BLEU._make([bleu, correct, total, precisions, brevity_penalty, sys_len, ref_len])


#%%corpus_bleu
def corpus_bleu(sys_stream, ref_streams, smooth='exp', smooth_floor=0.0, force=False, lowercase=False,
                tokenize=DEFAULT_TOKENIZER, use_effective_order=False) -> BLEU:
    """Produces BLEU scores along with its sufficient statistics from a source against one or more references.
    :param sys_stream: The system stream (a sequence of segments)
    :param ref_streams: A list of one or more reference streams (each a sequence of segments)
    :param smooth: The smoothing method to use
    :param smooth_floor: For 'floor' smoothing, the floor to use
    :param force: Ignore data that looks already tokenized
    :param lowercase: Lowercase the data
    :param tokenize: The tokenizer to use
    :return: a BLEU object containing everything you'd want
    """
    # detokenize
    sys_stream = [' '.join(tokens) for tokens in sys_stream]
    ref_streams = [' '.join(tokens)  for tokens in ref_streams]
    ref_streams = [ref_streams]

    sys_len = 0
    ref_len = 0

    correct = [0 for n in range(NGRAM_ORDER)]   #[0,0,0,0]
    total = [0 for n in range(NGRAM_ORDER)]    #[0,0,0,0]

    # look for already-tokenized sentences
    tokenized_count = 0

    fhs = [sys_stream] + ref_streams
    for lines in zip_longest(*fhs):
        if None in lines:
            raise EOFError("Source and reference streams have different lengths!")

        if lowercase:
            lines = [x.lower() for x in lines]

        if not (force or tokenize == 'none') and lines[0].rstrip().endswith(' .'):
            tokenized_count += 1

            if tokenized_count == 100:
                logging.warning('That\'s 100 lines that end in a tokenized period (\'.\')')
                logging.warning('It looks like you forgot to detokenize your test data, which may hurt your score.')
                logging.warning('If you insist your data is detokenized, or don\'t care, you can suppress this message with \'--force\'.')

        output, *refs = [TOKENIZERS[tokenize](x.rstrip()) for x in lines]

        ref_ngrams, closest_diff, closest_len = ref_stats(output, refs)

        sys_len += len(output.split())
        ref_len += closest_len

        sys_ngrams = extract_ngrams(output)
        for ngram in sys_ngrams.keys():
            n = len(ngram.split())
            correct[n-1] += min(sys_ngrams[ngram], ref_ngrams.get(ngram, 0))
            total[n-1] += sys_ngrams[ngram]

    return compute_bleu(correct, total, sys_len, ref_len, smooth, smooth_floor, use_effective_order)