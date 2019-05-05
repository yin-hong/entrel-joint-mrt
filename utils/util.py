import json
import nltk
from utils import constant
from collections import Counter
import os
from utils.vocab import Vocab
import numpy as np
import torch
import re


def build_vocab(filenames, vocabfile, min_freq):
    """
    use train, dev, test file to build vocabulary
    :param filenames: list type, containing training dev, test filename.
    :param vocabfile: str. saved vocab path
    :param: min_freq: int, if word freq is less than min_freq, then remove it
    :return:
    """
    tokens = []
    for filename in filenames:
        tokens += load_tokens(filename)
    counter = Counter(token for token in tokens)
    vocab = sorted([t for t in counter if counter.get(t) >= min_freq], key=counter.get, reverse=True)
    vocab = constant.VOCAB_PREFIX + vocab
    with open(vocabfile, 'w', encoding='utf8', errors='ignore') as f:
        for v in vocab:
            f.write(v + '\n')

def load_tokens(filename):
    """
    load tokens from filename
    :param filename: file path
    :return:list, every element is word
    """
    tokens = []
    with open(filename, 'r', encoding='utf8', errors='ignore') as infile:
        sentences = infile.readlines()
        for line in sentences:
            sent = json.loads(line.strip('\r\n'))
            sentText = sent['sentText']
            tokens += nltk.word_tokenize(sentText)
    return tokens

def load_word_vector(path):
    """
    loading word vector(this project employs GLOVE word vector), save GLOVE word, vector as file
    respectively
    :param path: GLOVE word vector path
    :return: glove vocab,: vocab object, vector(numpy array, of shape(words_num, word_dim))
    """
    base = os.path.splitext(os.path.basename(path))[0]
    glove_vocab_path = os.path.join('../data/glove/', base + '.vocab')
    glove_vector_path = os.path.join('../data/glove/', base + '.path')
    # haved loaded word vector
    if os.path.isfile(glove_vocab_path) and os.path.isfile(glove_vector_path):
        print('======> File found, loading memory <=====!')
        vocab = Vocab(glove_vocab_path)
        vector = np.load(glove_vector_path)
        return vocab, vector

    print('=====>Loading glove word vector<=====')
    with open(path, 'r', encoding='utf8', errors='ignore') as f:
        contents = f.readline().rstrip('\n').split(' ')
        word_dim = len(contents[1:])
        count = 1
        for line in f:
            count += 1

    vocab = [None] * count
    vector = np.zeros((count, word_dim))
    with open(path, 'r', encoding='utf8', errors='ignore') as f:
        idx = 0
        for line in f:
            contents = line.rstrip('\n').split(' ')
            vocab[idx] = contents[0]
            vector[idx] = np.array(list(map(float, contents[1:])), dtype=float)
            idx += 1
    assert count == idx
    with open(glove_vector_path, 'w', encoding='utf8', errors='ignore') as f:
        for token in vocab:
            f.write(token + '\n')

    vocab = Vocab(glove_vocab_path)
    torch.save(vector, glove_vector_path)
    return vocab, vector


def get_embedding(vocab, pre_embedding, pre_vocab):
    """
    Obtain the word embedding. If words are in vocab and glove at the same time, then using
    glove_embedding. Otherwise we can use random vector. Noting that <pad> should be all 0
    :param vocab: Vocab object
    :param pre_embedding: in this project, we use glove embedding
    :param pre_vocab: vocab of pre_embedding, Vocab object
    :return: embedding: np.array, of shape (vocab size, word dim)
    """
    embedding = np.random.uniform(-1, 1, (len(vocab.WordsToIdx), pre_embedding.size(1)))
    # <pad> should be all 0
    embedding[constant.PAD_ID] = 0.
    for word in vocab.WordsToIdx:
        if pre_vocab.get_index(word) is not None:
            embedding[vocab.get_index(word)] = pre_embedding[pre_vocab.get_index(word)]

    return embedding

def load_entity_and_relation_sequences(filename, sep='\t', schema='BIO'):
    def convert_sequence(source_path, target_path):
        fsource = open(source_path, 'r', encoding='utf8')
        ftarget = open(target_path, 'w', encoding='utf8')
        for line in fsource:
            sent = json.loads(line)
            tokens = sent['sentText'].split(' ')
            tags = ['0'] * len(tokens)
            id2ent = {}
            for men in sent['entityMentions']:
                id2ent[men['emId']] = men['offset']
                s, e = men['offset']
                if schema == 'BIO':
                    tags[s] = 'B-' + men['label']
                    for j in range(s+1, e):
                        tags[j] = 'I-' + men['label']
                else:
                    if e - s == 1:
                        tags[s] = 'U-' + men['label']
                    elif e - s == 2:
                        tags[s] = 'B-' + men['label']
                        tags[s+1] = 'E-' + men['label']
                    else:
                        tags[s] = 'B-' + men['label']
                        tags[e-1] = 'E-' + men['label']
                        for j in range(s+1, e-1):
                            tags[j] = 'I-' + men['label']
            for w, t in zip(tokens, tags):
                print('{0}\t{1}'.format(w, t), file=ftarget)
            for men in sent['relationMentions']:
                em1_idx = id2ent[men['em1Id']]
                em2_idx = id2ent[men['em2Id']]
                em1_text = men['em1Text']
                em2_text = men['em2Text']


def parse_tag(t):
    m = re.match(r'^([^-]*)-(.*)$', t)
    return m.groups() if m else (t, '')



if __name__ == '__main__':
    # build_vocab(['../data/nyt/train.json', '../data/nyt/test.json'], '../data/vocab.txt', 0)
    load_word_vector('../data/glove/glove.840B.300d.txt')


