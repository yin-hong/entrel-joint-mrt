#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@author: Michael Hong
"""
import json
import numpy as np
import os

data_dir = '../data'


def token2offset(tokens):
    """
    convert tokens(string) list into  dict: the key are token; the value are all position of token
    :param tokens: list
    :return: tok2off: dict
    """
    tok2off = {}
    for i, tok in enumerate(tokens):
        if tok not in tok2off:
            tok2off[tok] = []
        tok2off[tok].append(i)
    return tok2off

def offset_of_tokens(en_text, tok2off, tokens):
    toks = en_text.split(' ')
    for first_idx in tok2off[toks[0]]:
        flag = True
        for i in range(len(toks)):
            if tokens[first_idx + i] != toks[i]:
                flag = False
                break
        if flag:
            return (first_idx, first_idx + len(toks))


def replace_latin(string):
    r = {'á': 'a',
         'â': 'a',
         'Á': 'A',
         'É': 'E',
         'è': 'e',
         'é': 'e',
         'ê': 'e',
         'í': 'i',
         'ó': 'o',
         'ô': 'o',
         'ö': 'o',
         'Ó': 'O',
         'ú': 'u',
         'ü': 'u',
         'ñ': 'n'}
    for k, v in r.items():
        string = string.replace(k, v)
    return string

def convert_format(i, sent):
    new_sent = {'sentId': sent['sentId'],
                'articleId': sent['articleId'],
                'sentText': replace_latin(sent['sentText'].strip()),
                'entityMentions': [],
                'relationMentions': []
                }
    # the sentences of test dataset are begin with '"' and end with '"'
    if new_sent['sentText'][0] == '"' and new_sent['sentText'][-1] == '"':
        new_sent['sentText'] = new_sent['sentText'][1:-1]


    tokens = new_sent['sentText'].split(' ')
    if len(tokens[-1]) > 1 and tokens[-1].endswith('.'):
        temp = list()
        temp.append(tokens[-1][0:-1])
        temp.append('.')
        tokens = tokens[0:-1] + temp

    tok2off = token2offset(tokens)
    ent2id = {}

    for ent_mention in sent['entityMentions']:
        new_sent_mention = {}
        new_sent_mention['emId'] = ent_mention['start']
        new_sent_mention['label'] = ent_mention['label']
        new_sent_mention['text'] = replace_latin(ent_mention['text'])

        toks = new_sent_mention['text'].split(' ')

        if toks[0] not in tok2off:
            return None
        # assert toks[0] in tok2off

        new_sent_mention['offset'] = offset_of_tokens(new_sent_mention['text'], tok2off, tokens)

        if new_sent_mention['offset'] is None:
            return None
        # assert new_sent_mention['offset'] is not None

        recon_txt = ' '.join([tokens[e] for e in range(new_sent_mention['offset'][0], new_sent_mention['offset'][1])])

        assert new_sent_mention['text'] == recon_txt

        ent2id[new_sent_mention['text']] = new_sent_mention['emId']
        new_sent['entityMentions'].append(new_sent_mention)

    none_ct = 0
    for rel_mention in sent['relationMentions']:
        # exclude 'None' label
        if rel_mention['label'] == 'None':
            none_ct += 1
            continue

        new_rel_mention = {}
        new_rel_mention['em1Text'] = replace_latin(rel_mention['em1Text'])
        new_rel_mention['em2Text'] = replace_latin(rel_mention['em2Text'])
        new_rel_mention['label'] = rel_mention['label']

        new_rel_mention['em1Id'] = ent2id[new_rel_mention['em1Text']]
        new_rel_mention['em2Id'] = ent2id[new_rel_mention['em2Text']]
        new_sent['relationMentions'].append(new_rel_mention)
    return new_sent, len(sent['relationMentions']), none_ct


def process_data_and_save_json(data, save_path):
    """
    preprocess data and save into file
    :param data: list, every element is one sentence
    :param save_path: string, save path
    :return:
    """
    with open(save_path, 'w') as g:
        rel_all = 0
        rel_none = 0
        delete_sent_num = 0
        for i, sent in enumerate(data):
            result = convert_format(i, sent)
            if result is None:
                delete_sent_num += 1
                continue
            sent, r_all, r_none = result
            rel_all += r_all
            rel_none += r_none
            print(json.dumps(sent), file=g)
        print('======================================================')
        print('%s set Delete Sentences Num:'%save_path, delete_sent_num)
        print('%s set Relation Num (include None relation):'%save_path, rel_all)
        print('%s set None Relation Num:' % save_path, rel_none)
        print('%s set Relation Num(exclude None relation):' % save_path, rel_all - rel_none)
        print()

if __name__ == '__main__':
    with open(os.path.join(data_dir, 'nyt/train.json'), 'r', encoding='utf8', errors='ignore') as f:
        data = []
        for line in f:
            sent = json.loads(line)
            data.append(sent)

    process_data_and_save_json(data, os.path.join(data_dir, 'preprocess/train.json'))

    with open(os.path.join(data_dir, 'nyt/test.json'), 'r', encoding='utf8', errors='ignore') as f:
        data = []
        for line in f:
            data.append(json.loads(line))

    np.random.seed(0)
    np.random.shuffle(data)
    dev_len = int(len(data) * 0.1)
    dev_data, test_data = data[:dev_len], data[dev_len:]
    process_data_and_save_json(dev_data, os.path.join(data_dir, 'preprocess/dev.json'))
    process_data_and_save_json(test_data, os.path.join(data_dir, 'preprocess/test.json'))







