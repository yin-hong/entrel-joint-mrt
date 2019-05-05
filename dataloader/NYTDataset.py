import torch
import torch.utils.data as data
from utils import constant
import json
import random
import numpy as np
import os
import pickle
import time

class NYTDataset(data.Dataset):
    """
    NYT Dataset
    """
    def __init__(self, filename, batch_size, parent_file, rel_file, pos_file,
                 vocab, evaluation=False):
        """
        Load data from json files, preprocess and prepare batches
        :param filename: file path
        :param batch_size: batch size
        :param parent_file: dependency tree head file
        :param rel_file: dependency tree relation file
        :param pos_file: pos file path
        :param vocab: Vocab object
        :param evaluation: boolean value, train or test
        """
        super(NYTDataset, self).__init__()
        self.filename = filename
        self.batch_size = batch_size
        self.parent_file = parent_file
        self.rel_file = rel_file
        self.pos_file = pos_file
        self.vocab = vocab
        self.eval = evaluation
        self.label2id = constant.LABEL_TO_ID
        data = self.preprocess(filename, parent_file, rel_file, pos_file, vocab, evaluation)
        self.data = data
        self.max_len = self.max_length()
        print('{} batches created for {}'.format(len(data), filename))

    def __len__(self):
        return len(self.data_batch)

    def __getitem__(self, item):
        """
        Get data with index
        :param item:
        :return:
        """
        # if not isinstance(item, int):
        #     raise TypeError
        # start_time = time.time()
        if item < 0 or item > len(self.data_batch):
            raise IndexError
        batch = self.data_batch[item]
        batch_size = len(batch)
        batch = list(zip(*batch))
        assert len(batch) == 6

        # sort all fields by lens for easy RNN operations
        lens = [len(x) for x in batch[0]]
        batch, orig_idx = sort_all(batch, lens)


        if not self.eval:
            words = [word_dropout(sent, 0.04) for sent in batch[0]]
        else:
            words = batch[0]
        # convert to tensors
        tokens_tensor = get_long_tensor(words, batch_size) # of shape (batch_size, seq_length)
        masks = torch.eq(tokens_tensor, 0)
        pos_tensor = get_long_tensor(batch[1], batch_size)
        parent_tensor = get_long_tensor(batch[2], batch_size)
        deprel_tensor = get_long_tensor(batch[3], batch_size)
        entity_tag = get_long_tensor(batch[4], batch_size)
        relation_tags = batch[5]
        return tokens_tensor, masks, pos_tensor, parent_tensor, deprel_tensor, lens, entity_tag, relation_tags

    def __iter__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)

    def preprocess(self, filename, parent_file, rel_file, pos_file, vocab, evaluation):
        """
        Preprocess the data and convert into idx
        :param filename: token file path
        :param parent_file: dependency tree head file
        :param rel_file: dependency tree relation file
        :param vocab: vocabulary
        :param evaluation: boolean
        :return: processed, list type, every element is list(mean one sentence information), containing token_id, dependency head,and so on
        """
        processed = []
        # if os.path.exists('data/train_save_dataset/'):
        #     start = 0
        #     while start <= 235200:
        #         f = open('data/train_save_dataset/' + str(start) + '.pkl', 'rb')
        #         datatemp = pickle.load(f)
        #         processed += datatemp.data
        #         f.close()
        #         start += 600
        #     return processed
        with open(filename, 'r', encoding='utf8', errors='ignore') as infile, \
              open(parent_file, 'r', encoding='utf8', errors='ignore') as f_parent, \
              open(rel_file, 'r', encoding='utf8', errors='ignore') as f_rel, \
              open(pos_file, 'r', encoding='utf8', errors='ignore') as f_pos:
            sentences = infile.readlines()
            sentences_pos = f_pos.readlines()
            sentences_parent = f_parent.readlines()
            sentences_rel = f_rel.readlines()
            assert len(sentences) == len(sentences_pos)
            assert len(sentences_pos) == len(sentences_parent)
            assert len(sentences_parent) == len(sentences_rel)
            for line, pos, parent, deprel in zip(sentences, sentences_pos, sentences_parent, sentences_rel):
                sent = json.loads(line.strip('\r\n'))
                tokens = sent['tokens']
                relations = sent['relations']
                entityMention = sent['EntityMention']
                pos = pos.split()
                parent = parent.split(' ')
                deprel = deprel.split(' ')
                assert len(tokens) == len(pos)
                assert len(pos) == len(parent)
                assert len(parent) == len(deprel)
                tokens = map_to_ids(tokens, vocab.WordsToIdx)
                pos = map_to_ids(pos, constant.POS_TO_ID)
                parent = list(map(int, parent))
                deprel = map_to_ids(deprel, constant.DEPREL_TO_ID)
                entity_tag = [0] * len(tokens)
                for entity in entityMention:
                    ent_start = entity['start']
                    ent_end = entity['end']
                    ent_label = entity['label'][0]
                    if ent_start == ent_end:
                        entity_tag[ent_start] = constant.NER_TAG_TO_ID['u-' + ent_label.lower()]
                    elif ent_end - ent_start == 1:
                        entity_tag[ent_start] = constant.NER_TAG_TO_ID['b-' + ent_label.lower()]
                        entity_tag[ent_end] = constant.NER_TAG_TO_ID['l-' + ent_label.lower()]
                    else:
                        entity_tag[ent_start] = constant.NER_TAG_TO_ID['b-' + ent_label.lower()]
                        for idx in range(ent_start + 1, ent_end):
                            entity_tag[idx] = constant.NER_TAG_TO_ID['i-' + ent_label.lower()]
                        entity_tag[ent_end] = constant.NER_TAG_TO_ID['l-' + ent_label.lower()]
                relation_tags = list()
                for relation in relations:
                    # direction = '-->'
                    relation_label = relation['label']
                    if relation_label == 'None':
                        continue
                    subj_start, subj_end = relation['subj_start'], relation['subj_end']
                    obj_start, obj_end = relation['obj_start'], relation['obj_end']
                    # if subj_start > obj_start:
                    #     direction = '<--'
                    #     temp1, temp2 = subj_start, subj_end
                    #     subj_start, subj_end = obj_start, obj_end
                    #     obj_start, obj_end = temp1, temp2
                    # relation_label = relation_label + direction
                    rel = [subj_start, subj_end, obj_start, obj_end, relation_label]
                    relation_tags.append(rel)
                processed += [(tokens, pos, parent, deprel, entity_tag, relation_tags)]
        return processed

    def max_length(self):
        max_length = -10000
        for d in self.data:
            tokens = d[0]
            if len(tokens) > max_length:
                max_length = len(tokens)
        return max_length


    def batch_and_shuffle(self):
        if not self.eval:
            indices = list(range(len(self.data)))
            random.shuffle(indices)
            data = [self.data[i] for i in indices]
        else:
            data = self.data
        data = [data[i: i + self.batch_size] for i in range(0, len(data), self.batch_size)]
        self.data_batch = data


def map_to_ids(tokens, vocab):
    ids = [vocab[t] if t in vocab else constant.UNK_ID for t in tokens]
    return ids



def sort_all(batch, lens):
    """
    Sort all fields by descending order of lens, and return the original indices.
    :param batch: will sort with respect to lens
    :param lens: list
    :return: sorted batch, original indices
    """
    unsorted_all = [lens] + [range(len(lens))] + list(batch)
    sorted_all = [list(t) for t in zip(*sorted(zip(*unsorted_all), reverse=True))]
    return sorted_all[2:], sorted_all[1]

def get_long_tensor(token_list, batch_size):
    """
    Convert list of list of tokens to a padded LongTensor.
    Not only token need pad, but also POS, dependency, subj_position, obj_position and so on does .
    :param token_list: list, every element is list
    :param batch_size: int
    :return: tokens: torch LongTensor, of shape (batch_size, max length of token_list)
    """
    # token_len = token_len
    token_len = max(len(x) for x in token_list)
    tokens = torch.LongTensor(batch_size, token_len).fill_(constant.PAD_ID)
    # add PAD at the end of tokens to equal to token_len
    for i, s in enumerate(token_list):
        tokens[i, :len(s)] = torch.LongTensor(s)
    return tokens

def get_position(start, end, length):
    """
    Obtain entity mention position in token, entity mention pos is 0 while ohters
    is not 0
    :param start: entity mention start position
    :param end: entity mention end position
    :param length: tokens length
    :return: pos: list, entity mention pos is 0 while other position is not 0
    """
    return list(range(-start, 0)) + [0] * (end - start + 1) + list(range(1, length - end))

def word_dropout(tokens, dropout):
    """ Randomly dropout tokens (IDs) and replace them with <UNK> tokens. """
    return [constant.UNK_ID if x != constant.UNK_ID and np.random.random() < dropout \
            else x for x in tokens]


