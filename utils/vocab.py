"""
Vocabulary
"""
from utils import constant
class Vocab(object):
    """Vocabulary object"""
    def __init__(self, filename=None, special_words=None, lower=False):
        """

        :param filename: vocab filepath
        :param special_words: list or None : if special_words is not None(i.e. it is list), add its to vocab
        :param lower: whether lower
        """
        self.WordsToIdx = {}
        self.IdxToWords = {}

        # if special_words is not None, we add special words to vocab at first
        # It means that special words is in front of vocabulary
        if special_words is not None:
            for special_word in special_words:
                self.add(special_word)
        # load vocab from filename
        if filename is not None:
            self.loadVocab(filename)
        self.size = len(self.IdxToWords)

    def loadVocab(self, filename):
        """
        Load vocab from file
        :param: filename: vocab path
        :return: None
        """
        with open(filename, 'r', encoding='utf8', errors='ignore') as f:
            for word in f:
                word = word.strip()
                if word == '':
                    continue
                self.add(word)

    def add(self, word):
        """
        Add word into vocab
        :param word: word
        :return: None
        """
        if word not in self.WordsToIdx:
            index = len(self.WordsToIdx)
            self.WordsToIdx[word] = index
            self.IdxToWords[index] = word


    def get_index(self, word, defalut=None):
        """
        Get index of word. If word is not in vocab, then return default
        :param word: word
        :param defalut: if word is not in vocab, then return default
        :return: index or default
        """
        index = self.WordsToIdx.get(word, defalut)
        return index

    def get_word(self, index, default=None):
        """
        Get word of index. If word is not in vocab, then return default
        :param index: int
        :param default: if word is not in vocab, then return default
        :return: word or default
        """
        word = self.IdxToWords(index, default)
        return word

    def convert_words_to_idx(self, words, unkWord, bosWord=None, eosWord=None):
        """
        Convert words into index
        :param words: list: every element is word
        :param unkWord: if word is not in vocab, then replace it with unkWord
        :param bosWord: option, if bosWord is not None, then place it in the front of words
        :param eosWord: option, if eosWord is not None, then place it in the end of words
        :return: indices: list, every elemtent is word index
        """
        indices = []
        unk_index = self.get_index(unkWord)

        # if bosWord is not none, then add it in the front of words
        if bosWord is not None:
            indices.append(self.get_index(bosWord))

        for word in words:
            indices.append(self.get_index(word))

        # if eosWord is not none, then add it in the end of words
        if eosWord is not None:
            indices.append(self.get_index(eosWord))
        return indices

    def convert_idx_to_words(self, indices, unk_index):
        """
        convert indices into words
        :param indices: list, every element is index
        :param unk_index: if index is not in vocab, then use unkWord to replace
        :return: words: list, every element is word
        """
        words = []
        for index in indices:
            words.append(self.get_word(index, unk_index))
        return words



if __name__ == '__main__':
    vocab_path = '../data/vocab.txt'
    special_words = constant.VOCAB_PREFIX
    vocab = Vocab(filename=vocab_path, special_words=special_words, lower=False)


