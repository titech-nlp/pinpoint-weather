# -*- coding: utf-8 -*-
import os
import json
import collections
import itertools

from lib.constant import IDs, Tokens

class WNCGVocabulary:
    def __init__(self, dataset_dir, vocab_limit=3):
        """ @param:
                dataset_dir -- path to dataset dir
                vocab_limit -- # of vocabulary limitation
        """
        self.tgt_comment_list = []

        # load_dataset
        wncg_comments = json.load(open(os.path.join(dataset_dir, "wncg-comment.json")))
        for key, comment_data in wncg_comments.items():
            # construct vocaburaries from train-data
            if comment_data["data_split"] == "train":
                self.tgt_comment_list.append(comment_data["comment"])

        self.word2id, self.id2word = self._load_comment_data(self.tgt_comment_list, vocab_limit)

    # input data
    def _load_comment_data(self, text_list, vocab_limit):
        """ ファイルを読み込み、データセット, 単語辞書を返す.
            データセットは、辞書を使ってindex化
            @param:
                data -- input data. list of documents
                vocab_limit -- threshold to filter out words whose frequency is less than vocab_limit
                delimiter -- delimiter token to split each document into sequnce of words
        """
        # 単語-ID、ID-単語 辞書を作成 (Special token)
        word2id = {}
        id2word = {}
        for token in list(Tokens):
            word_id = IDs.__members__[token.name].value
            word2id[token.value] = word_id
            id2word[word_id] = token.value

        # 単語リストと文頭・文末文字を追加した文書リストを作成
        word_freq = {}
        for text in text_list:
            for word in text.strip().split(): # count word frequency
                if word not in word_freq:
                    word_freq[word] = 1
                else:
                    word_freq[word] += 1

        i = 0
        NUM_OF_SPECIAL_TOKENS = len(word2id)
        WORD_FREQ_THRESHOLD = vocab_limit
        for word, count in sorted(word_freq.items(), key=lambda x:x[1], reverse=True):
            word2id[word] = i + NUM_OF_SPECIAL_TOKENS
            id2word[i + NUM_OF_SPECIAL_TOKENS] = word
            i += 1
            # Regard the word as unknown when it is rare
            if count <= WORD_FREQ_THRESHOLD:
                break
        return word2id, id2word

    def itos(self, index):
        # convert Id-to-Word
        return self.id2word.get(index, Tokens.UNK.value)

    def stoi(self, token):
        # convert Word-to-Id
        return self.word2id.get(token, IDs.UNK.value)

    def load_vocab(self, word2id, id2word):
        """ ファイルから語彙辞書を読込
        """
        self.word2id = word2id
        self.id2word = id2word