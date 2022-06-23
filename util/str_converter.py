#!/usr/bin/python
# encoding: utf-8
import sys
import torch
import torch.nn as nn

from torch.autograd import Variable

import collections
import numpy as np

class strLabelConverterForAttention(object):
    """Convert between str and label.

    NOTE:
        Insert `EOS` to the alphabet for attention.

    Args:
        alphabet (str): set of the possible characters.
        ignore_case (bool, default=True): whether or not to ignore all of the case.
    """

    def __init__(self, alphabet, ignore_case=False):
        self._ignore_case = ignore_case
        self.alphabet = alphabet
        self.maxLen = -1

        self.dict = {}
        for i, item in enumerate(self.alphabet):
            self.dict[item] = i

        self.dict['<EOS>'] = len(self.alphabet)

    def encode(self, text):
        """Support batch or single str.

        Args:
            text (str or list of str): texts to convert.

        Returns:
            torch.IntTensor [length_0 + length_1 + ... length_{n - 1}]: encoded texts.
            torch.IntTensor [n]: length of each text.
        """
        
        if isinstance(text, str):
            text_in = [
                self.dict[char.lower() if self._ignore_case else char]
                for char in text
            ]

            length = [len(text_in)]
            # just for multi-GPU training, '[0]' is meaningless
            text_ex = text_in.copy()
            text_ex.extend([0]*(self.maxLen-len(text_ex)))


        elif isinstance(text, collections.Iterable):
            length = [len(t) for t in text]
            self.maxLen = max(length)

            text_in = []
            text_ex = []
            for t in text:
                t_in, t_ex, _ = self.encode(t)
                text_in.append(t_in)
                text_ex.append(t_ex)

            text_in = torch.cat(text_in, 0)
            text_ex = torch.cat(text_ex, 0)

        return (torch.LongTensor(text_in), 
            torch.LongTensor(text_ex), 
            torch.IntTensor(length))

    def decode(self, t, length):
        """Decode encoded texts back into strs.

        Args:
            torch.IntTensor [length_0 + length_1 + ... length_{n - 1}]: encoded texts.
            torch.IntTensor [n]: length of each text.

        Raises:
            AssertionError: when the texts and its length does not match.

        Returns:
            text (str or list of str): texts to convert.
        """
        if length.numel() == 1:
            length = length[0]
            assert t.numel() == length, "text with length: {} does not match declared length: {}".format(t.numel(), length)
            return ''.join([self.alphabet[i] if i < len(self.alphabet) else '' for i in t])
        else:
            # batch mode
            assert t.numel() == length.sum(), "texts with length: {} does not match declared length: {}".format(t.numel(), length.sum())
            texts = []
            index = 0
            for i in range(length.numel()):
                l = length[i]
                texts.append(
                    self.decode(
                        t[index:index + l], torch.LongTensor([l])))
                index += l
            return texts



class lexicontoid(object):
    def __init__(self, dictionary):
        self.dictionary = dictionary
        
    def encode(self, lexicon):
               
        n_batch = len(lexicon)
        imgs = []
        dictionary = self.dictionary
        new_labels = []
        len_labels = []
        #import pdb;pdb.set_trace()
        for label in lexicon:
            label = label.split()
            label2index = []
            len_labels.append(len(label))
            for w in label:
                if dictionary.__contains__(w):
                    label2index.append(dictionary[w])
                else:
                    print('a word not in the dictionary !!', w)
                    sys.exit()
            new_labels.append(label2index)
        #import pdb;pdb.set_trace()
        max_length = max(len_labels)
        np_labels = np.zeros((max_length+1,n_batch)).astype(np.int64)
        np_labels_mask = np.zeros((max_length+1,n_batch)).astype(np.float32)
        for idx,label in enumerate(new_labels):
            np_labels[:len_labels[idx],idx] = label
            np_labels_mask[:len_labels[idx]+1,idx] = 1.
        #import pdb;pdb.set_trace()
        new_labels = torch.from_numpy(np_labels)
        new_labels_mask = torch.from_numpy(np_labels_mask)
                
        return new_labels, new_labels_mask


class AttnLabelConverter(object):
    """ Convert between text-label and text-index """

    def __init__(self, character):
        # character (str): set of the possible characters.
        # [GO] for the start token of the attention decoder. [s] for end-of-sentence token.
        list_token = ['[GO]', '[s]']  # ['[s]','[UNK]','[PAD]','[GO]']
        #import pdb;pdb.set_trace()
        list_character = list(character)
        self.character = list_token + list_character

        self.dict = {}
        for i, char in enumerate(self.character):
            self.dict[char] = i
        

    def encode(self, text):
        """ convert text-label into text-index.
        input:
            text: text labels of each image. [batch_size]
            batch_max_length: max length of text label in the batch. 25 by default

        output:
            text : the input of attention decoder. [batch_size x (max_length+2)] +1 for [GO] token and +1 for [s] token.
                text[:, 0] is [GO] token and text is padded with [GO] token after [s] token.
            length : the length of output of attention decoder, which count [s] token also. [3, 7, ....] [batch_size]
        """
        #import pdb;pdb.set_trace()
        length = [len(s.split()) for s in text]  # +1 for [s] at end of sentence.
        # batch_max_length = max(length) # this is not allowed for multi-gpu setting
        batch_max_length  = max(length)+2
        # additional +1 for [GO] at first step. batch_text is padded with [GO] token after [s] token.
        batch_text = torch.LongTensor(len(text), batch_max_length).fill_(0)
        for i, t in enumerate(text):
            text = t.split()
            text.append('[s]')
            text = [self.dict[char] for char in text]
            batch_text[i][1:1 + len(text)] = torch.LongTensor(text)  # batch_text[:, 0] = [GO] token
        #import pdb;pdb.set_trace()
        return (batch_text, torch.IntTensor(length))

    def decode(self, text_index, length):
        """ convert text-index into text-label. """
        #import pdb;pdb.set_trace()
        texts = []
        for index, l in enumerate(length):
            text = ''.join([self.character[i] for i in text_index[index, :]])
            texts.append(text)
        #import pdb;pdb.set_trace()
        return texts


class AttnLabelConverter_IAM(object):
    """ Convert between text-label and text-index """

    def __init__(self, character):
        # character (str): set of the possible characters.
        # [GO] for the start token of the attention decoder. [s] for end-of-sentence token.
        list_token = ['[GO]', '[s]']  # ['[s]','[UNK]','[PAD]','[GO]']
        #import pdb;pdb.set_trace()
        list_character = list(character)
        self.character = list_token + list_character

        self.dict = {}
        for i, char in enumerate(self.character):
            self.dict[char] = i
        

    def encode(self, text):
        """ convert text-label into text-index.
        input:
            text: text labels of each image. [batch_size]
            batch_max_length: max length of text label in the batch. 25 by default

        output:
            text : the input of attention decoder. [batch_size x (max_length+2)] +1 for [GO] token and +1 for [s] token.
                text[:, 0] is [GO] token and text is padded with [GO] token after [s] token.
            length : the length of output of attention decoder, which count [s] token also. [3, 7, ....] [batch_size]
        """
        #import pdb;pdb.set_trace()
        length = [len(s) for s in text]  # +1 for [s] at end of sentence.
        # batch_max_length = max(length) # this is not allowed for multi-gpu setting
        batch_max_length  = max(length)+2
        # additional +1 for [GO] at first step. batch_text is padded with [GO] token after [s] token.
        batch_text = torch.LongTensor(len(text), batch_max_length).fill_(0)
        for i, t in enumerate(text):
            text_new = list(t)
            text_new.append('[s]')
            text_new = [self.dict[char] if char in self.dict else len(self.dict) for char in text_new]
            batch_text[i][1:1 + len(text_new)] = torch.LongTensor(text_new)  # batch_text[:, 0] = [GO] token
        #import pdb;pdb.set_trace()
        return (batch_text, torch.IntTensor(length))

    def decode(self, text_index, length):
        """ convert text-index into text-label. """
        #import pdb;pdb.set_trace()
        texts = []
        for index, l in enumerate(length):
            text = ''.join([self.character[i] for i in text_index[index, :]])
            texts.append(text)
        #import pdb;pdb.set_trace()
        return texts