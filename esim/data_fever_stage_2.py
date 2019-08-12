"""
Preprocessor and dataset definition for NLI.
"""
# Aurelien Coet, 2018.

import string
import torch
import numpy as np
import pickle
import os
import json

from collections import Counter
from torch.utils.data import Dataset


class Preprocessor(object):
    """
    Preprocessor class for Natural Language Inference datasets.

    The class can be used to read NLI datasets, build worddicts for them
    and transform their premises, hypotheses and labels into lists of
    integer indices.
    """

    def __init__(self,
                 lowercase=False,
                 ignore_punctuation=False,
                 num_words=None,
                 stopwords=[],
                 labeldict={},
                 bos=None,
                 eos=None):
        """
        Args:
            lowercase: A boolean indicating whether the words in the datasets
                being preprocessed must be lowercased or not. Defaults to
                False.
            ignore_punctuation: A boolean indicating whether punctuation must
                be ignored or not in the datasets preprocessed by the object.
            num_words: An integer indicating the number of words to use in the
                worddict of the object. If set to None, all the words in the
                data are kept. Defaults to None.
            stopwords: A list of words that must be ignored when building the
                worddict for a dataset. Defaults to an empty list.
            bos: A string indicating the symbol to use for the 'beginning of
                sentence' token in the data. If set to None, the token isn't
                used. Defaults to None.
            eos: A string indicating the symbol to use for the 'end of
                sentence' token in the data. If set to None, the token isn't
                used. Defaults to None.
        """
        self.lowercase = lowercase
        self.ignore_punctuation = ignore_punctuation
        self.num_words = num_words
        self.stopwords = stopwords
        self.labeldict = labeldict
        self.bos = bos
        self.eos = eos
        self.remove_parentheses_flag = False

    
        
    def read_data(self, filepath):
        """
        Read the premises, hypotheses and labels from some NLI dataset's
        file and return them in a dictionary. The file should be in the same
        form as SNLI's .txt files.

        Args:
            filepath: The path to a file containing some premises, hypotheses
                and labels that must be read. The file should be formatted in
                the same way as the SNLI (and MultiNLI) dataset.

        Returns:
            A dictionary containing three lists, one for the premises, one for
            the hypotheses, and one for the labels in the input data.
        """
        with open(filepath, "rb") as input_data:
            dict_out = pickle.load(input_data)
    
#         with open(filepath, "r", encoding="utf8") as input_data:
#             ids, premises, hypotheses, labels = [], [], [], []

        # Translation tables to remove parentheses and punctuation from
        # strings.
        parentheses_table = str.maketrans({"(": None, ")": None})
        punct_table = str.maketrans({key: " "
                                     for key in string.punctuation})
#         dict_out = {}
        
        # process labels
        dict_out['labels'] = [convert_label_stage_2(label) for label in dict_out['labels']]
        # process hypotheses, hypotheses
        for text_type in ['hypotheses', 'premises']:
            if self.lowercase:
                for sentence_id in range(len(dict_out[text_type])):
                    for word_id in range(len(dict_out[text_type][sentence_id])):
                        dict_out[text_type][sentence_id][word_id] = dict_out[text_type][sentence_id][word_id].lower()
            if self.ignore_punctuation:
                for sentence_id in range(len(dict_out[text_type])):
                    for word_id in range(len(dict_out[text_type][sentence_id])):
                        dict_out[text_type][sentence_id][word_id] = dict_out[text_type][sentence_id][word_id].translate(punct_table)
            if self.remove_parentheses_flag:
                for sentence_id in range(len(dict_out[text_type])):
                    dict_out[text_type][sentence_id] = [value for value in dict_out[text_type][sentence_id] if value not in ['(',')']]
            
            # Ignore the headers on the first line of the file.
#             next(input_data)

#             for line in input_data:
#                 line = line.strip().split("\t")
                
#                 if self.remove_parantheses_flag:
                # Remove '(' and ')' from the premises and hypotheses.
#                 premise = premise.translate(parentheses_table)
#                 hypothesis = hypothesis.translate(parentheses_table)

#                 if self.lowercase:
#                     dict_out['hypotheses'] = dict_out['hypotheses']
#                     premise = premise.lower()
#                     hypothesis = hypothesis.lower()

#                 if self.ignore_punctuation:
#                     premise = premise.translate(punct_table)
#                     hypothesis = hypothesis.translate(punct_table)

#                 # Each premise and hypothesis is split into a list of words.
#                 premises.append([w for w in premise.rstrip().split()
#                                  if w not in self.stopwords])
#                 hypotheses.append([w for w in hypothesis.rstrip().split()
#                                    if w not in self.stopwords])
#                 labels.append(line[0])
#                 ids.append(pair_id)

        return dict_out
#                 {"ids": ids,
#                 "premises": premises,
#                 "hypotheses": hypotheses,
#                 "premises_part_of_speech": premises_part_of_speech,
#                 "premises_out_of_vocabulary": premises_out_of_vocabulary,
#                 "hypotheses_part_of_speech": hypotheses_part_of_speech,
#                 "hypotheses_out_of_vocabulary": hypotheses_out_of_vocabulary,
#                 "labels": labels}

    def build_worddict(self, data):
        """
        Build a dictionary associating words to unique integer indices for
        some dataset. The worddict can then be used to transform the words
        in datasets to their indices.

        Args:
            data: A dictionary containing the premises, hypotheses and
                labels of some NLI dataset, in the format returned by the
                'read_data' method of the Preprocessor class.
        """
        words = []
        [words.extend(sentence) for sentence in data["premises"]]
        [words.extend(sentence) for sentence in data["hypotheses"]]
    
        path_text_claim_dev = '/mnt/01_thesis/01_code/ESIM/word_list_wiki_claim.json'
        dict_vocab = dict_load_json(path_text_claim_dev)
        vocab = dict_vocab['vocab']
        [words.extend(word) for word in vocab]
        
        print('length_vocab', len(vocab), 'length_words', len(words))
        
        counts = Counter(words)
        num_words = self.num_words
        if self.num_words is None:
            num_words = len(counts)

        self.worddict = {}

        # Special indices are used for padding, out-of-vocabulary words, and
        # beginning and end of sentence tokens.
        self.worddict["_PAD_"] = 0
        self.worddict["_OOV_"] = 1

        offset = 2
        if self.bos:
            self.worddict["_BOS_"] = 2
            offset += 1
        if self.eos:
            self.worddict["_EOS_"] = 3
            offset += 1

        for i, word in enumerate(counts.most_common(num_words)):
            self.worddict[word[0]] = i + offset

        if self.labeldict == {}:
            label_names = set(data["labels"])
            self.labeldict = {label_name: i
                              for i, label_name in enumerate(label_names)}

    def words_to_indices(self, sentence):
        """
        Transform the words in a sentence to their corresponding integer
        indices.

        Args:
            sentence: A list of words that must be transformed to indices.

        Returns:
            A list of indices.
        """
        indices = []
        # Include the beggining of sentence token at the start of the sentence
        # if one is defined.
        if self.bos:
            indices.append(self.worddict["_BOS_"])
        
        oov_word_list = []
        
        for word in sentence:
            if word in self.worddict:
                index = self.worddict[word]
            else:
                # Words absent from 'worddict' are treated as a special
                # out-of-vocabulary word (OOV).
                oov_word_list.append(word)
                index = self.worddict["_OOV_"]
            indices.append(index)
        # Add the end of sentence token at the end of the sentence if one
        # is defined.
        if self.eos:
            indices.append(self.worddict["_EOS_"])
        
        path_file = 'oov_word_list_stage_2.json'
        if os.path.isfile(path_file):
            dict_oov = dict_load_json(path_file)
        else:
            dict_oov = {}
            dict_oov['oov'] = []
        dict_oov['oov'] += oov_word_list
        dict_save_json(dict_oov, path_file)
        
        return indices

    def indices_to_words(self, indices):
        """
        Transform the indices in a list to their corresponding words in
        the object's worddict.

        Args:
            indices: A list of integer indices corresponding to words in
                the Preprocessor's worddict.

        Returns:
            A list of words.
        """
        return [list(self.worddict.keys())[list(self.worddict.values())
                                           .index(i)]
                for i in indices]

    def transform_to_indices(self, data):
        """
        Transform the words in the premises and hypotheses of a dataset, as
        well as their associated labels, to integer indices.

        Args:
            data: A dictionary containing lists of premises, hypotheses
                and labels, in the format returned by the 'read_data'
                method of the Preprocessor class.

        Returns:
            A dictionary containing the transformed premises, hypotheses and
            labels.
        """
        transformed_data = {"ids": [],
                            "premises": [],
                            "hypotheses": [],
                            "premises_part_of_speech": [],
                            "premises_out_of_vocabulary": [],
                            "hypotheses_part_of_speech": [],
                            "hypotheses_out_of_vocabulary": [],
                            "labels": []
                           }

        for i, premise in enumerate(data["premises"]):
            # Ignore sentences that have a label for which no index was
            # defined in 'labeldict'.
            label = data["labels"][i]
            if label not in self.labeldict and label != "hidden":
                continue

            transformed_data["ids"].append(data["ids"][i])
            transformed_data["premises_part_of_speech"].append([17] + data["premises_part_of_speech"][i] + [17])
            transformed_data["premises_out_of_vocabulary"].append([0] + data["premises_out_of_vocabulary"][i] + [1])
            transformed_data["hypotheses_part_of_speech"].append([17] + data["hypotheses_part_of_speech"][i] + [17])
            transformed_data["hypotheses_out_of_vocabulary"].append([0] + data["hypotheses_out_of_vocabulary"][i] + [1])
            
            if label == "hidden":
                transformed_data["labels"].append(-1)
            else:
                transformed_data["labels"].append(self.labeldict[label])

            indices = self.words_to_indices(premise)
            transformed_data["premises"].append(indices)

            indices = self.words_to_indices(data["hypotheses"][i])
            transformed_data["hypotheses"].append(indices)

        return transformed_data

    def build_embedding_matrix(self, embeddings_file):
        """
        Build an embedding matrix with pretrained weights for object's
        worddict.

        Args:
            embeddings_file: A file containing pretrained word embeddings.

        Returns:
            A numpy matrix of size (num_words+n_special_tokens, embedding_dim)
            containing pretrained word embeddings (the +n_special_tokens is for
            the padding and out-of-vocabulary tokens, as well as BOS and EOS if
            they're used).
        """
        # Load the word embeddings in a dictionnary.
        embeddings = {}
        with open(embeddings_file, "r", encoding="utf8") as input_data:
            for line in input_data:
                line = line.split()

                try:
                    # Check that the second element on the line is the start
                    # of the embedding and not another word. Necessary to
                    # ignore multiple word lines.
                    float(line[1])
                    word = line[0]
                    if word in self.worddict:
                        embeddings[word] = line[1:]

                # Ignore lines corresponding to multiple words separated
                # by spaces.
                except ValueError:
                    continue

        num_words = len(self.worddict)
        embedding_dim = len(list(embeddings.values())[0])
        embedding_matrix = np.zeros((num_words, embedding_dim))

        # Actual building of the embedding matrix.
        missed = 0
        for word, i in self.worddict.items():
            if word in embeddings:
                embedding_matrix[i] = np.array(embeddings[word], dtype=float)
            else:
                if word == "_PAD_":
                    continue
                missed += 1
                # Out of vocabulary words are initialised with random gaussian
                # samples.
                embedding_matrix[i] = np.random.normal(size=(embedding_dim))
        print("Missed words: ", missed)

        return embedding_matrix


class FEVERDataset(Dataset):
    """
    Dataset class for Natural Language Inference datasets.

    The class can be used to read preprocessed datasets where the premises,
    hypotheses and labels have been transformed to unique integer indices
    (this can be done with the 'preprocess_data' script in the 'scripts'
    folder of this repository).
    """

    def __init__(self,
                 data,
                 padding_idx=0,
                 max_premise_length=None,
                 max_hypothesis_length=None,
                 max_premise_part_of_speech_length=None, 
                 max_premise_out_of_vocabulary_length=None,
                 max_hypothesis_part_of_speech_length=None,
                 max_hypothesis_out_of_vocabulary_length=None                 
                ):
        """
        Args:
            data: A dictionary containing the preprocessed premises,
                hypotheses and labels of some dataset.
            padding_idx: An integer indicating the index being used for the
                padding token in the preprocessed data. Defaults to 0.
            max_premise_length: An integer indicating the maximum length
                accepted for the sequences in the premises. If set to None,
                the length of the longest premise in 'data' is used.
                Defaults to None.
            max_hypothesis_length: An integer indicating the maximum length
                accepted for the sequences in the hypotheses. If set to None,
                the length of the longest hypothesis in 'data' is used.
                Defaults to None.
        """
        self.premises_lengths = [len(seq) for seq in data["premises"]]
        self.max_premise_length = max_premise_length
        if self.max_premise_length is None:
            self.max_premise_length = max(self.premises_lengths)

        self.hypotheses_lengths = [len(seq) for seq in data["hypotheses"]]
        self.max_hypothesis_length = max_hypothesis_length
        if self.max_hypothesis_length is None:
            self.max_hypothesis_length = max(self.hypotheses_lengths)
            
        self.premises_part_of_speech_lengths = [len(seq) for seq in data["premises_part_of_speech"]]
        self.max_premise_part_of_speech_length = max_premise_part_of_speech_length
        if self.max_premise_part_of_speech_length is None:
            self.max_premise_part_of_speech_length = max(self.premises_part_of_speech_lengths)
        
        self.premises_out_of_vocabulary_lengths = [len(seq) for seq in data["premises_out_of_vocabulary"]]
        self.max_premise_out_of_vocabulary_length = max_premise_out_of_vocabulary_length
        if self.max_premise_out_of_vocabulary_length is None:
            self.max_premise_out_of_vocabulary_length = max(self.premises_out_of_vocabulary_lengths)
        
        self.hypotheses_part_of_speech_lengths = [len(seq) for seq in data["hypotheses_part_of_speech"]]
        self.max_hypothesis_part_of_speech_length = max_hypothesis_part_of_speech_length
        if self.max_hypothesis_part_of_speech_length is None:
            self.max_hypothesis_part_of_speech_length = max(self.hypotheses_part_of_speech_lengths)
        
        self.hypotheses_out_of_vocabulary_lengths = [len(seq) for seq in data["hypotheses_out_of_vocabulary"]]
        self.max_hypothesis_out_of_vocabulary_length = max_hypothesis_out_of_vocabulary_length
        if self.max_hypothesis_out_of_vocabulary_length is None:
            self.max_hypothesis_out_of_vocabulary_length = max(self.hypotheses_out_of_vocabulary_lengths)

        self.num_sequences = len(data["premises"])
        
        part_of_speech_length = 1#len(data['premises_part_of_speech'][0][0])
        out_of_vocabulary_length = 1#len(data['premises_out_of_vocabulary'][0][0])
        self.data = {"ids": [],
                     "premises": torch.ones((self.num_sequences,
                                             self.max_premise_length),
                                            dtype=torch.long) * padding_idx,
                     "hypotheses": torch.ones((self.num_sequences,
                                               self.max_hypothesis_length),
                                              dtype=torch.long) * padding_idx,
                     "premises_part_of_speech": torch.ones((self.num_sequences,
                                               self.max_premise_part_of_speech_length),
                                              dtype=torch.long) * padding_idx,
                     "premises_out_of_vocabulary": torch.ones((self.num_sequences,
                                               self.max_premise_out_of_vocabulary_length),
                                              dtype=torch.long) * padding_idx,
                     "hypotheses_part_of_speech": torch.ones((self.num_sequences,
                                               self.max_hypothesis_part_of_speech_length),
                                              dtype=torch.long) * padding_idx,
                     "hypotheses_out_of_vocabulary": torch.ones((self.num_sequences,
                                               self.max_hypothesis_out_of_vocabulary_length),
                                              dtype=torch.long) * padding_idx,
                     "labels": torch.tensor(data["labels"], dtype=torch.long)}

        for i, premise in enumerate(data["premises"]):            
            self.data["ids"].append(data["ids"][i])
            
            end = min(len(premise), self.max_premise_length)
            self.data["premises"][i][:end] = torch.tensor(premise[:end])
            
            hypothesis = data["hypotheses"][i]
            end = min(len(hypothesis), self.max_hypothesis_length)
            self.data["hypotheses"][i][:end] = torch.tensor(hypothesis[:end])
                
            premise_part_of_speech = data["premises_part_of_speech"][i]
            end = min(len(premise_part_of_speech), self.max_premise_part_of_speech_length)
            self.data["premises_part_of_speech"][i][:end] = torch.tensor(premise_part_of_speech[:end])
            
            premise_out_of_vocabulary = data["premises_out_of_vocabulary"][i]
            end = min(len(premise_out_of_vocabulary), self.max_premise_out_of_vocabulary_length)
            self.data["premises_out_of_vocabulary"][i][:end] = torch.tensor(premise_out_of_vocabulary[:end])
            
            hypothesis_part_of_speech = data["hypotheses_part_of_speech"][i]
            end = min(len(hypothesis_part_of_speech), self.max_hypothesis_part_of_speech_length)
            self.data["hypotheses_part_of_speech"][i][:end] = torch.tensor(hypothesis_part_of_speech[:end])
            
            hypothesis_out_of_vocabulary = data["hypotheses_out_of_vocabulary"][i]
            end = min(len(hypothesis_out_of_vocabulary), self.max_hypothesis_out_of_vocabulary_length)
            self.data["hypotheses_out_of_vocabulary"][i][:end] = torch.tensor(hypothesis_out_of_vocabulary[:end])

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, index):
        return {"id": self.data["ids"][index],
                "premise": self.data["premises"][index],
                "premise_length": min(self.premises_lengths[index],
                                      self.max_premise_length),
                "hypothesis": self.data["hypotheses"][index],
                "hypothesis_length": min(self.hypotheses_lengths[index],
                                         self.max_hypothesis_length),
                "premise_part_of_speech": self.data["premises_part_of_speech"][index],
                "premise_part_of_speech_length": min(self.premises_part_of_speech_lengths[index],
                                      self.max_premise_part_of_speech_length),
                "premise_out_of_vocabulary": self.data["premises_out_of_vocabulary"][index],
                "premise_out_of_vocabulary_length": min(self.premises_out_of_vocabulary_lengths[index],
                                      self.max_premise_out_of_vocabulary_length),
                "hypothesis_part_of_speech": self.data["hypotheses_part_of_speech"][index],
                "hypothesis_part_of_speech_length": min(self.hypotheses_part_of_speech_lengths[index],
                                      self.max_hypothesis_part_of_speech_length),
                "hypothesis_out_of_vocabulary": self.data["hypotheses_out_of_vocabulary"][index],
                "hypothesis_out_of_vocabulary_length": min(self.hypotheses_out_of_vocabulary_lengths[index],
                                      self.max_hypothesis_out_of_vocabulary_length),
                "label": self.data["labels"][index]}

def convert_label_stage_2(input_label):
    if input_label == 'SUPPORTS':
        return 'contains_evidence'
    elif input_label == 'REFUTES':
        return 'contains_evidence'
    elif input_label == 'NOT ENOUGH INFO':
        return 'no_evidence'
    else:
        raise ValueError('input label not in options', input_label)

def dict_load_json(path_file):
    if os.path.isfile(path_file):
        with open(path_file, "r") as f:
            dictionary = json.load(f)
    else:
        raise ValueError('json file does not exist', path_file)
    return dictionary

def dict_save_json(dictionary, path_file):
    if os.path.isfile(path_file):
        print('overwriting file: %s'%(path_file))
    with open(path_file, "w") as f:
        json.dump(dictionary, f)