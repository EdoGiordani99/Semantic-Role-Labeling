import json
import torch
from collections import Counter
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from config import *


def build_labels_vocab(dataset: Dataset):
    """
    Args:
        dataset (Dataset): The SRL dataset
    Return:
        vocab (Dictionary): containing the mapping label --> idx
    """

    counter = Counter()

    for i in range(dataset.__len__()):
        for item in dataset.labels[i]['roles']:
            # roles for only one verb at a time
            roles = dataset.labels[i]['roles'][item]

            for role in roles:
                if role is not None:
                    counter[role] += 1

    vocab = {'<pad>': 0}
    for i, lab in enumerate(counter):
        vocab[lab] = i + 1

    print('Vocabulary created!')

    return vocab


def build_postags_vocab(dataset: Dataset):
    """
    Args:
        dataset (Dataset): The SRL dataset
    Return:
        vocab (Dictionary): containing the mapping pos_tag --> idx
    """

    counter = Counter()

    for sentence in dataset.sentences.values():
        tags = sentence['pos_tags']

        for tag in tags:
            if tag is not None:
                counter[tag] += 1

    vocab = {'<pad>': 0}
    for i, lab in enumerate(counter):
        vocab[lab] = i + 1

    print('POS tags vocabulary created!')

    return vocab


def collate_fn(batch):
    batch_out = {'input_ids': torch.LongTensor([s['input_ids'] for s in batch]),
                 'attention_mask': torch.LongTensor([s['attention_mask'] for s in batch]),
                 'predicates': torch.FloatTensor([s['predicates'] for s in batch]),
                 'pos_tags': torch.FloatTensor([s['pos_tags'] for s in batch]),
                 'labels': torch.LongTensor([s['labels'] for s in batch]),
                 'predicate_idx': torch.LongTensor([s['predicate_idx'] for s in batch])}

    return batch_out


class SRL_Aug_Dataset(Dataset):
    """
    This is a dataset class implementing Augmentation on the dataset
    Args:
        data_path (str): path to the .json dataset file
        toy_data (bool): if True, dataset will contain only 10 elements
    """

    def __init__(self, data_path: str, toy_data: bool = False):

        self.data_path = data_path
        self.toy_data = toy_data
        self.empty_roles = []

        self.sentences, self.labels = self.read_data()

        self.raw_data = None
        self.pos_tags = None

    def __len__(self):
        return self.len

    def __get_raw_item__(self, idx):
        sample = {'words': self.sentences[idx]['words'],
                  'predicates': self.sentences[idx]['predicates'],
                  'pos_tags': self.pos_tags[idx],
                  'srl_tags': self.labels[idx]}
        return sample

    def __getitem__(self, idx):
        sample = {'input_ids': self.out['input_ids'][idx].tolist(),
                  'attention_mask': self.out['attention_mask'][idx].tolist(),
                  'predicates': self.out['predicates'][idx],
                  'pos_tags': self.out['pos_tags'][idx],
                  'labels': self.out['labels'][idx],
                  'predicate_idx': self.out['predicate_idx'][idx]}
        return sample

    def read_data(self):
        """
        Reading the dataset, taking only the useful information such as words,
        predicates (inputs of the NN) and roles (ouput)
        """

        with open(self.data_path) as f:
            data = json.load(f)

        self.raw_data = data

        sentences, labels = {}, {}
        counter = 0
        err = 0

        for id, sentence in data.items():

            try:
                # some samples don't have the 'roles' labels
                labels[counter] = {'roles': sentence['roles']}
                sentences[counter] = {'words': sentence['words'],
                                      'pos_tags': sentence['pos_tags'],
                                      'predicates': sentence['predicates']
                                      }
                counter += 1
            except:
                self.empty_roles.append(id)
                err += 1

        print(f"\nWARNING: {err} samples have been discarted since no 'roles' was assigned\n")

        # checking if dimension are good
        if len(labels) == len(sentences):
            print('Reading successfully done!\n')
        else:
            raise ValueError("The 'labels' and 'sencences' list have different sizes")

        return sentences, labels

    def process_data(self, labels_vocab, postags_vocab):
        """
        This is a simple pipeline of data preprocessing
        """

        # Replicating sentences with multiples predicates
        self.sentences, self.labels = self.separate_predicates()

        # Encoding the labels and pos_tags according to their vocabs
        self.labels = self.encode_labels(labels_vocab)
        self.pos_tags = self.encode_postags(postags_vocab)

        if self.toy_data:
            self.sentences = self.sentences[:10]
            self.labels = self.labels[:10]
            self.pos_tags = self.pos_tags[:10]

        data = []
        for i in range(len(self.sentences)):
            data.append(self.__get_raw_item__(i))

        batch = self.augment()

        # adding to the original dataset, the augmented samples
        for i in range(len(self.sentences)):
            batch.append(self.__get_raw_item__(i))

        # Since batch processing requires some time, to speed up training phase
        # collate preprocessing is done before creating the data loaders.
        self.out = self.collate_fn(batch)
        self.len = len(batch)

        print(f'Dataset contains now {len(batch)} samples!')

    def augment(self):
        """
        It finds samples with low frequency labels (expect for the '_' one) and
        subdivide samples according to the contained label index. Those samples
        will be replicated according to a weight for each class.
        Returns:
            to_add (list of dict): new augmented samples that has to be added.
        """

        augment = {k: [] for k in range(4, 28, 1)}

        for i, label in enumerate(self.labels):
            asw, idx = self.check_aug_label(label)
            if asw:
                augment[idx].append(self.__get_raw_item__(i))

        weights = [2, 1, 2, 2, 2, 4, 3, 4, 4, 4, 8, 4, 4, 4, 8, 4, 10, 10, 10, 13, 6, 10, 10, 11]
        to_add = []

        for i, samples in enumerate(augment.values()):
            for j in range(weights[i]):
                to_add.extend(samples)

        return to_add

    def check_aug_label(self, label):
        """
        Tells whether the sample contains very frequent classes or not. If sample
        contains not frequent labels, retuns True and the contained label index.
        """
        for lab in label:
            if lab != 2 and lab != 1 and lab != 3 and lab != 0:
                return True, lab
            elif lab == 1 or lab == 3:
                return False, None

        return False, None

    def separate_predicates(self):
        """
        In this function we want to generate a dataset where for each sample
        is only specified one predicate and its relative roles.
        """
        new_sentences = []
        new_labels = []

        # check where verbs are
        verbs_ids = []

        for label in self.labels.values():
            verb_id = [int(id) for id in label['roles'].keys()]

            verbs_ids.append(verb_id)

        # count how many are them & replicate the sentence
        for i in range(len(self.sentences)):

            ids = verbs_ids[i]
            words = self.sentences[i]['words']
            predicate = self.sentences[i]['predicates']
            pos_tags = self.sentences[i]['pos_tags']

            for j in range(len(ids)):
                id = ids[j]

                # creating a verbs_vector with just one predicate
                new_predicate = ['_'] * len(predicate)
                new_predicate[id] = predicate[id]

                new_sentences.append({'words': words,
                                      'predicates': new_predicate,
                                      'pos_tags': pos_tags})
                new_labels.append(self.labels[i]['roles'][str(id)])

        # Checking dimensions, just for sure
        if len(new_sentences) != len(new_labels):
            raise ValueError("New sentences and New lables have different lenght")

        return new_sentences, new_labels

    def add_verb_info(self):
        """
        Adding to each sample the '[SEP]' token and the verb.
        """

        with_verb_info = []
        pos_tags = []

        for sentence in self.sentences:

            for i in range(len(sentence['words'])):

                if sentence['predicates'][i] != '_':
                    verb = sentence['words'][i]

            new = sentence['words'] + ['[SEP]', verb]
            with_verb_info.append(new)
            pos_tags.append(sentence['pos_tags'])

        return with_verb_info, pos_tags

    def encode_postags(self, postags_vocab):
        """
        Given the postags_vocabulary, we transform the pos tags into index.
        """
        encoded_tags = []

        for sentence in self.sentences:
            pos_tag = sentence['pos_tags']
            encoded_tags.append([postags_vocab[tag] for tag in pos_tag])

        return encoded_tags

    def encode_labels(self, labels_vocab):
        """
        Given the labels_vocabulary, we transform the labels into index.
        """
        encoded = []

        for sentence_labels in self.labels:
            encoded.append([labels_vocab[label] for label in sentence_labels])

        return encoded

    def collate_fn(self, batch):
        """
        Given the dataset, like it was a train dataloader batch, is processed to
        obtain a more compatible format.
        batch_out is a list of dictionaries. Each dictionaty contains: words,
        labels, predicate OHE, predicate idx and pos_tags.
        """

        batch_sentences = [sentence['words'] for sentence in batch]
        batch_predicates = [sentence['predicates'] for sentence in batch]
        tags = [sentence['pos_tags'] for sentence in batch]

        # Converting predicates into 1 hot encoding vectors
        predicates = []
        for sentence in batch_predicates:
            predicates.append([0 if pred == '_' else 1 for pred in sentence])

        # Tokenizing sentences
        tokenizer = AutoTokenizer.from_pretrained(LANGUAGE_MODEL_NAME)

        batch_out = tokenizer(batch_sentences,
                              return_tensors="pt",
                              padding=True,
                              is_split_into_words=True)

        labels = []
        pad_predicates = []
        pad_pos_tags = []

        srl_tags = [sentence['srl_tags'] for sentence in batch]

        for i, label in enumerate(srl_tags):

            # word_ids is a list which contains one index for each bert token specifing
            # to which word the token is refering. Example:
            # ['I', 'love', 'embeddings'] -tokenizer-> [ None, 1, 2, 3, 3, 3, None]
            # None is for [CLS] and [SEP] tokens (for the model)
            # notice how the "embeddings" word is divided into 3 tokens.

            word_ids = batch_out.word_ids(batch_index=i)
            predicate = predicates[i]

            previous_word_idx = None

            labels_ids = []
            pad_predicate = []
            pad_pos_tag = []

            for idx in word_ids:
                # We want to ignore the None labels (special tokens). By setting the
                # label to -100 loss function will automatically ignore them. We want
                # to keep the word labels but we also want to set -100 for the labels
                # of supplement sub-tokens. In the previous example:
                # [ None, 1, 2, 3, 3, 3, None ] --> [-100, 1, 2, 3, -100, -100, -100]

                if idx is None:
                    labels_ids.append(-100)
                    pad_predicate.append([0])
                    pad_pos_tag.append([0])

                elif idx != previous_word_idx:
                    labels_ids.append(label[idx])
                    pad_predicate.append([predicate[idx]])
                    pad_pos_tag.append([tags[i][idx]])

                else:
                    labels_ids.append(0)
                    pad_predicate.append([0])
                    pad_pos_tag.append([0])

                previous_word_idx = idx

            labels.append(labels_ids)
            pad_predicates.append(pad_predicate)
            pad_pos_tags.append(pad_pos_tag)

        # Padding the labels
        batch_longest_sentence = max(labels, key=len)
        batch_max_length = len(batch_longest_sentence)

        labels = [l + ([-100] * abs(batch_max_length - len(l))) for l in labels]

        batch_out["labels"] = labels
        batch_out['predicates'] = pad_predicates
        batch_out['pos_tags'] = pad_pos_tags

        # Extracting predicate index for each sentence
        predicate_idx = []
        for predicate in pad_predicates:
            for i, p in enumerate(predicate):
                if p == [1]:
                    predicate_idx.append(i)

        batch_out['predicate_idx'] = predicate_idx

        return batch_out


class SRL_Dataset(Dataset):

    def __init__(self, data_path: str, toy_data: bool = False):
        """
        Args:
            data_path (str): path to the .json dataset file
            toy_data (bool): if True, dataset will contain only 10 elements
        """

        self.data_path = data_path
        self.toy_data = toy_data
        self.empty_roles = []

        self.sentences, self.labels = self.read_data()

        self.raw_data = None
        self.pos_tags = None

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sample = {'input_ids': self.out['input_ids'][idx].tolist(),
                  'attention_mask': self.out['attention_mask'][idx].tolist(),
                  'predicates': self.out['predicates'][idx],
                  'pos_tags': self.out['pos_tags'][idx],
                  'labels': self.out['labels'][idx],
                  'predicate_idx': self.out['predicate_idx'][idx]}
        return sample

    def __get_raw_item__(self, idx):
        sample = {'words': self.sentences[idx]['words'],
                  'predicates': self.sentences[idx]['predicates'],
                  'pos_tags': self.pos_tags[idx],
                  'srl_tags': self.labels[idx]}
        return sample

    def read_data(self):
        """
        Extracting from the .json file all the useful informations such as:
        roles, words, pos_tags and predicates.
        """

        with open(self.data_path) as f:
            data = json.load(f)

        self.raw_data = data

        sentences, labels = {}, {}
        counter = 0
        err = 0

        for id, sentence in data.items():

            try:
                # some samples don't have the 'roles' labels
                labels[counter] = {'roles': sentence['roles']}
                sentences[counter] = {'words': sentence['words'],
                                      'pos_tags': sentence['pos_tags'],
                                      'predicates': sentence['predicates']
                                      }
                counter += 1
            except:
                self.empty_roles.append(id)
                err += 1

        print(f"\nWARNING: {err} samples have been discarted since no 'roles' was assigned\n")

        # checking any reading error
        if len(labels) == len(sentences):
            print('Reading successfully done!\n')
        else:
            raise ValueError("The 'labels' and 'sencences' list have different sizes")

        return sentences, labels

    def process_data(self, labels_vocab, postags_vocab):
        """
        This is a simple pipeline of data preprocessing
        """

        # Replicating sentences with multiples predicates
        self.sentences, self.labels = self.separate_predicates()

        # Encoding the labels and pos_tags according to their vocabs
        self.labels = self.encode_labels(labels_vocab)
        self.pos_tags = self.encode_postags(postags_vocab)

        if self.toy_data:
            self.sentences = self.sentences[:10]
            self.labels = self.labels[:10]
            self.pos_tags = self.pos_tags[:10]

        data = []
        for i in range(len(self.sentences)):
            data.append(self.__get_raw_item__(i))

        # Since batch processing requires some time, to speed up training phase
        # collate preprocessing is done before creating the data loaders.
        self.out = self.collate_fn(data)

        print(f'Dataset contains now {len(self.sentences)} samples!')

    def separate_predicates(self):
        """
        In this function we want to generate a dataset where for each sample
        is only specified one predicate and its relative roles.
        """
        new_sentences = []
        new_labels = []

        # check where verbs are
        verbs_ids = []

        for label in self.labels.values():
            verb_id = [int(id) for id in label['roles'].keys()]

            verbs_ids.append(verb_id)

        # count how many are them & replicate the sentence
        for i in range(len(self.sentences)):

            ids = verbs_ids[i]
            words = self.sentences[i]['words']
            predicate = self.sentences[i]['predicates']
            pos_tags = self.sentences[i]['pos_tags']

            for j in range(len(ids)):
                id = ids[j]

                # creating a verbs_vector with just one predicate
                new_predicate = ['_'] * len(predicate)
                new_predicate[id] = predicate[id]

                new_sentences.append({'words': words,
                                      'predicates': new_predicate,
                                      'pos_tags': pos_tags})
                new_labels.append(self.labels[i]['roles'][str(id)])

        # Checking dimensions, just for sure
        if len(new_sentences) != len(new_labels):
            raise ValueError("New sentences and New lables have different lenght")

        return new_sentences, new_labels

    def add_verb_info(self):
        """
        Adding to each sample the '[SEP]' token and the verb.
        """

        with_verb_info = []
        pos_tags = []

        for sentence in self.sentences:

            for i in range(len(sentence['words'])):

                if sentence['predicates'][i] != '_':
                    verb = sentence['words'][i]

            new = sentence['words'] + ['[SEP]', verb]
            with_verb_info.append(new)
            pos_tags.append(sentence['pos_tags'])

        return with_verb_info, pos_tags

    def encode_postags(self, postags_vocab):
        """
        Given the postags_vocabulary, we transform the pos tags into index.
        """
        encoded_tags = []

        for sentence in self.sentences:
            pos_tag = sentence['pos_tags']
            encoded_tags.append([postags_vocab[tag] for tag in pos_tag])

        return encoded_tags

    def encode_labels(self, labels_vocab):
        """
        Given the labels_vocabulary, we transform the labels into index.
        """
        encoded = []

        for sentence_labels in self.labels:
            encoded.append([labels_vocab[label] for label in sentence_labels])

        return encoded

    def collate_fn(self, batch):
        """
        Given the dataset, like it was a train dataloader batch, is processed to
        obtain a more compatible format.
        batch_out is a list of dictionaries. Each dictionaty contains: words,
        labels, predicate OHE, predicate idx and pos_tags.
        """

        batch_sentences = [sentence['words'] for sentence in batch]
        batch_predicates = [sentence['predicates'] for sentence in batch]
        tags = [sentence['pos_tags'] for sentence in batch]

        # Converting predicates into 1 hot encoding vectors
        predicates = []
        for sentence in batch_predicates:
            predicates.append([0 if pred == '_' else 1 for pred in sentence])

        # Tokenizing sentences
        tokenizer = AutoTokenizer.from_pretrained(LANGUAGE_MODEL_NAME)

        batch_out = tokenizer(batch_sentences,
                              return_tensors="pt",
                              padding=True,
                              is_split_into_words=True)

        labels = []
        pad_predicates = []
        pad_pos_tags = []

        srl_tags = [sentence['srl_tags'] for sentence in batch]

        for i, label in enumerate(srl_tags):

            # word_ids is a list which contains one index for each bert token specifing
            # to which word the token is refering. Example:
            # ['I', 'love', 'embeddings'] -tokenizer-> [ None, 1, 2, 3, 3, 3, None]
            # None is for [CLS] and [SEP] tokens (for the model)
            # notice how the "embeddings" word is divided into 3 tokens.

            word_ids = batch_out.word_ids(batch_index=i)
            predicate = predicates[i]

            previous_word_idx = None

            labels_ids = []
            pad_predicate = []
            pad_pos_tag = []

            for idx in word_ids:
                # We want to ignore the None labels (special tokens). By setting the
                # label to -100 loss function will automatically ignore them. We want
                # to keep the word labels but we also want to set -100 for the labels
                # of supplement sub-tokens. In the previous example:
                # [ None, 1, 2, 3, 3, 3, None ] --> [-100, 1, 2, 3, -100, -100, -100]

                if idx is None:
                    labels_ids.append(-100)
                    pad_predicate.append([0])
                    pad_pos_tag.append([0])

                elif idx != previous_word_idx:
                    labels_ids.append(label[idx])
                    pad_predicate.append([predicate[idx]])
                    pad_pos_tag.append([tags[i][idx]])

                else:
                    labels_ids.append(0)
                    pad_predicate.append([0])
                    pad_pos_tag.append([0])

                previous_word_idx = idx

            labels.append(labels_ids)
            pad_predicates.append(pad_predicate)
            pad_pos_tags.append(pad_pos_tag)

        # Padding the labels
        batch_longest_sentence = max(labels, key=len)
        batch_max_length = len(batch_longest_sentence)

        labels = [l + ([-100] * abs(batch_max_length - len(l))) for l in labels]

        batch_out["labels"] = labels
        batch_out['predicates'] = pad_predicates
        batch_out['pos_tags'] = pad_pos_tags

        # Extracting predicate index for each sentence
        predicate_idx = []
        for predicate in pad_predicates:
            for i, p in enumerate(predicate):
                if p == [1]:
                    predicate_idx.append(i)

        batch_out['predicate_idx'] = predicate_idx

        return batch_out
