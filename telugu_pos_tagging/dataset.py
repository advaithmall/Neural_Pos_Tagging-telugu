import torch
import pandas as pd
from collections import Counter
import re
import random
from conllu import parse
device = "cuda" if torch.cuda.is_available() else "cpu"

class PosDataset(torch.utils.data.Dataset):
    def __init__(self, args):
        self.args = args
        self.words = self.load_words(self)
        self.uniq_words = self.get_uniq_words()
        self.tags = self.load_tags(self)
        self.uniq_tags = self.get_uniq_tags()
        self.index_to_word = {index: word for index,
                              word in enumerate(self.uniq_words)}
        self.word_to_index = {word: index for index,
                              word in enumerate(self.uniq_words)}
        self.tag_to_index = {tag: index for index,
                             tag in enumerate(self.uniq_tags)}
        self.index_to_tag = {index: tag for index,
                             tag in enumerate(self.uniq_tags)}
        self.word_indexes = [self.word_to_index.get(
            w, self.word_to_index["<unk>"]) for w in self.words]
        self.tag_indexes = [self.tag_to_index.get(t, 1) for t in self.tags]
        self.vocab_size = len(self.uniq_words)
        self.tag_size = len(self.uniq_tags)

    def load_words(self, args):
        random.seed(0)
        #write a function to replace 1 percent of random words with <unk> without using random
        
        def replace_with_unk(lst):
            # calculate the number of items to replace (0.5% of list length)
            num_items_to_replace = len(lst) // 200
            # randomly select the indices to replace
            idxs_to_replace = random.sample(
                range(len(lst)), num_items_to_replace)
            for idx in idxs_to_replace:
                # replace the item at the selected index with "<unk>"
                lst[idx] = "<unk>"
            return lst
        file_name = "te_mtg-ud-" + self.args.dataset + ".conllu"
        file_1 = open(file_name)
        data = file_1.read()
        parsed = parse(data)
        list_of_words = list()
        for i in range(self.args.sequence_length):
            list_of_words.append("<pad>")
        for i in range(0, len(parsed)):
            for j in parsed[i]:
                list_of_words.append(j['form'])
          #write an expression to replace 1 percent of random words with <unk>
        list_of_words = replace_with_unk(list_of_words)
        return list_of_words

    def load_tags(self, args):
        file_name = "te_mtg-ud-" + self.args.dataset + ".conllu"
        file_1 = open(file_name)
        data = file_1.read()
        parsed = parse(data)
        list_of_tags = list()
        for i in range(self.args.sequence_length):
            list_of_tags.append("<pad>")
        for i in range(0, len(parsed)):
            for j in parsed[i]:
                list_of_tags.append(j['upostag'])
        return list_of_tags

    def get_uniq_words(self):
        words_all = dict()
        file_name = "te_mtg-ud-" + "train" + ".conllu"
        file_1 = open(file_name)
        data = file_1.read()
        parsed = parse(data)
        for i in range(0, len(parsed)):
            for j in parsed[i]:
                words_all[j['form']] = 1
        words_all['<pad>'] = 1
        words_all["<unk>"] = 1
        return list(set(words_all.keys()))

    def get_uniq_tags(self):
        #write a function to get all the tags
        tags_all = dict()
        file_name = "te_mtg-ud-" + "train" + ".conllu"
        file_1 = open(file_name)
        data = file_1.read()
        parsed = parse(data)
        for i in range(0, len(parsed)):
            for j in parsed[i]:
                tags_all[j['upostag']] = 1
        file_name = "te_mtg-ud-" + "dev" + ".conllu"
        file_1 = open(file_name)
        data = file_1.read()
        parsed = parse(data)
        for i in range(0, len(parsed)):
            for j in parsed[i]:
                tags_all[j['upostag']] = 1
        file_name = "te_mtg-ud-" + "test" + ".conllu"
        file_1 = open(file_name)
        data = file_1.read()
        parsed = parse(data)
        for i in range(0, len(parsed)):
            for j in parsed[i]:
                tags_all[j['upostag']] = 1
        tags_all['<pad>'] = 1
        return list(set(tags_all.keys()))
    # def get_uniq_words_f(self):
    #     #open the file and parse it

    def __len__(self):
        return len(self.words)

    def __getitem__(self, index):
        #print(len(self.word_indexes))
        #print(index, len(self.word_indexes[index:index+self.args.sequence_length]))
        try:
            return torch.tensor(self.word_indexes[index:index+self.args.sequence_length]).to(device), torch.tensor(self.tag_indexes[index+self.args.sequence_length-1]).to(device)
        except:
            index_s = index - self.args.sequence_length
            return torch.tensor(self.word_indexes[index_s:index_s+self.args.sequence_length]).to(device), torch.tensor(self.tag_indexes[index_s+self.args.sequence_length-1]).to(device)
