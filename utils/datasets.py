#! /usr/bin/env python
# -*- coding: utf-8 -*-
# __author__ = "Sponge_sy"
# Date: 2022/4/16

import csv
import json
import random
class Datasets():

    def __init__(self, dataset_name="", n_th_set=1):
        self.dataset_name = dataset_name
        self.train_path, self.dev_path, self.test_path = "", "", ""

        if (dataset_name in ['SST-2']):
            self.train_path = r"./datasets/SST-2/train.tsv"
            self.dev_path = r"./datasets/SST-2/dev.tsv"
            self.test_path = r"./datasets/SST-2/dev.tsv" # test set has no golden label
            self.label_num = 2

        # The metric of dataset
        if (dataset_name in ['SST-2']):
            self.metric = 'Acc'

    def load_data(self, filename, sample_num=-1, is_shuffle=False, random_seed=0):
        D = []

        if (self.dataset_name in ['SST-2']):
            with open(filename, encoding='utf-8') as f:
                for i, l in enumerate(f.readlines()):
                    if (i == 0):
                        continue
                    rows = l.strip().split('\t')
                    text = rows[-2]
                    label = rows[-1]
                    D.append((text, int(label)))

        # Shuffle the dataset.
        if (is_shuffle):
            random.seed(random_seed)
            random.shuffle(D)

        # Set the number of samples.
        if (sample_num == -1):
            # -1 for all the samples
            return D
        else:
            return D[:sample_num]
