import numpy as np
import os
import torch
import argparse
import torch.nn as nn
import torch.optim as optim
import random
import copy
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import nltk
from nltk.tokenize import word_tokenize
from Encoder_models import *
from utils import *


def MINDsmall_load(data_path, behaviors_fname="behaviors.tsv", news_fname="news.tsv"):
    train_behaviors = np.genfromtxt(fname=os.path.join(
        data_path, "MINDsmall_train", behaviors_fname), delimiter="\t", comments=None, dtype=str, encoding='UTF-8')
    test_behaviors = np.genfromtxt(fname=os.path.join(
        data_path, "MINDsmall_dev", behaviors_fname), delimiter="\t", comments=None, dtype=str, encoding='UTF-8')
    # print(train_behaviors.shape)
    train_Users_ID = train_behaviors[:, 0]
    train_browsed_data_raw = train_behaviors[:, 2]
    train_candidate_data_raw = train_behaviors[:, 3]

    train_newsdata = np.genfromtxt(fname=os.path.join(
        data_path, "MINDsmall_train", news_fname), delimiter="\t", comments=None, dtype=str, encoding='UTF-8')
    test_newsdata = np.genfromtxt(fname=os.path.join(
        data_path, "MINDsmall_dev", news_fname), delimiter="\t", comments=None, dtype=str, encoding='UTF-8')
    news = {}
    for line in train_newsdata:
        news[line[0]] = word_tokenize(line[3].lower())
    for line in test_newsdata:
        if line[0] not in news:
            news[line[0]] = word_tokenize(line[3].lower())
    newsindex = {'NULL': 0}
    for newsid in news:
        newsindex[newsid] = len(newsindex)
    with open(os.path.join(data_path, "news_id2index.txt"), 'w') as f:
        f.write(str(newsindex))
    word_dict = {'PADDING': 0}

    news_title = [[0]*30]

    for newsid in news:
        title = []
        for word in news[newsid]:
            if word not in word_dict:
                word_dict[word] = len(word_dict)
            title.append(word_dict[word])
        title = title[:30]
        news_title.append(title+[0]*(30-len(title)))

    news_title = np.array(news_title, dtype='int32')

    neg_pos_ratio = 4
    train_candidate = []
    train_label = []
    train_user_his = []

    for i in range(len(train_browsed_data_raw)):
        click_his_ids = [x for x in str(
            train_browsed_data_raw[i]).split()][-50:]
        click_his_index = [newsindex[x] for x in click_his_ids]
        candidate_data = np.array([x.split('-')
                                   for x in str(train_candidate_data_raw[i]).split()])

        news_id_in_impression = candidate_data[:, 0]
        news_label_in_impression = candidate_data[:, 1]

        pos_news_ids = news_id_in_impression[news_label_in_impression == '1']
        pos_news_index = np.array([newsindex[x] for x in pos_news_ids])
        neg_news_ids = news_id_in_impression[news_label_in_impression == '0']
        neg_news_index = np.array([newsindex[x] for x in neg_news_ids])
        for pos_news in pos_news_index:
            negd = gen_neg_samples_with_npratio(neg_news_index, neg_pos_ratio)
            negd.append(pos_news)
            candidate_label = [0]*neg_pos_ratio+[1]
            candidate_order = list(range(neg_pos_ratio+1))
            random.shuffle(candidate_order)
            candidate_shuffle = []
            candidate_label_shuffle = []
            for i in candidate_order:
                candidate_shuffle.append(negd[i])
                candidate_label_shuffle.append(candidate_label[i])
            train_candidate.append(candidate_shuffle)
            train_label.append(candidate_label_shuffle)
            train_user_his.append(
                click_his_index+[0]*(50-len(click_his_index)))
    test_Users_ID = test_behaviors[:, 0]
    test_browsed_data_raw = test_behaviors[:, 2]
    test_candidate_data_raw = test_behaviors[:, 3]

    test_candidate = []
    test_user_his = []
    test_index = []
    test_session_data = []

    for i in range(len(test_browsed_data_raw)):
        click_his_ids = [x for x in str(
            test_browsed_data_raw[i]).split()][-50:]
        click_his_index = [newsindex[x] for x in click_his_ids]
        candidate_data = np.array([x.split('-')
                                   for x in str(test_candidate_data_raw[i]).split()])

        news_id_in_impression = candidate_data[:, 0]
        news_label_in_impression = candidate_data[:, 1]

        alldoc_id = news_id_in_impression[news_label_in_impression == '1']
        alldoc_index = np.array([newsindex[x] for x in alldoc_id])
        neg_news_ids = news_id_in_impression[news_label_in_impression == '0']
        test_session_data.append([test_Users_ID[i], alldoc_id, neg_news_ids])
        index = []
        index.append(len(test_candidate))
        for doc in alldoc_index:
            test_candidate.append(doc)
            test_user_his.append(click_his_ids+[0]*(50-len(click_his_ids)))
        index = []
        index.append(len(test_candidate))
        test_index.append(index)

    train_candidate = np.array(train_candidate, dtype='int32')
    train_label = np.array(train_label, dtype='int32')
    train_user_his = np.array(train_user_his, dtype='int32')

    test_candidate = np.array(test_candidate, dtype='int32')
    test_user_his = np.array(test_user_his, dtype='int32')

    return train_candidate, train_label, train_user_his, test_candidate, \
        test_user_his, test_index, test_session_data, word_dict


class NRMS(nn.Module):
    def __init__(self, n_src_vocab, d_word_vec, n_head, d_k, d_v, d_model):
        super().__init__()

        self.src_word_emb = nn.Embedding(n_src_vocab, d_word_vec)
        self.title_encoder = Encoder(n_head, d_k, d_v, d_model)
        # Embedding, dropout, self-attention
        self.user_encoder = Encoder(n_head, d_k, d_v, d_model)

    def forward(self, ):



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="MINDsmall")
    parser.add_argument("--d_word_vec", type=int, default=300)
    parser.add_argument("--head_num", type=int, default=20)
    args = parser.parse_args()

    data_path = args.data_path
    d_word_vec = args.d_word_vec
    head_num = args.head_num

    train_candidate, train_label, train_user_his, test_candidate, \
        test_user_his, test_index, test_session_data, word_dict = MINDsmall_load(
            data_path)
