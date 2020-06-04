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
import torch.optim as optim


def MINDsmall_load(data_path, behaviors_fname="behaviors.tsv", news_fname="news.tsv"):
    train_behaviors = np.genfromtxt(fname=os.path.join(
        data_path, "MINDsmall_train", behaviors_fname), delimiter="\t", comments=None, dtype=str, encoding='UTF-8')
    test_behaviors = np.genfromtxt(fname=os.path.join(
        data_path, "MINDsmall_dev", behaviors_fname), delimiter="\t", comments=None, dtype=str, encoding='UTF-8')
    # print(train_behaviors.shape)
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

    test_browsed_data_raw = test_behaviors[:, 2]
    test_candidate_data_raw = test_behaviors[:, 3]

    test_candidate = []
    test_label = []
    test_user_his = []

    for i in range(len(test_browsed_data_raw)):
        click_his_ids = [x for x in str(
            test_browsed_data_raw[i]).split()][-50:]
        click_his_index = [newsindex[x] for x in click_his_ids]
        candidate_data = np.array([x.split('-')
                                   for x in str(test_candidate_data_raw[i]).split()])

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
            test_candidate.append(candidate_shuffle)
            test_label.append(candidate_label_shuffle)
            test_user_his.append(
                click_his_index+[0]*(50-len(click_his_index)))

    train_candidate = np.array(train_candidate, dtype='int32')
    train_label = np.array(train_label, dtype='int32')
    train_user_his = np.array(train_user_his, dtype='int32')

    test_candidate = np.array(test_candidate, dtype='int32')
    test_label = np.array(test_label, dtype='int32')
    test_user_his = np.array(test_user_his, dtype='int32')

    return train_candidate, train_label, train_user_his, test_candidate, test_label, \
        test_user_his, word_dict, news_title


class mind_dataset(torch.utils.data.Dataset):
    """Some Information about mind_dataset"""

    def __init__(self, candidate, user_his, news_title, label):
        super(mind_dataset, self).__init__()
        idlist = np.arange(len(label))
        np.random.shuffle(idlist)
        self.y = label
        self.candidate = candidate
        self.user_his = user_his
        self.news_title = news_title

    def __getitem__(self, index):
        return (
            self.news_title[self.candidate[index]],
            self.news_title[self.user_his[index]],
            self.y[index]
        )

    def __len__(self):
        return len(self.y)


class NRMS(nn.Module):
    def __init__(self, n_src_vocab, d_word_vec, n_head, d_k, d_v, d_model):
        super().__init__()

        self.src_word_emb = nn.Embedding(n_src_vocab, d_word_vec)
        # self.news_encoder = Encoder(n_head, d_k, d_v, d_model)

        self.news_encoder_list_his = nn.ModuleList(
            Encoder(n_head, d_k, d_v, d_model) for _ in range(50)
        )
        self.news_encoder_list_candidates = nn.ModuleList(
            Encoder(n_head, d_k, d_v, d_model) for _ in range(5)
        )

        self.user_encoder = Encoder(n_head, d_k, d_v, d_model)

    def forward(self, news_input, candidates):
        news_emb_seq = self.src_word_emb(news_input)
        cand_emb_seq = self.src_word_emb(candidates)

        news_enc_out = self.news_encoder_list_his[0](news_emb_seq[:, 0])
        news_enc_out = news_enc_out.view(
            1, news_enc_out.shape[0], news_enc_out.shape[1])
        for i, encoder in enumerate(self.news_encoder_list_his[1:]):
            tmp = encoder(news_emb_seq[:, i+1])
            tmp = tmp.view(1, tmp.shape[0], tmp.shape[1])
            news_enc_out = torch.cat((news_enc_out, tmp), dim=0)
        news_enc_out = news_enc_out.transpose(0, 1)

        user_rep = self.user_encoder(news_enc_out).reshape(-1, 1, 300)
        candidates_enc_out = self.news_encoder_list_candidates[0](
            cand_emb_seq[:, 0])
        candidates_enc_out = candidates_enc_out.view(
            1, candidates_enc_out.shape[0], candidates_enc_out.shape[1])
        for i, encoder in enumerate(self.news_encoder_list_candidates[1:]):
            tmp = encoder(cand_emb_seq[:, i+1])
            tmp = tmp.view(1, tmp.shape[0], tmp.shape[1])
            candidates_enc_out = torch.cat((candidates_enc_out, tmp), dim=0)
        candidates_enc_out = candidates_enc_out.transpose(0, 1).transpose(1, 2)
        # candi_vec = torch.tensor(candidates_enc_out).reshape(-1, 5)
        pred_score = F.softmax(torch.bmm(user_rep, candidates_enc_out))
        pred_score = pred_score.view(
            pred_score.shape[0], pred_score.shape[2])
        print(pred_score.shape)
        return pred_score


def train(model, traindataloader, testdataloader, batch_size, lr, epN, gpu):

    print("Training Strat......")
    optimizer = optim.Adam(params=model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epN):
        running_loss = 0.0
        for i, data in enumerate(traindataloader, 0):
            candidates, news_input, labels = data
            candidates = candidates.long()
            news_input = news_input.long()
            # candidates: 64*5*30
            # news_input: 64*50*30
            # lables
            if i == 1:
                print(labels.shape)
            
            if gpu:
                news_input.cuda()
                candidates.cuda()
                labels.cuda()
                criterion.cuda()

            optimizer.zero_grad()

            outputs = model(news_input, candidates)
            print(outputs.shape, labels.shape)
            loss = criterion(outputs, torch.max(labels, 1)[1])
            
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print("Epoch: {}".format(epoch))
        print("Loss: {}".format(running_loss))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="../MINDsmall")
    parser.add_argument("--d_word_vec", type=int, default=300)
    parser.add_argument("--head_num", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epoch_num", type=float, default=10)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    data_path = args.data_path
    d_word_vec = args.d_word_vec
    head_num = args.head_num
    batch_size = args.batch_size
    lr = args.lr
    epN = args.epoch_num
    gpu = args.gpu
    
    train_candidate, train_label, train_user_his, test_candidate, test_label, \
    test_user_his, word_dict, news_title = MINDsmall_load(data_path)
    model = NRMS(len(word_dict), 300, 16, 16, 16, 300)
    train_set = mind_dataset(
        train_candidate, train_user_his, news_title, train_label)
    traindataloader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True)

    test_set = mind_dataset(
        test_candidate, test_user_his, news_title, test_label)
    testdataloader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False)
    train(model, traindataloader, testdataloader, batch_size, lr, epN, gpu)
    '''
    model = NRMS(100000, 300, 16, 16, 16, 300)
    test_news_input = torch.randint(0, 1000, (64, 50, 30))
    test_candidates_input = torch.randint(0, 1000, (64, 5, 30))
    model.forward(test_news_input, test_candidates_input)
    '''
