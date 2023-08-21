from random import random

import dgl
import torch
import numpy as np
import torch.nn as nn
import networkx as nx
import dgl.function as fn
import torch.nn.functional as F
from torch.optim import *
import matplotlib as mpl
# mpl.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from numpy import *
from torch import optim, nonzero
from sklearn.metrics import auc
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from dgl.nn.pytorch import edge_softmax, GATConv

import torch.utils.data as Data
from numpy import mat, matrix, vstack
from torch.autograd import Variable
from numpy import ndarray, eye, matmul, vstack, hstack, array, newaxis, zeros, genfromtxt, savetxt, exp

import os
from pandas import DataFrame as df
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score
import math
import sortscore

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

seed = 5
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)  
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

path = "/home/tea_yiming/guoyuwei/lab/GAT3/data/"

def ReadTxt():
    dis_sim = pd.read_excel(path + 'dis_sim.xlsx', header=None)
    dis_sim = array(dis_sim)
    circ_sim = pd.read_excel(path + 'circ_sim.xlsx', header=None)
    circ_sim = circ_sim.abs()
    circ_sim = array(circ_sim)
    circ_dis = pd.read_excel(path + 'circ_dis.xlsx', header=None)
    circ_dis = array(circ_dis)
    mi_dis = pd.read_excel(path + 'mi_dis.xlsx', header=None)
    mi_dis = array(mi_dis)
    circ_mi = pd.read_excel(path + 'circ_mi.xlsx', header=None)
    circ_mi = array(circ_mi)

    return dis_sim, circ_sim, circ_dis, mi_dis, circ_mi

def count_negative(rel_matrix,circ_dis, num):
    index_0 = (np.where(rel_matrix == 0))
    rel_negative_list = list(zip(index_0[0], index_0[1]))
    random.shuffle(rel_negative_list)  # 打乱
    negative_train = rel_negative_list[0:num]

    all_negative_list = rel_negative_list[num:]

    return all_negative_list, negative_train


def make_train_test_set(circrna_disease_matrix):
    index_tuple = (np.where(circrna_disease_matrix == 1))
    one_list = list(zip(index_tuple[0], index_tuple[1]))
    random.shuffle(one_list)
    split = math.ceil(len(one_list) / 5)

    train_set_list_total = []
    test_set_list_total = []
    for i in range(0, len(one_list), split):
        test_index = one_list[i:i + split]
        new_circrna_disease_matrix = circrna_disease_matrix.copy()

        for index in test_index:
            new_circrna_disease_matrix[index[0], index[1]] = 0
        roc_circrna_disease_matrix = new_circrna_disease_matrix + circrna_disease_matrix

        index_2 = (np.where(roc_circrna_disease_matrix == 2))
        positive_train = list(zip(index_2[0], index_2[1]))
        index_1 = (np.where(roc_circrna_disease_matrix == 1))
        positive_test = list(zip(index_1[0], index_1[1]))

        all_negative_test, negative_train = count_negative(rel_matrix,circ_dis, len(positive_train))

        train_set = positive_train + negative_train
        train_set_list_total.append(train_set)
        test_set = positive_test + all_negative_test
        test_set_list_total.append(test_set)
    return train_set_list_total,test_set_list_total

def Make_One_Features(dis_sim: ndarray, circ_sim: ndarray, circ_dis: ndarray, mi_dis: ndarray, circ_mi: ndarray, X: int,
                      Y: int):
    a1 = circ_sim[X]
    b1 = circ_dis[X]
    c1 = circ_mi[X]
    A1 = np.hstack((a1, b1, c1))

    a2 = circ_dis[:, Y]
    b2 = dis_sim[:, Y]
    c2 = mi_dis[:, Y]
    B1 = np.hstack((a2, b2, c2))

    C1 = np.vstack((A1, B1))

    return C1




def Make_Tow_Graph(circ_sim: ndarray, dis_sim: ndarray):

    g_circRNA = dgl.DGLGraph().to(device)
    g_circRNA.add_nodes(889)
    for i in range(circ_sim.shape[0]):
        for j in range(circ_sim.shape[1]):
            if circ_sim[i][j] > 0.5:
                g_circRNA.add_edges(i, j)

    g_Dise = dgl.DGLGraph().to(device)
    g_Dise.add_nodes(84)
    for m in range(dis_sim.shape[0]):
        for n in range(dis_sim.shape[1]):
            if dis_sim[m][n] > 0.8:
                g_Dise.add_edges(m, n)

    print(g_circRNA.number_of_nodes())
    print(g_circRNA.number_of_edges())
    print(g_circRNA.node_attr_schemes())
    print(g_circRNA.edge_attr_schemes())

    print(g_Dise.number_of_nodes())
    print(g_Dise.number_of_edges())
    print(g_Dise.node_attr_schemes())
    print(g_Dise.edge_attr_schemes())

    return g_circRNA, g_Dise



def Make_Tow_Graph_Feature(dis_sim: ndarray, circ_sim: ndarray, circ_dis: ndarray, mi_dis: ndarray, circ_mi: ndarray):
    circRNA_Feature = np.hstack((circ_sim, circ_dis, circ_mi))
    Dise_Feature = np.hstack((circ_dis.T, dis_sim, mi_dis.T))

    return circRNA_Feature, Dise_Feature



class My_Dataset(Dataset):

    def __init__(self, dis_sim, circ_sim, circ_dis, mi_dis, circ_mi, matrix):
        self.dis_sim = dis_sim
        self.circ_sim = circ_sim
        self.circ_dis = circ_dis
        self.mi_dis = mi_dis
        self.circ_mi = circ_mi
        self.matrix = matrix

    def __getitem__(self, idx):
        X, Y = self.matrix[idx]
        feature_map = Make_One_Features(self.dis_sim, self.circ_sim, self.circ_dis, self.mi_dis, self.circ_mi, X, Y)
        label = self.circ_dis[X][Y]

        return X, Y, feature_map, label


    def __len__(self):
        return len(self.matrix)


class My_CNN_Tow(nn.Module):
    def __init__(self):
        super(My_CNN_Tow, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=2,
                stride=1,
                padding=1,
            ),

            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1, padding=0),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 16, 2, 1, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 1, 0),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)

        return x



class My_CNN(nn.Module):
    def __init__(self):
        super(My_CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=2,
                stride=1,
                padding=1,
            ),

            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1, padding=0),
        )

    def forward(self, x):
        x = self.conv1(x)

        return x


class Attention_feature_level(nn.Module):
    def __init__(self, input_size: int, output_size: int):
        super(Attention_feature_level, self).__init__()

        self.node = nn.Linear(input_size, output_size, bias=True)
        torch.manual_seed(seed)
        nn.init.xavier_normal_(self.node.weight)

        self.h_n_parameters = nn.Parameter(torch.randn(output_size, input_size))
        torch.manual_seed(seed)
        nn.init.xavier_normal_(self.h_n_parameters)

    def forward(self, h_n_states):

        temp_nodes = self.node(h_n_states)
        temp_nodes = torch.tanh(temp_nodes)

        nodes_score = torch.matmul(temp_nodes, self.h_n_parameters)

        alpha = F.softmax(nodes_score, dim=2)

        y_i = alpha * h_n_states

        return y_i


class My_FCN(nn.Module):
    def __init__(self):
        super(My_FCN, self).__init__()
        self.out = nn.Sequential(nn.Linear(16 * 2 * 3636, 2),
                                 nn.Dropout(0.5),
                                 nn.Sigmoid())

    def forward(self, x):
        x = x.view(x.size(0), -1)
        out = self.out(x).float()
        return out


class Attention_model(nn.Module):

    def __init__(self, input_size_circ, input_size_A, input_size_circmi, input_size_dis, input_size_midis,
                 output_size1, output_size2, output_size3, output_size4, output_size5, output_size6, batch_size):
        super(Attention_model, self).__init__()

        self.attention_ls = Attention_feature_level(input_size_circ, output_size1)
        self.attention_A = Attention_feature_level(input_size_A, output_size2)
        self.attention_lm = Attention_feature_level(input_size_circmi, output_size3)
        self.attention_AT = Attention_feature_level(input_size_circ, output_size4)
        self.attention_ds = Attention_feature_level(input_size_dis, output_size5)
        self.attention_dm = Attention_feature_level(input_size_midis, output_size6)

        self.My_CNN_Tow = My_CNN_Tow()
        self.My_CNN = My_CNN()
        self.My_FCN = My_FCN()

    def forward(self, x):

        ls = x[:, :, 0, :889]
        A = x[:, :, 0, 889:973]
        lm = x[:, :, 0, 973:3636]

        AT = x[:, :, 1, :889]
        ds = x[:, :, 1, 889:973]
        dm = x[:, :, 1, 973:3636]


        result_ls = self.attention_ls(ls)
        result_A = self.attention_A(A)
        result_lm = self.attention_lm(lm)

        result_AT = self.attention_AT(AT)
        result_ds = self.attention_ds(ds)
        result_dm = self.attention_dm(dm)

        circ_RNA = torch.cat((result_ls, result_A, result_lm), dim=2)
        disease = torch.cat((result_AT, result_ds, result_dm), dim=2)
        reslut = torch.cat((circ_RNA, disease), dim=1)
        reslut = reslut.unsqueeze(dim=1)

        out1 = self.My_CNN(reslut)

        out2 = self.My_CNN_Tow(out1)


        out3 = out1 + out2


        out = self.My_FCN(out3)

        return out


class GAT(nn.Module):
    def __init__(self,
                 g,
                 in_dim,
                 output1,
                 output2,
                 output3,
                 heads,
                 activation,
                 num_items):

        super(GAT, self).__init__()
        self.g = g
        self.gat_layers = nn.ModuleList()
        self.activation = activation
        self.output1 = output1
        self.output2 = output2
        self.output3 = output3
        self.num_items = num_items
        self.gat_layers.append(GATConv(in_feats=in_dim, out_feats=output1,
                                       num_heads=heads, activation=self.activation))

    def forward(self, inputs):
        h = inputs
        h = self.gat_layers[0](self.g, h)

        association_matrix = torch.bmm(h, h.mT)
        softmax_score = F.softmax(association_matrix, dim=2)
        juzhen = torch.bmm(softmax_score, h) + h
        sigmoid_phat = torch.sigmoid(juzhen)

        T1_output = sigmoid_phat.mean(1)

        return T1_output


class My_FCN_GAT(nn.Module):
    def __init__(self):
        super(My_FCN_GAT, self).__init__()
        self.out = nn.Sequential(nn.Linear(2 * 1000, 2),
                                 nn.Dropout(0.2),
                                 nn.Sigmoid())

    def forward(self, x):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        x = x.type(torch.float).to(device)

        out = self.out(x)
        return out



class My_model(nn.Module):
    def __init__(self, g_circRNA, g_Dise,in_dim,output1,output2,output3,heads, activation,
                 circRNA_Feature, Dise_Feature,input_size_circ,input_size_dis):
        super(My_model, self).__init__()

        self.GAT_Module_circRNA = GAT(g_circRNA,in_dim,output1,output2,output3, heads,activation,input_size_circ)
        self.GAT_Module_Dis = GAT(g_Dise,in_dim, output1,output2,output3,heads, activation,input_size_dis)

        self.circRNA_Feature = circRNA_Feature
        self.Dise_Feature = Dise_Feature

        self.My_FCN_GAT = My_FCN_GAT()

    def forward(self, X, Y):
        result_circ = self.GAT_Module_circRNA(self.circRNA_Feature)
        result_Dis = self.GAT_Module_Dis(self.Dise_Feature)

        circ = result_circ[X]
        dis = result_Dis[Y]
        result_FCN = self.My_FCN_GAT(torch.cat((circ, dis), dim=1))

        return result_FCN


if __name__ == '__main__':

    dis_sim, circ_sim, circ_dis, mi_dis, circ_mi = ReadTxt()
    circ_dis = circ_dis.astype('float32')
    Score_circ_dis = circ_dis / 1

    Score_left = circ_dis / 1
    Score_right = circ_dis / 1
    Circ_dis = circ_dis / 1

    circRNA_Feature, Dise_Feature = Make_Tow_Graph_Feature(dis_sim, circ_sim, circ_dis, mi_dis, circ_mi)
    circRNA_Feature = torch.from_numpy(circRNA_Feature).float().to(device)
    print(circRNA_Feature)
    Dise_Feature = torch.from_numpy(Dise_Feature).float().to(device)

    g_circRNA, g_Dise = Make_Tow_Graph(circ_sim, dis_sim)

    learning_rate = 1e-3
    batch_size = 64
    num_epoches_right = 200
    num_epoches_left = 200
    r = 0.5

    input_size_circ = 889
    input_size_A = 84
    input_size_circmi = 2663
    input_size_dis = 84
    input_size_midis = 2663

    output_size1 = 889
    output_size2 = 84
    output_size3 = 2663
    output_size4 = 889
    output_size5 = 84
    output_size6 = 2663

    in_dim = 3636
    output1 = 1000
    output2 = 800
    output3 = 600
    heads = 3
    activation = F.sigmoid

    outputs_right_all = []
    outputs_left_all = []
    train_set_list_total, test_set_list_total = make_train_test_set(circ_dis)


    for b in range(len(train_set_list_total)):
        print(torch.cuda.device_count())
        Coordinate_Matrix_Train = np.array(train_set_list_total[b])
        Coordinate_Matrix_Test = np.array(test_set_list_total[b])
        train_set = My_Dataset(dis_sim, circ_sim, circ_dis, mi_dis, circ_mi, Coordinate_Matrix_Train)
        test_set = My_Dataset(dis_sim, circ_sim, circ_dis, mi_dis, circ_mi, Coordinate_Matrix_Test)
        train_data = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        test_data = DataLoader(test_set, batch_size=batch_size, shuffle=False)


        model_right = Attention_model(input_size_circ, input_size_A, input_size_circmi, input_size_dis,
                                      input_size_midis,
                                      output_size1, output_size2, output_size3, output_size4, output_size5,
                                      output_size6,
                                      batch_size).to(device)
        print(next(model_right.parameters()).device)

        criterion_right = nn.CrossEntropyLoss()
        criterion_right.to(device)
        optimizer_right = optim.SGD(model_right.parameters(), lr=learning_rate)
        scheduler = lr_scheduler.StepLR(optimizer_right, step_size=2, gamma=0.95)


        ave_loss_train_right_list = []
        model_right = model_right.train()

        for epoch in range(num_epoches_right):
            total_loss_train = 0
            total_step_train = 0
            for step, (_, _, x, y) in enumerate(train_data):
                feature = Variable(x).float().to(device)
                feature = feature.unsqueeze(1)
                label = Variable(y).long().to(device)

                output_right = model_right(feature)

                loss_right = criterion_right(output_right, label)
                total_loss_train = loss_right + total_loss_train
                total_step_train = total_step_train + 1

                optimizer_right.zero_grad()
                loss_right.backward()
                optimizer_right.step()

            ave_loss_train_right_epoch = total_loss_train / total_step_train
            ave_loss_train_right_epoch = ave_loss_train_right_epoch.cpu().detach().numpy()
            ave_loss_train_right_list.append(ave_loss_train_right_epoch)

            scheduler.step()



        model_right.eval()
        label_list_right = []
        output_list_right = []
        with torch.no_grad():
            for step, (_, _, x, y,) in enumerate(test_data):
                label_list_right.append(y.numpy())

                feature = Variable(x).float().to(device)
                feature = feature.unsqueeze(1)
                label = Variable(y).long().to(device)

                output_right = model_right(feature)
                output_right = F.softmax(output_right, dim=1)
                output_list_right.append(output_right.cpu().detach().numpy())
                loss_right = criterion_right(output_right, label)


        outputs_right = []
        for out in output_list_right:
            for i in range(out.shape[0]):
                outputs_right.append(out[i])
        outputs_right = np.array(outputs_right)
        outputs_right_all.append(outputs_right)


        ############################################################################################

        model_left = My_model(g_circRNA, g_Dise, in_dim, output1, output2, output3, heads, activation,
                              circRNA_Feature, Dise_Feature, input_size_circ, input_size_dis).to(device)
        print(next(model_right.parameters()).device)


        criterion_left = nn.CrossEntropyLoss()
        criterion_left.to(device)
        optimizer_left = optim.SGD(model_left.parameters(), lr=1e-2)
        scheduler = lr_scheduler.StepLR(optimizer_left, step_size=2, gamma=0.90)



        ave_loss_train_left_list = []
        model_left = model_left.train()
        for epoch in range(num_epoches_left):

            total_loss_train1 = 0
            total_step_train1 = 0
            for step, (X, Y, _, label) in enumerate(train_data):
                label = label.long().to(device)
                X = X.to(device)
                Y = Y.to(device)
                output_left = model_left(X, Y)
                loss_left = criterion_left(output_left, label)
                total_loss_train1 += loss_left
                total_step_train1 += 1

                optimizer_left.zero_grad()
                loss_left.backward()
                optimizer_left.step()

            ave_loss_train_left_epoch = total_loss_train1 / total_step_train1
            ave_loss_train_left_epoch = ave_loss_train_left_epoch.cpu().detach().numpy()
            ave_loss_train_left_list.append(ave_loss_train_left_epoch)

            scheduler.step()


        model_left.eval()
        label_list_left = []
        output_list_left = []
        with torch.no_grad():
            for step, (X, Y, _, label) in enumerate(test_data):

                label_list_left.append(label.numpy())
                label = label.long().to(device)
                X = X.to(device)
                Y = Y.to(device)
                output_left = model_left(X, Y)
                output_left = F.softmax(output_left, dim=1).to(device)
                output_list_left.append(output_left.cpu().detach().numpy())


        outputs_left = []
        for out in output_list_left:
            for i in range(out.shape[0]):
                outputs_left.append(out[i])
        outputs_left = np.array(outputs_left)
        outputs_left_all.append(outputs_left)


    fig = plt.figure(figsize=(16, 8), dpi=120)
    grid = plt.GridSpec(1, 2, hspace=0.3, wspace=0.5)
    ax_1 = fig.add_subplot(grid[0:1, 0:1])
    ax_2 = fig.add_subplot(grid[0:1, 1:2])

    tpr_list = []
    fpr_list = []
    ap_list = []
    precision_list = []
    recall_list = []


    for k in range(len(train_set_list_total)):
        Coordinate_Matrix_Train = np.array(train_set_list_total[k])
        Coordinate_Matrix_Test = np.array(test_set_list_total[k])

        for j in range(Coordinate_Matrix_Test.shape[0]):
            circ_dis[Coordinate_Matrix_Test[j][0]][Coordinate_Matrix_Test[j][1]] = r * outputs_right_all[k][j][1] + (
                        1 - r) * \
                                                                                   outputs_left_all[k][j][1]


        for i in range(Coordinate_Matrix_Train.shape[0]):
            circ_dis[Coordinate_Matrix_Train[i][0]][Coordinate_Matrix_Train[i][1]] = -1


        test_sample_loc = array(np.where(circ_dis > 0))
        test_sample_loc = test_sample_loc.T
        print(test_sample_loc.shape)

        val_list = []
        y_true_list = []

        for location in test_sample_loc.tolist():
            val = circ_dis[location[0], location[1]]
            val_list.append(val)
            label = Circ_dis[location[0], location[1]]
            y_true_list.append(label)
        print(y_true_list)


        y_true = np.array(val_list)
        y_label = np.array(y_true_list)
        fpr, tpr, thresholds_1 = metrics.roc_curve(y_label, y_true)
        precision, recall, thresholds_2 = precision_recall_curve(y_label, y_true)

        roc_auc = auc(fpr, tpr)
        AP = average_precision_score(y_label, y_true, average='macro', pos_label=1, sample_weight=None)