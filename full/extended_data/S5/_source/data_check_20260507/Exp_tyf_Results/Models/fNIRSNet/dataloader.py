import torch
import pandas as pd
import numpy as np
import scipy.io as scio
import os


def UFFT_subject_data(data_path, subject=1, sec=8, win_start=27, win_length=40, win_stride=13):












    data = []
    label = []
    END = win_start + (sec-1) * win_stride

    for num in range(subject, subject+1):
        name = data_path + '/' + str(num) + '/' + str(num) + '.xls'
        Hb_org = pd.read_excel(name, header=None, sheet_name=None)
        name = data_path + '/' + str(num) + '/' + str(num) + '_desc.xls'
        desc = pd.read_excel(name, header=None)

        Hb = []
        for i in range(1, 76):
            name = 'Sheet' + str(i)
            Hb.append(Hb_org[name].values)

        Hb = np.array(Hb).transpose((0, 2, 1))
        desc = np.array(desc)

        for i in range(75):
            win_data = []
            win_label = []
            start = win_start
            while(start <= END):
                win_data.append(Hb[i, :, start:start+win_length])
                win_label.append(desc[i][0]-1)
                start = start + win_stride

            data.append(win_data)
            label.append(win_label)

        print(str(num) + '  OK')

    data = np.array(data)
    label = np.array(label)


    return data, label


def MA_subject_data(path, sub):







    data = []
    label = []


    file_path = os.path.join(path, str(sub), str(sub)+'_desc.mat')
    signal_label = np.array(scio.loadmat(file_path)['label']).squeeze()
    for k in range(len(signal_label)):
        if signal_label[k] == 1:
            signal_label[k] = 0
        elif signal_label[k] == 2:
            signal_label[k] = 1


    for wins in range(9, 19):
        file_path = os.path.join(path, str(sub), str(wins) + '_oxy.mat')
        oxy = np.array(scio.loadmat(file_path)['signal']).transpose((2, 1, 0))[:, :, :30]
        file_path = os.path.join(path, str(sub), str(wins) + '_deoxy.mat')
        deoxy = np.array(scio.loadmat(file_path)['signal']).transpose((2, 1, 0))[:, :, :30]

        hb = np.concatenate((oxy, deoxy), axis=1)

        data.append(hb)
        label.append(signal_label)

    print(str(sub) + '  OK')
    data = np.array(data).transpose((1, 0, 2, 3))
    label = np.array(label).transpose((1, 0))


    return data, label


def KFold_train_test_set(sub_data, label, data_index, test_index, n_fold):
    train_index = np.setdiff1d(data_index, test_index[n_fold])
    X_train = sub_data[train_index]
    y_train = label[train_index]
    X_test = sub_data[test_index[n_fold]]
    y_test = label[test_index[n_fold]]

    T, W, C, S = X_train.shape
    X_train = X_train.reshape((T * W, 1, C, S))
    y_train = y_train.reshape((T * W))
    T, W, C, S = X_test.shape
    X_test = X_test.reshape((T * W, 1, C, S))
    y_test = y_test.reshape((T * W))

    return X_train, y_train, X_test, y_test


def LOSO_train_test_set(all_data, all_label, n_sub, task_id):
    if task_id == 0:
        all_sub = 30
    elif task_id == 1:
        all_sub = 29

    sub_index = [np.arange(all_sub)]
    train_index = np.setdiff1d(sub_index, n_sub)
    X_train = all_data[train_index]
    y_train = all_label[train_index]
    X_test = all_data[n_sub]
    y_test = all_label[n_sub]
    Sub, N, D, C, S = X_train.shape
    X_train = X_train.reshape((Sub * N, D, C, S))
    y_train = y_train.reshape((Sub * N))
    return X_train, y_train, X_test, y_test


def load_all_data(data_path, task_id):







    all_data = []
    all_label = []
    if task_id == 0:
        all_sub = 30
    elif task_id == 1:
        all_sub = 29

    for n_sub in range(1, all_sub + 1):
        if task_id == 0:
            sub_data, sub_label = UFFT_subject_data(data_path, subject=n_sub)
        elif task_id == 1:
            sub_data, sub_label = MA_subject_data(path=data_path, sub=n_sub)

        T, W, C, S = sub_data.shape
        sub_data = sub_data.reshape((T * W, 1, C, S))
        sub_label = sub_label.reshape((T * W))
        all_data.append(sub_data)
        all_label.append(sub_label)

    all_data = np.array(all_data)
    all_label = np.array(all_label)


    return all_data, all_label


class Dataset(torch.utils.data.Dataset):
    def __init__(self, feature, label, transform=True):
        self.feature = feature
        self.label = label
        self.transform = transform
        self.feature = torch.tensor(self.feature, dtype=torch.float)
        self.label = torch.tensor(self.label, dtype=torch.float)

    def __len__(self):
        return len(self.label)

    def __getitem__(self, item):
        if self.transform:
            mean, std = self.feature[item].mean(), self.feature[item].std()
            self.feature[item] = (self.feature[item] - mean) / std

        return self.feature[item], self.label[item]
