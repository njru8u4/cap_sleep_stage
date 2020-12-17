import numpy as np
from sklearn.preprocessing import OneHotEncoder

def get_data(train_data, label_data, classes):
    out_train = []
    out_label = []
    for i in range(len(label_data)):
        if classes == 5:
            if label_data[i] == 0:
                out_train.append(train_data[i,:])
                out_label.append(0)
            elif label_data[i] == 1:
                out_train.append(train_data[i,:])
                out_label.append(1)
            elif label_data[i] == 2:
                out_train.append(train_data[i,:])
                out_label.append(2)
            elif label_data[i] == 3 or label_data[i] == 4:
                out_train.append(train_data[i,:])
                out_label.append(3)
            elif label_data[i] == 5:
                out_train.append(train_data[i,:])
                out_label.append(4)
            else:
                continue
        elif classes == 4:
            if label_data[i] == 0:
                out_train.append(train_data[i,:])
                out_label.append(0)
            elif label_data[i] == 1 or label_data[i] == 2:
                out_train.append(train_data[i,:])
                out_label.append(1)
            elif label_data[i] == 3 or label_data[i] == 4:
                out_train.append(train_data[i,:])
                out_label.append(2)
            elif label_data[i] == 5:
                out_train.append(train_data[i,:])
                out_label.append(3)
            else:
                continue
    out_train = np.array(out_train)
    out_label = np.array(out_label)
    out_train = out_train.reshape(len(out_train), len(out_train[0]), 1)
    out_label = out_label.reshape(len(out_label), 1)
    onehot = OneHotEncoder()
    out_label = onehot.fit_transform(out_label).toarray()
    return out_train, out_label

def get_data_balance(train_data, label_data, classes = 4):
    W = []
    S1_2 = []
    S3_4 = []
    REM = []
    for i in range(len(label_data)):
        if label_data[i] == 0:
            W.append(train_data[i,:])
        elif label_data[i] == 1 or label_data[i] == 2:
            S1_2.append(train_data[i,:])
        elif label_data[i] == 3 or label_data[i] == 4:
            S3_4.append(train_data[i,:])
        elif label_data[i] == 5:
            REM.append(train_data[i,:])
        else:
            continue
    min_num = min([len(W), len(S1_2), len(S3_4), len(REM)])
    print (min_num)
    output_train = W[:min_num] + S1_2[:min_num] + S3_4[:min_num] + REM[:min_num]
    output_label = [0]*min_num + [1] * min_num + [2] * min_num + [3] * min_num
    out_train = np.array(output_train)
    out_label = np.array(output_label)
    out_train = out_train.reshape(len(out_train), len(out_train[0]), 1)
    out_label = out_label.reshape(len(out_label), 1)
    onehot = OneHotEncoder()
    out_label = onehot.fit_transform(out_label).toarray()
    #print (out_train.shape)
    #print (out_label.shape)
    return out_train, out_label

if __name__ == "__main__":
    inter = np.load("./datas/cap_PPG_150s_initial.npy")
    label = np.load("./datas/cap_label_for_PPG_150s_initial.npy")
    t1, t2 = get_data_balance(inter, label, 4)
    print (t1.shape)
    print (t2.shape)


