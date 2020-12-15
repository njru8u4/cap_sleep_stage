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

if __name__ == "__main__":
    inter = np.load("./datas/cap_PPG_150s_initial.npy")
    label = np.load("./datas/cap_label_for_PPG_150s_initial.npy")
    t1, t2 = get_data(inter, label, 4)
    print (t1.shape)
    print (t2.shape)


