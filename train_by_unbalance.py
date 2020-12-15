import numpy as np
from keras.layers import Conv1D  # Convolution Operation
from keras.layers import MaxPooling1D, AveragePooling1D
from keras.layers import Flatten, BatchNormalization
from keras.layers import Dense, Dropout
from keras.layers import Input, concatenate
from keras.optimizers import Adam, SGD
from keras import Model
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import time
from sklearn import metrics

import preprocess

def analysis(actual, predict, classes):
    real = np.zeros(actual.shape[0], dtype=int)
    pred = np.zeros(predict.shape[0], dtype=int)
    
    for i in range(predict.shape[0]):
        real[i] = np.argmax(actual[i])
        pred[i] = np.argmax(predict[i])

    param = {'matrix': metrics.confusion_matrix(real, pred),
            'kappa': metrics.cohen_kappa_score(real, pred),
            'accuracy': metrics.accuracy_score(real, pred),
            }
    precision = []
    recall = []
    for i in range(classes):
        precision.append(param['matrix'][i,i] / sum(param['matrix'][:, i]))
        recall.append(param['matrix'][i,i] / sum(param['matrix'][i, :]))
    param['precision'] = np.array(precision)
    param['recall'] = np.array(recall)

    return param

def inter_model(input1, pool_type):
    def cell(filter_num, inputs, pool):
        xx = BatchNormalization()(inputs)
        x1 = Conv1D(filter_num, kernel_size = 1, activation = "relu")(xx)
        
        x2 = Conv1D(filter_num*4, kernel_size = 5, padding = "same", activation = "relu", kernel_initializer='normal')(xx)
        x2 = Conv1D(filter_num, kernel_size = 1, padding = "same", activation = "relu", kernel_initializer='normal')(x2)

        x3 = Conv1D(filter_num*4, kernel_size = 2, padding = "same", activation = "relu", kernel_initializer='normal')(xx)
        x3 = Conv1D(filter_num, kernel_size = 1, padding = "same", activation = "relu", kernel_initializer='normal')(x3)

        x4 = Conv1D(filter_num*4, kernel_size = 3, padding = "same", activation = "relu", kernel_initializer='normal')(xx)
        x4 = Conv1D(filter_num, kernel_size = 1, activation = "relu", kernel_initializer='normal')(x4)
        
        merge = concatenate([x1, x2, x3, x4], axis=2)
        if pool == 1:
            pooling = MaxPooling1D(pool_size=2)(merge)
        else:
            pooling = AveragePooling1D(pool_size=2)(merge)
        return pooling

    x = input1
    cell1 = cell(16, x, pool_type)
    cell1 = cell(16, cell1, pool_type)
    cell1 = cell(16, cell1, pool_type)
    cell1 = Dropout(0.5)(cell1)
    
    return cell1

def get_model(input_data, label, classes):
    input1 = Input(shape=(len(input_data[0]), 1), name = "input_1")
    M = inter_model(input1, 2)
    M2 = inter_model(input1, 1)
    MM = concatenate([M, M2], axis=2)
    flat = Flatten()(M2)
    D = Dense(units=64, activation="relu", name="dense_out")(flat)
    D_out = Dense(units=classes, activation="softmax", name="output_layer")(D)

    model = Model(inputs= input1, outputs= D_out)
    model.compile(optimizer = Adam(lr = 0.001), loss = 'categorical_crossentropy', metrics= ['categorical_accuracy'])
    model.summary()
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=2)
    model.fit(input_data, label, epochs=200, validation_split=0.3, callbacks=[early_stopping], batch_size=64)
    return model

def result(train, label, classes, model_name = None):
    x_train1, x_test1, y_train1, y_test1 = train_test_split(train, label, test_size = 0.2, shuffle = True)
    model = get_model(x_train1, y_train1, classes)

    if model_name != None:
        model.save("./models"+model_name+".h5")
    
    pred = model.predict(x_test1)
    results = analysis(y_test1, pred, classes)
    return results

if __name__ == "__main__":
    inter = np.load("./datas/cap_PPG_150s_initial.npy")
    label = np.load("./datas/cap_label_for_PPG_150s_initial.npy")
    classes = 5
    train, stage = preprocess.get_data(inter, label, classes)
    results = result(train, stage, classes)
    print(results)




