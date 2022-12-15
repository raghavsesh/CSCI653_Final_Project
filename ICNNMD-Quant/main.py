import argparse
from http.client import REQUEST_URI_TOO_LONG
from re import X
import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
import functools
import tensorflow  as tf
import os, pickle, time, tensorflow_quantization
from tensorflow_quantization import quantize_model
from tensorflow_quantization import utils

parser = argparse.ArgumentParser(description='ICNNMD')
# The data file to read
parser.add_argument('--nc1_file')
# The location the generated 'bin' file to save
parser.add_argument('--nc2_file')
parser.add_argument('--pdb1_file')
parser.add_argument('--pdb2_file')
parser.add_argument('--print_acc')
parser.add_argument('--save_models')
parser.add_argument('--print_detail')
args = parser.parse_args()

def save_as_onnx(model, intermed, final):
    os.system('mkdir -p %s' %(intermed))
    tf.keras.models.save_model(model, intermed)
    utils.convert_saved_model_to_onnx(saved_model_dir = intermed, onnx_model_path = final)
    return

def create_model(x_train, num_classes):
    inputs = tf.keras.Input(shape=x_train[0].shape)
    x = tf.keras.layers.Conv2D(32, (3,3), padding='same')(inputs)
    x = tf.keras.layers.Conv2D(32, (3,3), activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2,2))(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    x = tf.keras.layers.Conv2D(64, (3,3), padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv2D(64, (3,3), padding='same', activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2,2))(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    model = tf.keras.Model(inputs, outputs)
    opt = keras.optimizer.RMSprop(lr=0.0001, decay=1e-6)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    return model 


def cnn_build(X_train, k, num_classes):

    model = []
    for i in range(k):
        modelt = create_model(X_train[i], num_classes)
        model.append(modelt)
        if( i == 0 ):
            modelt.summary()
    print('model ready')
    return model
    # =============================================================================

    
    
def cnn_train(model, X_train, y_train, X_test, y_test, k, batch_size, epochs, p):
    # =============================================================================
    history = []
    for i in range(k):
        historyt = model[i].fit(X_train[i], y_train[i],
                  batch_size=batch_size,
                  epochs=epochs,
                  validation_data=(X_test[i], y_test[i]),
                  shuffle=True, verbose=p)
        history.append(historyt)
    print('CNN Training done for %d models'%(k))
    return model,history
    # =============================================================================



def count_scores(model, X):

    k = len(model)

    # =============================================================================
    scores = []
    for i in range(k):
        if i==0:
            scores = [x[1] for x in model[i].predict_proba(X)]
        elif i!=4:
            score = [x[1] for x in model[i].predict_proba(X)]
            for j in range(len(scores)):
                scores[j] = scores[j] + score[j]
        else:
            score = [x[1] for x in model[i].predict_proba(X)]
            for j in range(len(scores)):
                scores[j] = (scores[j] + score[j])/k
                
    return scores
    # =============================================================================
    

def pre_results(model, X):

    scores = count_scores(model, X)
    
    results = scores
    
    for i in range(len(results)):
        if results[i]>0.5:
            results[i]=1
        else:
            results[i]=0
    
    return results
    
    
def eva_acc(model, X, y):

    # =============================================================================
    # compute on remaining test data
    
    results = pre_results(model, X)
#     pipe_pred_test = model[0].predict_classes(X)
    
    n = 0
    for i in range(len(y)):
        if results[i]==y[i]:
            n+=1
        
    acc = n/len(y)
    return acc
    # =============================================================================

    
def eva_cross_acc(model,X,y):
    k = len(model)
    
    for i in range(k):
        y_pre  = np.argmax(model[i].predict(X), axis=-1)
        acc = com_accuracy(y_pre, y)
        print("The acc of model ",i,":",acc)
        
def eva_cross_acc_all(model,X_train,X_test,y_train,y_test):
    k = len(model)
    
    for i in range(k):
        y_train_pre  = np.argmax(model[i].predict(X_train[i]), axis=-1)
        acc_train = com_accuracy_cross(y_train_pre, y_train[i])
        y_test_pre  = np.argmax(model[i].predict(X_test[i]), axis=-1)
        acc_test = com_accuracy_cross(y_test_pre, y_test[i])
        
        print("The acc of model ",i,": train: ",acc_train," ,test: ",acc_test)

def com_accuracy_cross(y1,y2):
    n = 0
    length_y = len(y1)
    
    for i in range(length_y):
        if y1[i]==y2[i][1]:
            n += 1
            
    accuracy = n/length_y
    return accuracy

def eva_table(model, X, y):

    # =============================================================================
    # compute on remaining test data
    
    results = pre_results(model, X)
#     pipe_pred_test = model[0].predict_classes(X)
    from sklearn.metrics import classification_report
    print(classification_report(y_true=y, y_pred = results))
    # =============================================================================


def eva_roc(model, X, y):

    # =============================================================================
    from sklearn.metrics import roc_curve, auc
    from sklearn import metrics

    scores = count_scores(model, X)

    fpr, tpr, thresholds = metrics.roc_curve(y, scores, pos_label=1)

    auc = metrics.auc(fpr, tpr)

    import matplotlib.pyplot as plt
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()
    # =============================================================================


def eva_pr(model, X, y):

    # =============================================================================
    from sklearn.metrics import precision_recall_curve
    import matplotlib.pyplot as plt

    scores = count_scores(model, X)
    
    precision, recall, _ =precision_recall_curve(y, scores)
    plt.plot(recall, precision,color='navy')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.plot(recall, precision)
    plt.title("Precision-Recall")

    plt.show()


    # =============================================================================
    
    
def com_accuracy(y1, y2):

    n = 0
    length_y = len(y1)
    
    for i in range(length_y):
        if y1[i]==y2[i]:
            n += 1
            
    accuracy = n/length_y
    return accuracy


if __name__ == '__main__':
    file0_1 = args.nc1_file
    file0_2 = args.pdb1_file
    file1_1 = args.nc2_file
    file1_2 = args.pdb2_file
    if_prt = args.print_acc
    if_save = args.save_models
    if_dtl = args.print_detail
    try:
        from . import traj_utils
    except Exception:
        import traj_utils
    _, y_all, X_all = traj_utils.traj_pre(file0_1, file1_1, file1_2, file1_2)
    print("Preprocess Done.\n")
    try:
        from . import split_data
    except Exception:
        import split_data
    group = 20
    cap = int(len(X_all) / 20)
    k = 2
    X_train, y_train, X_test, y_test = split_data.split_by_group(X_all, y_all, group, cap, k)


    # =============================================================================
    try:
        from . import data_utils
    except Exception:
        import data_utils
    num_classes = 2
    X_train, y_train, X_test, y_test = data_utils.data_encode(X_train, y_train, X_test, y_test, k, num_classes)

    batch_size = 128
    epochs = 5
    model = cnn_build(X_train, k, num_classes)
    model, history = cnn_train(model, X_train, y_train, X_test, y_test, k, batch_size, epochs, if_dtl)

    if (if_save):
        for t in range(k):
            model[t].save('model' + str(t) + '.h5')

    qaware_models = []

    for t in range(k):
        save_as_onnx(model[t], './output/keras_reg_model_%d'%(t), './output/onnx_reg_model_%d'%(t))

    for t in range(k):
        quant_input = tf.keras.models.clone_model(model[t])
        quant_input.compile(optimizer = 'rmsprop', loss='categorical_crossentropy')
        quant_input.set_weights(model[t].get_weights())
        opt = keras.optimizers.RMSprop(lr=0.0001,decay=1e-6)
        qaware_models.append(quantize_model(quant_input))
        qaware_models[t].compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
        
    qaware_models[0].summary()

    print('Quantizing models...')

    model, history = cnn_train(qaware_models, X_train, y_train, X_test, y_test, k, batch_size, epochs, if_dtl)

    if (if_prt):
        eva_cross_acc_all(model, X_train, X_test, y_train, y_test)
        eva_cross_acc_all(qaware_models, X_train, X_test, y_train, y_test)
    
    for t in range(k):
        save_as_onnx(model[t], './output/keras_qdq_model_%d'%(t), './output/onnx_qdq_model_%d'%(t))