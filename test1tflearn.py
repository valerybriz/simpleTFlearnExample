# -*- coding: utf-8 -*-
import numpy as np
import tflearn
import random


def crear_set_de_features():

    # patrones conocidos (5 features)
    features = []
    features.append([[0, 0, 0, 0, 0], [0,1]])
    features.append([[0, 0, 0, 0, 1], [0,1]])
    features.append([[0, 0, 0, 1, 1], [0,1]])
    features.append([[0, 0, 1, 1, 1], [0,1]])
    features.append([[0, 1, 1, 1, 1], [0,1]])
    features.append([[1, 1, 1, 1, 0], [0,1]])
    features.append([[1, 1, 1, 0, 0], [0,1]])
    features.append([[1, 1, 0, 0, 0], [0,1]])
    features.append([[1, 0, 0, 0, 0], [0,1]])
    features.append([[1, 0, 0, 1, 0], [0,1]])
    features.append([[1, 0, 1, 1, 0], [0,1]])
    features.append([[1, 1, 0, 1, 0], [0,1]])
    features.append([[0, 1, 0, 1, 1], [0,1]])
    #features.append([[0, 0, 1, 0, 1], [0,1]])
    features.append([[1, 0, 1, 1, 1], [1,0]])
    features.append([[1, 1, 0, 1, 1], [1,0]])
    features.append([[1, 0, 1, 0, 1], [1,0]])
    features.append([[1, 0, 0, 0, 1], [1,0]])
    features.append([[1, 1, 0, 0, 1], [1,0]])
    features.append([[1, 1, 1, 0, 1], [1,0]])
    features.append([[1, 1, 1, 1, 1], [1,0]])
    features.append([[1, 0, 0, 1, 1], [1,0]])

    # mezclamos nuestras features y las convertimos en un np.array
    random.shuffle(features)
    features = np.array(features)

    # creamos las listas de prueba
    train_x = list(features[:,0])
    train_y = list(features[:,1])

    return train_x, train_y

if __name__ == "__main__":

    train_x, train_y = crear_set_de_features()

    # La red neural
    net = tflearn.input_data(shape=[None, 5])
    #dos capas ocultas
    net = tflearn.fully_connected(net, 32)
    net = tflearn.fully_connected(net, 32)
    #la prediccion tendra 2 clases
    net = tflearn.fully_connected(net, 2, activation='softmax')
    #regresion
    net = tflearn.regression(net)

    # definimos el modelo e instanciamos el tensorboard para los logs
    model = tflearn.DNN(net, tensorboard_dir='tflearn_logs')
    # Empezamos el entrenamiento
    model.fit(train_x, train_y, n_epoch=800, batch_size=16, show_metric=True)

    print(model.predict([[0, 0, 1, 0, 1]]))
