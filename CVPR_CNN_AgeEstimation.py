# -*- coding: utf-8 -*-

import numpy as np
import csv
from PIL import Image
#from keras.datasets import mnist
#from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Input
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.models import Model
from keras.optimizers import SGD
from keras import backend as K
from keras.models import load_model
from keras.callbacks import ModelCheckpoint
from keras.utils.visualize_util import plot

#from keras.utils.visualize_util import plot

#from keras.layers.normalization2 import LRN2D

#画像読み込み
age_interval = [0,5,10,15,20,25,30,35,40,45,50,55,60]
age_count_train = [0,0,0,0,0,0,0,0,0,0,0,0,0]
age_count_test = [0,0,0,0,0,0,0,0,0,0,0,0,0]

X_train = []
Y_train = []

Y_train_1 = []
Y_train_2 = []
Y_train_3 = []
Y_train_4 = []
Y_train_5 = []
Y_train_6 = []
Y_train_7 = []
Y_train_8 = []
Y_train_9 = []
Y_train_10 = []
Y_train_11 = []
Y_train_12 = []
Y_train_13 = []

Y_train_list = []

Y_train_list.append(Y_train_1)
Y_train_list.append(Y_train_2)
Y_train_list.append(Y_train_3)
Y_train_list.append(Y_train_4)
Y_train_list.append(Y_train_5)
Y_train_list.append(Y_train_6)
Y_train_list.append(Y_train_7)
Y_train_list.append(Y_train_8)
Y_train_list.append(Y_train_9)
Y_train_list.append(Y_train_10)
Y_train_list.append(Y_train_11)
Y_train_list.append(Y_train_12)
Y_train_list.append(Y_train_13)

Y_test_1 = []
Y_test_2 = []
Y_test_3 = []
Y_test_4 = []
Y_test_5 = []
Y_test_6 = []
Y_test_7 = []
Y_test_8 = []
Y_test_9 = []
Y_test_10 = []
Y_test_11 = []
Y_test_12 = []
Y_test_13 = []

Y_test_list = []

Y_test_list.append(Y_test_1)
Y_test_list.append(Y_test_2)
Y_test_list.append(Y_test_3)
Y_test_list.append(Y_test_4)
Y_test_list.append(Y_test_5)
Y_test_list.append(Y_test_6)
Y_test_list.append(Y_test_7)
Y_test_list.append(Y_test_8)
Y_test_list.append(Y_test_9)
Y_test_list.append(Y_test_10)
Y_test_list.append(Y_test_11)
Y_test_list.append(Y_test_12)
Y_test_list.append(Y_test_13)

X_test = []
Y_test = []

y_ = []

sample_name = []
testdata_name = []

f = csv.reader(open("csv", "rb"), delimiter=",", quotechar="'")

for row in f:
    for age, num in zip(age_interval, range(13)):
        print "TrainData",
        print row[1],
        print age,
        #print age_count[num]
        if row[1] == str(age) and age_count_train[num] < 1000:
            img_name = "croppedImg/" + str(row[0])
            try:
                img = np.array( Image.open(img_name) ).transpose(2,0,1)
            except:
                continue
            X_train.append(img)
            for i in range(13):
                if i >= num:
                    y_.append(1)
                    Y_train_list[i].append(1)
                else:
                    y_.append(0)
                    Y_train_list[i].append(0)
            #print num
            #print y_
            Y_train.append(y_)
            y_ = []
            age_count_train[num] += 1

        elif row[1] == str(age) and age_count_test[num] < 100:
            img_name = "croppedImg/" + str(row[0])
            sample_name.append(row[1])
            testdata_name.append(row[0])
            try:
                img = np.array( Image.open(img_name) ).transpose(2,0,1)
            except:
                continue
            X_test.append(img)
            for i in range(13):
                if i >= num:
                    y_.append(1)
		            Y_test_list[i].append(1)
                else:
                    y_.append(0)
		            Y_test_list[i].append(0)
            #print num
            #print y_
            Y_test.append(y_)
            y_ = []
            age_count_test[num] += 1

X_train = np.asarray(X_train)
print(X_train.shape)
#X_train = X_train.transpose(0, 3, 1, 2)
Y_train = np.asarray(Y_train)

for i in range(13):
    Y_train_list[i] = np.asarray(Y_train_list[i])

X_test = np.asarray(X_test)
print(X_test.shape)
#X_test = X_test.transpose(0, 3, 1, 2)
Y_test = np.asarray(Y_test)

for i in range(13):
    Y_test_list[i] = np.asarray(Y_test_list[i])

print Y_train_list[0]

input_image_size_rows = 60
input_image_size_cols = 60

nb_filters_1 = 20
nb_conv_1 = 5
nb_pool_1 = 2

nb_filters_2 = 40
nb_conv_2 = 7
nb_pool_2 = 2

nb_filters_3 = 80
nb_conv_3 = 11

main_input = Input(shape=(3, 60, 60))

#model = Convolution2D(nb_filters_1, nb_conv_1, nb_conv_1,
#                        border_mode="valid",
#                        input_shape=(3, input_image_size_rows, input_image_size_cols))(main_input)
y = Convolution2D(nb_filters_1, nb_conv_1, nb_conv_1, border_mode="valid")(main_input)

y = Activation("relu")(y)
#model.add(LRN2D())
y = MaxPooling2D(pool_size=(nb_pool_1, nb_pool_1), strides=(2,2))(y)

y = Convolution2D(nb_filters_2, nb_conv_2, nb_conv_2, border_mode="valid")(y)
y = Activation("relu")(y)
y = MaxPooling2D(pool_size=(nb_pool_2, nb_pool_2), strides=(2,2))(y)

y = Convolution2D(nb_filters_3, nb_conv_3, nb_conv_3, border_mode="valid")(y)
y = Activation("relu")(y)

y = Flatten()(y)
y = Dense(80)(y)
#model.add(Activation("softmax"))

#outputs = []
#for i in range(13):
#    outputs.append(Dense(2, activation="softmax", name=("output" + str(i)))(y))

output1 = Dense(1, activation="sigmoid")(y)
output2 = Dense(1, activation="sigmoid")(y)
output3 = Dense(1, activation="sigmoid")(y)
output4 = Dense(1, activation="sigmoid")(y)
output5 = Dense(1, activation="sigmoid")(y)
output6 = Dense(1, activation="sigmoid")(y)
output7 = Dense(1, activation="sigmoid")(y)
output8 = Dense(1, activation="sigmoid")(y)
output9 = Dense(1, activation="sigmoid")(y)
output10 = Dense(1, activation="sigmoid")(y)
output11 = Dense(1, activation="sigmoid")(y)
output12 = Dense(1, activation="sigmoid")(y)
output13 = Dense(1, activation="sigmoid")(y)

model = Model(input = [main_input], output = [output1,output2,output3,output4,output5,output6,output7,output8,output9,output10,output11,output12,output13])

sgd = SGD(lr=0.01, momentum=0.9, decay=1e-6, nesterov=True)
model.compile(loss = "binary_crossentropy",optimizer=sgd, metrics=['accuracy'])
#model.compile(loss = "binary_crossentropy", optimizer=sgd, metrics=['accuracy'])
#plot(model, to_file='model.png')

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

checkpointer = ModelCheckpoint(filepath="weights.{epoch:02d}-{val_loss:.2f}.hdf5", verbose=1, save_best_only=True)

model.fit(X_train, [Y_train_list[0],Y_train_list[1],
			Y_train_list[2],Y_train_list[3],
			Y_train_list[4],Y_train_list[5],
			Y_train_list[6],Y_train_list[7],
			Y_train_list[8],Y_train_list[9],
			Y_train_list[10],Y_train_list[11],Y_train_list[12]],
			verbose = 1, nb_epoch=50, batch_size=100,
			validation_data = (X_test, [Y_test_list[0], Y_test_list[1], Y_test_list[2],
						    Y_test_list[3], Y_test_list[4], Y_test_list[5],
						    Y_test_list[6], Y_test_list[7], Y_test_list[8],
						    Y_test_list[9], Y_test_list[10], Y_test_list[11], Y_test_list[12]]))
score = model.evaluate(X_test, [Y_test_list[0], Y_test_list[1], Y_test_list[2],
				                 Y_test_list[3], Y_test_list[4], Y_test_list[5],
				                 Y_test_list[6], Y_test_list[7], Y_test_list[8],
				                 Y_test_list[9], Y_test_list[10], Y_test_list[11], Y_test_list[12]],
				                 verbose=1, batch_size =100)
print("Test score:", score[0])
print("Test accuracy:", score[1])

model.save('model_epoch_50.h5')

score = model.predict(X_test, batch_size=len(X_test),verbose=1)
print 'len score:', len(score)

maeN = 0
age_count_0 = 0

for j,k in zip(range(len(X_test)), sample_name):
    print testdata_name[j]
    print "result:", int(k)/5,
    for i in range(len(score)):
        print int(score[i][j][0] + 0.5),
        if(int(score[i][j][0] + 0.5) == 0):
            age_count_0 += 1
    print " "
    maeN += abs(int(k) - (age_count_0*5))
    age_count_0 = 0

print "MAE:", float(maeN) / float(len(X_test))

plot(model, to_file="model.png")
