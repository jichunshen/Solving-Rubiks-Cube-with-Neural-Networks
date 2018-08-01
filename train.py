import pycuber as pc
from random import randint
from time import time
import numpy as np
import keras
import os
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, Conv3D
from keras import backend as K
from keras.models import load_model
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from sklearn.model_selection import train_test_split, StratifiedKFold
import train_cube

def load_data(path='cube_dataset.npz'):
    f = np.load(path)
    x_train, y_train, x_test, y_test = f['X_train'], f['y_train'], f['X_test'], f['y_test']
    f.close()
    return (x_train, y_train, x_test, y_test)

(X_train, y_train, X_test, y_test) = load_data('ds_10000_10000_10_18aug.npz')

# def data2_18(X_train, X_test):
#     X_train = X_train.reshape(-1, 18, 3, 6)
#     X_test = X_test.reshape(-1, 18, 3, 6)
#     return (X_train, X_test)
#
# X_train18, X_test18 = data2_18(X_train, X_test)

print(X_train.shape)

NB_EPOCH = 8000
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.2
VERBOSE = 1
INPUT_SHAPE = (18, 3, 6)
NUM_CLASSES = 18
MODEL_DIR = "./tmp_conv3D_6000_32"

def batch_generator(X_train = X_train, y_train = y_train, batch_size = BATCH_SIZE):
    while True:
        idx = np.random.randint(0, X_train.shape[0], batch_size)
        X = X_train[idx]
        y = y_train[idx]
        yield (X,y)

# model = Sequential()
# model.add(Conv3D(256, kernel_size=(2,2,2),
#                  activation='relu',
#                  input_shape=INPUT_SHAPE))
# model.add(Conv3D(512, kernel_size=(2,2,1),
#                  activation='relu'))
# model.add(Dropout(0.5))
# model.add(Flatten())
# model.add(Dense(512, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(NUM_CLASSES, activation='softmax'))
# model.summary()
#
# model.compile(loss=keras.losses.categorical_crossentropy,
#               optimizer=keras.optimizers.Adadelta(),
#               metrics=['accuracy'])
# earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto')
# tensorboard = TensorBoard(log_dir="logs_conv3D_6000_32/{}".format(time()))
# checkpoint = ModelCheckpoint(filepath=os.path.join(MODEL_DIR, "model-{epoch:02d}.h5"), save_best_only=True)
# # model.fit(x=X_train, y=y_train, batch_size=BATCH_SIZE, epochs=NB_EPOCH, verbose=VERBOSE, callbacks=[tensorboard, checkpoint, earlystop], validation_split=VALIDATION_SPLIT, shuffle=True, steps_per_epoch=10)
# # score = model.evaluate(X_test, y_test, batch_size=BATCH_SIZE, verbose=VERBOSE)
#
#
# model.fit_generator(generator=batch_generator(),
#                     steps_per_epoch=1,
#                     epochs=NB_EPOCH,
#                     verbose=VERBOSE,
#                     validation_data=batch_generator(X_test, y_test),
#                     validation_steps=1,
#                     callbacks=[tensorboard, checkpoint])

# model = load_model('rubiks_model.h5')
# score = model.evaluate_generator(train_cube.generate_data(5000), steps=5, max_queue_size=10, workers=1, use_multiprocessing=False)
# # score = model.evaluate_generator(batch_generator(X_test, y_test, 5000), steps=5, max_queue_size=10, workers=1, use_multiprocessing=False)
#
# print("Test score:", score[0])
# print("Test accuracy:", score[1])

# model.save('tmp_conv3D_6000_32.h5')
# # serialize model to JSON
# model_json = model.to_json()
# with open("model.json", "w") as json_file:
#     json_file.write(model_json)
# # serialize weights to HDF5
# model.save_weights("model.h5")
# print("Saved model to disk")



# # for j in range(num_epochs):
# #
# #     if (j%10 == 0):
# #         print ('epoch #',j)
# #     model.fit_generator(generator= generate_data(10),steps_per_epoch=50,
# #                                   epochs=1,verbose=2,validation_data=None,max_queue_size=1,use_multiprocessing=True,workers=6,initial_epoch =0)#generate_data(8)
# #     model.evaluate_generator(generator= generate_data(2), steps=1)
# # model.save('rubiks_model_wtvr.h5')  # creates a HDF5 file 'my_model.h5'
