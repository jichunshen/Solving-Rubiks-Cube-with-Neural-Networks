import pycuber as pc
from random import randint
import numpy as np
import keras
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.models import load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import train_test_split, StratifiedKFold
import itertools

np.random.seed(1337)
max_moves = 10
TRAIN_NUM = 10000
TEST_NUM = 10000

mycube = pc.Cube()
faces = ['L','U','R','D','F','B']
colors = ['[r]','[y]','[o]','[w]','[g]','[b]']
possible_moves = ["R","R'","R2","U","U'","U2","F","F'","F2","D","D'","D2","B","B'","B2","L","L'","L2"]

def sol2cat(solution):
    # transform solution to one hot vector encoding
    global possible_moves
    sol_tmp = []
    for j in range(len(solution)):
        sol_tmp.append(possible_moves.index(solution[j]))
    sol_cat = to_categorical(sol_tmp)
    return sol_cat

def cube2np(mycube):
    # transform cube object to np array with one hot vector represents color
    global faces
    global colors
    cube_np = np.zeros((6,3,3))
    for i,face in enumerate(faces):
        face_tmp = mycube.get_face(face)
        for j in range(3):
            for k in range(len(face_tmp[j])):
                caca = face_tmp[j][k]
                cube_np[i,j,k] = colors.index(str(caca))
    cube_np = to_categorical(cube_np.reshape(-1)).reshape((6,3,3,-1))
    return cube_np

def generate_game(max_moves = max_moves):
    # generate a single game with max number of permutations number_moves
    mycube = pc.Cube()
    global possible_moves
    formula = []
    cube_original = cube2np(mycube)
    number_moves = max_moves#randint(3,max_moves)
    for j in range(number_moves):
        formula.append(possible_moves[randint(0,len(possible_moves)-1)])
    my_formula = pc.Formula(formula)
    mycube = mycube((my_formula))
    cube_scrambled = mycube.copy()
    solution = my_formula.reverse()
    return cube_scrambled, solution

def generate_action_space(number_games=100):
    D = [] # action space
    global max_moves
    for i in range(number_games):
        scrambled_cube, solutions = generate_game(max_moves=max_moves)
        state = scrambled_cube
        for j in range(len(solutions)):
            action = solutions[j]
            current_state = state.copy()
            state_next = state(action)
            state_next = state_next.copy()
            D.append([current_state,action,state_next])
            state = state_next.copy()
    return D

def generate_data(N=32):
    x = []
    y = []
    D = generate_action_space(N)
    for d in D:
        x.append(cube2np(d[0]))
        y.append(to_categorical(possible_moves.index((str(d[1]))),len(possible_moves)))
    x = np.asarray(x)
    x = x.reshape(x.shape[0],6, 3, 3, 6)
    x = x.astype('float32')

    y = np.asarray(y)
    y = y.reshape(y.shape[0],y.shape[1])
    yield (x,y)


def cube2np_augment(mycube):
    # transform cube object to np array with one hot vector represents color
    global faces
    global colors
    perm_list = list(itertools.permutations(colors))
    for perm in perm_list:
        cube_np = np.zeros((6, 3, 3))
        for i, face in enumerate(faces):
            face_tmp = mycube.get_face(face)
            for j in range(3):
                for k in range(len(face_tmp[j])):
                    caca = face_tmp[j][k]
                    cube_np[i, j, k] = perm.index(str(caca))
        cube_np = to_categorical(cube_np.reshape(-1)).reshape((6, 3, 3, -1))
        yield cube_np

def generate_augment_data(N=32):
    while True:
        x = []
        y = []
        D = generate_action_space(N)
        for d in D:
            for item in cube2np_augment(d[0]):
                x.append(item)
                y.append(to_categorical(possible_moves.index((str(d[1]))), len(possible_moves)))

        x = np.asarray(x)
        x = x.reshape(x.shape[0], 18, 3, -1)
        x = x.astype('float32')

        y = np.asarray(y)

        yield (x, y)

def generate_data18(N=32):
    while True:
        x = []
        y = []
        D = generate_action_space(N)
        for d in D:
            x.append(cube2np(d[0]))
            y.append(to_categorical(possible_moves.index((str(d[1]))), len(possible_moves)))
        x = np.asarray(x)
        x = x.reshape(x.shape[0], 18, 3, 6)
        x = x.astype('float32')
        y = np.asarray(y)
        yield (x, y)

def generate_store_data(train_num=TRAIN_NUM, test_num=TEST_NUM):
    X_train, y_train = next(generate_data(train_num))
    X_test, y_test = next(generate_data(test_num))
    np.savez("cube_dataset.npz", X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)

#generate_store_data()

# def generate_store_data18(train_num=TRAIN_NUM, test_num=TEST_NUM):
#     X_train, y_train = next(generate_data18(train_num))
#     X_test, y_test = next(generate_data18(test_num))
#     np.savez("ds_10000_10000_10_18.npz", X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
#
# generate_store_data18()
#
def generate_store_augdata18(train_num=TRAIN_NUM, test_num=TEST_NUM):
    X_train, y_train = next(generate_augment_data(train_num))
    X_test, y_test = next(generate_augment_data(test_num))
    np.savez("ds_10000x720_10000x720_10_18aug.npz", X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)

generate_store_augdata18()

# NB_EPOCH = 10
# BATCH_SIZE = 10
# VALIDATION_SPLIT = 0.2
# VERBOSE = 1
#
# num_classes = len(possible_moves)
# num_epochs = 150
# input_shape = (18, 3, 1)
#
# model = Sequential()
# model.add(Conv2D(256, kernel_size=(3, 3),
#                  activation='relu',
#                  input_shape=input_shape))
# # model.add(Conv2D(128, kernel_size=(3, 3),
# #                  activation='relu',
# #                  input_shape=input_shape))
# #model.add(Conv2D(64, (3, 3), activation='relu'))
# model.add(Flatten())
# model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(num_classes, activation='softmax'))
# model.summary()
#
# # tbCallBack = keras.callbacks.TensorBoard(log_dir='/Users/SpringCurry/rubiks_cube_convnet/log', histogram_freq=0, write_graph=True, write_images=True)
#
#
# model.compile(loss=keras.losses.categorical_crossentropy,
#               optimizer=keras.optimizers.Adadelta(),
#               metrics=['accuracy'])
#
#
# model.fit(x=X_train, y=y_train, batch_size=BATCH_SIZE, epochs=NB_EPOCH, verbose=VERBOSE, callbacks=None, validation_split=VALIDATION_SPLIT, shuffle=True, steps_per_epoch=10, validation_steps=10)
# # score = model.evaluate(X_test, Y_test, verbose=1)
# # # for j in range(num_epochs):
# # #
# # #     if (j%10 == 0):
# # #         print ('epoch #',j)
# # #     model.fit_generator(generator= generate_data(10),steps_per_epoch=50,
# # #                                   epochs=1,verbose=2,validation_data=None,max_queue_size=1,use_multiprocessing=True,workers=6,initial_epoch =0)#generate_data(8)
# # #     model.evaluate_generator(generator= generate_data(2), steps=1)
# # # model.save('rubiks_model_wtvr.h5')  # creates a HDF5 file 'my_model.h5'
