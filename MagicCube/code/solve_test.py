import pycuber as pc
from keras.utils import to_categorical
import numpy as np
from random import randint
from keras.models import load_model
import matplotlib.pyplot as plt

TEST_NUM = 1000

possible_moves = ["R", "R'", "R2", "U", "U'", "U2", "F", "F'", "F2", "D", "D'", "D2", "B", "B'", "B2", "L", "L'", "L2"]
faces = ['L', 'U', 'R', 'D', 'F', 'B']  # for pycuber
colors = ['[r]', '[y]', '[o]', '[w]', '[g]', '[b]']  # for pycuber

def cube2np(mycube):
    # transform cube object to np array
    # works around the weird data type used
    global faces
    global colors
    cube_np = np.zeros((6,3,3))
    for i,face in enumerate(faces):
        face_tmp = mycube.get_face(face)
        for j in range(3):
            for k in range(len(face_tmp[j])):
                caca = face_tmp[j][k]
                cube_np[i,j,k] = colors.index(str(caca))
    return cube_np

model_conv3d = load_model("/Users/SpringCurry/rubiks_cube_convnet/model/conv3D_6000_32.h5")
model_conv2d = load_model("/Users/SpringCurry/rubiks_cube_convnet/model/1Conv3dense_12000_32_18.h5")
model_conv2dn = load_model("/Users/SpringCurry/rubiks_cube_convnet/model/1Conv3dense_no_onehot_12000_32_18.h5")

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

def _solve_cube_NN(max_move, model):
    cube_solved = pc.Cube()
    cube_shuffled = generate_game(max_move)
    for j in range(10):
        cube_np = cube2np(cube_shuffled)
        cube_np = np.reshape(cube_np, (1, 6, 3, 3, 6))
        move = possible_moves[np.argmax(model.predict(cube_np))]
        cube_shuffled(move)
        if cube_shuffled == cube_solved:
            return 1
    return 0

def generate_game(max_moves = 10):
    # generate a single game with max number of permutations number_moves
    mycube = pc.Cube()
    global possible_moves
    formula = []
    number_moves = max_moves#randint(3,max_moves)
    for j in range(number_moves):
        formula.append(possible_moves[randint(0,len(possible_moves)-1)])
    my_formula = pc.Formula(formula)
    mycube = mycube((my_formula))
    cube_scrambled = mycube.copy()
    return cube_scrambled

def solve_test(max_move, model, test_num = TEST_NUM):
    count = 0
    for i in range(test_num):
        count+=_solve_cube_NN(max_move, model)
    return count/test_num

def manytest(max_move, acc, model, test_num = TEST_NUM):
    for i in range(10):
        acc.append(solve_test(i+1, model))

conv3d_acc = []
conv2d_acc = []
conv2dn_acc = []
move = list(range(1,11))

manytest(10, conv3d_acc, model_conv3d)
manytest(10, conv2d_acc, model_conv2d)
manytest(10, conv2dn_acc, model_conv2dn)

print(conv3d_acc)
print(conv2d_acc)
print(conv2dn_acc)

plt.title('Probability of Solving Cube in 10 Moves')
plt.plot(move, conv3d_acc, color='green', label='2Conv3d 2Dense')
plt.plot(move, conv2d_acc, color='red', label='1Conv2d 3Dense')
plt.plot(move, conv2dn_acc,  color='skyblue', label='1Conv2d 4Dense no one-hot')
plt.legend()

plt.xlabel('shuffle moves')
plt.ylabel('probability of solving')
plt.savefig("./result.jpg")
plt.show()

