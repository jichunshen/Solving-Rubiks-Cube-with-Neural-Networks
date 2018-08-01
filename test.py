from keras.utils import to_categorical
from random import randint
import pycuber as pc
import numpy as np
import itertools

possible_moves = ["R","R'","R2","U","U'","U2","F","F'","F2","D","D'","D2","B","B'","B2","L","L'","L2"]

mycube = pc.Cube()
faces = ['L','U','R','D','F','B']
colors = ['[r]','[y]','[o]','[w]','[g]','[b]']

def sol2cat(solution):
    # transform solution to one hot vector encoding
    global possible_moves
    sol_tmp = []
    for j in range(len(solution)):
        sol_tmp.append(possible_moves.index(solution[j]))
    sol_cat = to_categorical(sol_tmp)
    return sol_cat

perm_list = list(itertools.permutations(colors))
print(perm_list)

# def cube2np(mycube):
#     # transform cube object to np array
#     # works around the weird data type used
#     global faces
#     global colors
#     cube_np = np.zeros((6,3,3,6))
#     for i,face in enumerate(faces):
#         face_tmp = mycube.get_face(face)
#         for j in range(3):
#             for k in range(len(face_tmp[j])):
#                 caca = face_tmp[j][k]
#                 cube_np[i,j,k] = to_categorical(colors.index(str(caca)), 6)
#     return cube_np

# print(cube2np(mycube).shape)

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
    # cube_np = to_categorical(cube_np.reshape(-1)).reshape((6,3,3,-1))
    return cube_np

max_moves = 2

def generate_game(max_moves=max_moves):
    # generate a single game with max number of permutations number_moves

    mycube = pc.Cube()

    global possible_moves
    formula = []
    cube_original = cube2np(mycube)
    number_moves = max_moves  # randint(3,max_moves)
    for j in range(number_moves):
        formula.append(possible_moves[randint(0, len(possible_moves) - 1)])

    # my_formula = pc.Formula("R U R' U' D' R' F R2 U' D D R' U' R U R' D' F'")

    my_formula = pc.Formula(formula)

    # print(my_formula)

    mycube = mycube((my_formula))
    # use this instead if you want it in OG data type

    cube_scrambled = mycube.copy()

    solution = my_formula.reverse()

    # print(mycube)

    return cube_scrambled, solution

# cube_scrambled, solution = generate_game(1)
# cube_scrambleds = []
# cube_scrambleds.append(cube_scrambled)
# print(cube_scrambleds)
# print(solution)

def generate_N_games(N=10, max_moves=max_moves):
    scrambled_cubes = []
    solutions = []
    for j in range(N):
        cube_scrambled, solution = generate_game(max_moves=max_moves)
        scrambled_cubes.append(cube_scrambled)
        solutions.append(solution)

    return scrambled_cubes, solutions

# scrambled_cubes, solutions = generate_N_games(3,2)
# print(scrambled_cubes)
# print(solutions)

def generate_action_space(number_games=100):
    D = []  # action space
    states_hist = []
    game_count = 0
    play_game = True
    global max_moves
    while play_game:
        scrambled_cube, solutions = generate_game(max_moves=max_moves)
        state = scrambled_cube.copy()
        for j in range(len(solutions)):
            states_hist.append(state.copy())
            action = solutions[j]
            current_state = state.copy()
            state_next = state(action)

            state_next = state_next.copy()

            reward = j + 1

            D.append([current_state, action, reward, state_next])

            state = state_next.copy()

        states_hist.append(state.copy())

        game_count += 1

        if game_count >= number_games:
            break

    return D

# print(generate_action_space(3))

def generate_data(N=3):
    while True:
        x = []
        y = []

        D = generate_action_space(N)
        # print(D)
        for d in D:
            x.append(cube2np(d[0]))
            y.append(to_categorical(possible_moves.index((str(d[1]))), len(possible_moves)))

        x = np.asarray(x)
        x = x.reshape(x.shape[0], 18, 3, 1)
        x = x.astype('float32')

        y = np.asarray(y)
        y = y.reshape(y.shape[0], y.shape[1])
        return x,y

# for x,y in generate_data(1):
#     print(x)
#     print(y)
