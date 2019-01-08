#!/usr/bin/env python
# 
# solver16.py : Solve the 16 puzzle problem - upto 3 tiles to move.
#
# (1)   
# State space: All possible formulation of the tiles in 16 puzzle problem.
# For example, a sample of the state look like this,
# S0 = array([[ 2,  3,  0,  4],
#             [ 1,  6,  7,  8],
#             [ 5, 10, 11, 12],
#             [ 9, 13, 14, 15]])
#
# Successor function: Possible position of the tiles after 1 move (either moving 1, 2, or 3 tiles at ones)
# I marked the each successor function with its appropriate move with the 3 character notation.
#
# The successor gets in the input of current state and the move up to the state.
# Then it returns list of all the possible next states paired with moves taken upto that state, heuristic, and the cost.
# 
# >>> successor(S0, [])
# [[array([[ 1,  2,  3,  4],
        # [ 5,  6,  7,  8],
        # [ 9, 10,  0, 12],
        # [13, 14, 11, 15]]), ['D14'], 1.3333333333333333, 2.333333333333333], 
# [array([[ 1,  2,  3,  4],
        # [ 5,  6,  0,  8],
        # [ 9, 10,  7, 12],
        # [13, 14, 11, 15]]), ['D24'], 2.0, 3.0], 
# [array([[ 1,  2,  0,  4],
        # [ 5,  6,  3,  8],
        # [ 9, 10,  7, 12],
        # [13, 14, 11, 15]]), ['D34'], 2.6666666666666665, 3.6666666666666665], 
# [array([[ 1,  2,  3,  4],
        # [ 5,  6,  7,  8],
        # [ 9, 10, 11, 12],
        # [13,  0, 14, 15]]), ['R13'], 1.3333333333333333, 2.333333333333333], 
# [array([[ 1,  2,  3,  4],
        # [ 5,  6,  7,  8],
        # [ 9, 10, 11, 12],
        # [ 0, 13, 14, 15]]), ['R23'], 2.0, 3.0], 
# [array([[ 1,  2,  3,  4],
        # [ 5,  6,  7,  8],
        # [ 9, 10, 11, 12],
        # [13, 14, 15,  0]]), ['L13'], 0.0, 1.0]]
# 
# Edge weights: 1 (One valid move is calculated as cost of 1)
#
# Goal state: Following is the goal state
#
# array([[ 1,  2,  3,  4],
#        [ 5,  6,  7,  8],
#        [ 9, 10, 11, 12],
#        [13, 14, 15,  0]])

# Heuristic function: (Sum of Manhattan cost) / 3
# 
# If I use the sum of the Manhattan cost as in the notes, it would be not admissble due to the over-estimating.
# I can move the tiles upto 3, which means that Dividing the sum of Manhattan cost by 3 won't over-estimate.
# Hence, this huristic function is admissible.
# Also, it is consistent, because it meets the triangle inequality.
#
# (2) How the search algorithm work
#
# For each step, the algorithm chooses to branch the node with the minimum f value, which is (heuristic + cost). The algorithm also keeps track of the revisited states. In the successor function, if the child state is previously visited, then it doesn't return the visited child state. It keeps branching until it reaches the goal state.
#
# (3) Any problem I faced, assumptions, simplifications, design decisions
#
# The heuristic function I am using is admissble, hence it would be complete and optimal. 
# However, when the input board gets very complicated, the power of the heuristics to find the goal state tend to get weaker.
# I found that instead of using the admissible heuristic, if I used a heuristic with sum of Manhattan distance without dividing it by 3, 
# the performance got much better. Here is the comparison of the two heuristics.
#
# < Heuristic: sum of Manhattan distance divided by 3 >
#
# [hankjang@silo problem3]$ time python solver16.py input_board10.txt
# D14 R24 U13 L22 D24 R14 D12 R23 U31 L31

# real    0m22.801s
# user    0m22.755s
# sys     0m0.030s
#
# < Heuristic: sum of Manhattan distance (not dividing by 3)>
#
# [hankjang@silo problem3]$ time python solver16.py input_board10.txt
# D14 R24 U13 L22 D24 R14 D12 R23 U31 L31

# real    0m0.587s
# user    0m0.558s
# sys     0m0.026s
# 
# The difference in performance was stable for over 10 different input boards I tested.
# However, since the heuristic of using sum of Manhattan distance (not dividing by 3) is not admissible,
# I decided to stick with the slower, but admissible heuristic.
#

from __future__ import division
import sys
import numpy as np
from scipy.spatial.distance import cdist

n_tile = range(4)
col_move = {0:["L11","L21","L31"],1:["R12","L12","L22"],2:["R13","R23","L13"],3:["R14","R24","R34"]}
row_move = {0:["U11","U21","U31"],1:["D12","U12","U22"],2:["D13","D23","U13"],3:["D14","D24","D34"]}

G = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,0]])

# Save the position to dictionary
# Each dictionary holds the row, col index as value in a numpy array
# Then, returns the sum of the mahattan distances per all tiles divided by 3.
def manhattan_distance(s1, s2):
    s1_dict = {}
    s2_dict = {}
    for i in n_tile:
        for j in n_tile:
            s1_dict[s1[i][j]] = np.array([i, j])
            s2_dict[s2[i][j]] = np.array([i, j])
    return sum([np.abs(s1_dict[key]-s2_dict[key]).sum() for key in s1_dict]) / 3

def initial_state(filename):
    file = open(filename, "r")
    return np.array([[int(tile) for tile in line.split()] for line in file])
    
def swap(s, r1, c1, r2, c2):
    temp = s[r1,c1]
    s[r1,c1] = s[r2,c2]
    s[r2,c2] = temp

def successor(s, cur_m):
    successor_list = []
    # Get the index of the blank tile
    row, col = np.where(s==0)
    row_idx, col_idx = row[0], col[0]
    # Get the 6 possible moves
    possible_move = row_move[row_idx] + col_move[col_idx]
    for move in possible_move:
        row, col = row_idx, col_idx
        s_next = np.copy(s)
        # Get direction and the number of tiles to move
        direction, n, _ = move
        n = int(n)
        if direction=='D':
            for i in range(n):
                swap(s_next,row,col,row-1,col)
                row -= 1
        elif direction=='U':
            for i in range(n):
                swap(s_next,row,col,row+1,col)
                row += 1
        elif direction=='R':
            for i in range(n):
                swap(s_next,row,col,row,col-1)
                col -= 1
        elif direction=='L':
            for i in range(n):
                swap(s_next,row,col,row,col+1)
                col += 1
        
        # Don't add the child if it's already checked
        if any((s_next == s).all() for s in puzzle_tracking):
            continue

        h = heuristic(s_next, G)
        m = cur_m + [move]
        c = h + len(m)
        successor_list.append([s_next, m, h, c])

    return successor_list
    
def is_goal(s):
    return (s == G).all()

def heuristic(s1, s2):
    return manhattan_distance(s1, s2)
    # return number_of_misplaced_tiles(s1, s2)

# f = heuristic + cost so far
def find_best_state(fringe):
    f_list = [s[3] for s in fringe]
    return f_list.index(min(f_list))

def solve(initial_board):
    global puzzle_tracking
    h = heuristic(initial_board, G)
    fringe = [[initial_board, [], h, h]]

    while len(fringe) > 0:
        s, m, _, c = fringe.pop(find_best_state(fringe))
        puzzle_tracking.append(s)
        if is_goal(s):
            return fringe, m  

        fringe.extend(successor(s, m))
    return False

def printable_result(path):
    return " ".join(path)

filename = sys.argv[1]
S0 = initial_state(filename)
puzzle_tracking = []

fringe, path = solve(S0)
print printable_result(path)
