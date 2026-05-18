# goal state
cube_goal = {
    'initial': [['N', 'U', 'S'],
                ['N', 'U', 'S'],
                ['N', 'U', 'S']],
    'goal': [['N', 'U', 'S'],
             ['N', 'U', 'S'],
             ['N', 'U', 'S']],
    'solution': [],
}

# 3x3 cube
cube_3x3 = {
    'initial': [[1, 2, 0],
                [3, 4, 5],
                [6, 7, 8]],
    'goal': [[0, 1, 2],
             [3, 4, 5],
             [6, 7, 8]],
    'solution': [[0, 'right']],
}

# one step away from goal state
cube_one_step = {
    'initial': [['S', 'N', 'U'],
                ['N', 'U', 'S'],
                ['N', 'U', 'S']],
    'goal': [['N', 'U', 'S'],
             ['N', 'U', 'S'],
             ['N', 'U', 'S']],
    'solution': [[0, 'left']],
}

# two steps away from goal state
cube_two_steps = {
    'initial': [[1, 2, 3],
                [6, 4, 5],
                [0, 7, 8]],
    'goal': [[0, 1, 2],
             [3, 4, 5],
             [6, 7, 8]],
    'solution': [[0, 'right'], [0, 'down']],
}

# transposes the cube
cube_transpose = {
    'initial': [['S', 'O', 'C'],
                ['S', 'O', 'C'],
                ['S', 'O', 'C']],
    'goal': [['S', 'S', 'S'],
             ['O', 'O', 'O'],
             ['C', 'C', 'C']],
    'solution': [[2, 'right'], [1, 'left'], [1, 'down'], [2, 'up']],
}

# flips the cube
cube_flip = {
    'initial': [['N', 'U', 'S'],
                ['N', 'U', 'S'],
                ['N', 'U', 'S']],
    'goal': [['S', 'U', 'N'],
             ['N', 'S', 'U'],
             ['U', 'N', 'S']],
    'solution': [[0, 'left'], [1, 'right'], [0, 'up'], [1, 'down']],
}

# intermediate state for cube_flip
cube_flip_intermediate = {
    'initial': [['U', 'S', 'N'],
                ['N', 'U', 'S'],
                ['N', 'U', 'S']],
    'goal': [['S', 'U', 'N'],
             ['N', 'S', 'U'],
             ['U', 'N', 'S']],
    'solution': [[1, 'right'], [0, 'up'], [1, 'down']],
}


# 3x4 cube
cube_3x4 = {
    'initial': [[1, 1, 9, 0],
                [2, 2, 0, 2],
                [9, 0, 1, 9]],
    'goal': [[1, 0, 9, 2],
             [2, 1, 0, 9],
             [2, 1, 0, 9]],
    'solution': [[1, 'down'], [3, 'up'], [2, 'left']],
}

# 4x4 cube
cube_4x4 = {
    'initial': [[1, 2, 3, 4],
                [5, 6, 7, 8],
                [9, 0, 1, 2],
                [3, 4, 5, 6],],
    'goal': [[4, 1, 2, 3],
                [5, 6, 7, 8],
                [9, 0, 1, 2],
                [3, 4, 5, 6],],
    'solution': [[0, 'right']],
}
