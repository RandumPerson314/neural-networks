import numpy as np
rng = np.random.default_rng()

def weight_matrix(rows, cols, initialization):
    matrix = np.zeros((rows, cols))
    if initialization.lower() == "random":
        for i in range(rows):
            row = []
            for j in range(cols):
                row.append(rng.random()/cols) 
            matrix[i] = row
    else:
        raise ValueError("Invalid initialization method")
    return(matrix)

def bias(rows, initialization):
    v_b = np.zeros(rows)
    if initialization.lower() == "random":
        for i in range(rows):
            v_b[i] = rng.random()
    else:
        raise ValueError("Invalid initialization method")
    return(v_b)