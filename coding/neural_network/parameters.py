import numpy as np
rng = np.random.default_rng()

def xavier_uniform_initialization(rows, cols):
    limit = np.sqrt(6/(rows+cols))
    return np.random.uniform(-limit, limit,(rows,cols))

def xavier_normal_initialization(rows, cols):
    std = np.sqrt(2.0 / (rows + cols))
    return np.random.normal(0, std, (rows,cols))

def he_normal_initialization(rows, cols):
    std = np.sqrt(2 / rows)
    return np.random.randn(rows, cols) * std

def random_initialization(rows, cols):
    return np.random.rand(rows, cols) / cols

initializations = {
    "xavier_uniform": xavier_uniform_initialization,
    "xavier_normal": xavier_normal_initialization,
    "he_normal": he_normal_initialization,
    "random": random_initialization
}

def weight_matrix(rows, cols, initialization):
    matrix = np.zeros((rows, cols))
    matrix = initializations[initialization](rows, cols)
    return(matrix)

def bias(rows, initialization):
    v_b = np.zeros(rows)
    if initialization.lower() == "random":
        for i in range(rows):
            v_b[i] = rng.random()
    else:
        raise ValueError("Invalid initialization method")
    return(v_b)