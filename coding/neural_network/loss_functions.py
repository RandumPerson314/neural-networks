import numpy as np

def MSE (guesses, real_answers):
    if len(guesses) != len(real_answers):
        print(f"{len(guesses)} != {len(real_answers)}")
        raise ValueError("guess and real answer are different lengths")
    else:
        error = 0
        
        for i in range(len(guesses)):
            error += (guesses[i] - real_answers[i])**2

        error /= len(guesses)
        return(error)