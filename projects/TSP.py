import numpy as np
from python_tsp.exact import solve_tsp_dynamic_programming
def custom_tsp(distance_matrix):
    permutation, distance = solve_tsp_dynamic_programming(distance_matrix)
    return permutation, distance