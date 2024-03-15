import numpy as np 


def create_position_matrix(matrix=None, **kwargs):
    position = np.zeros_like(matrix, dtype=bool)
    for j in range(matrix.shape[0]):
        for i in range(j + 1, matrix.shape[1]):
            if matrix[i][j] <= matrix[j][j]:
                position[i, j] = True
    return position


def average_compatibility(matrix=None, position=None, **kwargs):
    steps = matrix.shape[0]
    if position is None:
        position = create_position_matrix(matrix)
    max_ac = (steps * (steps-1)) / 2
    if max_ac < 1:
        max_ac = 1
    ac = max_ac - np.sum(position)
    return (1/max_ac) * ac


def replace_zero_with_nan(matrix, **kwargs):
    idx = np.where(matrix == 0)
    matrix[idx] = np.nan
    return matrix

def average_accuracy(matrix=None, per_task=False, **kwargs):
    if max(matrix[-1]) < 1:
        matrix = matrix * 100
    copy_matrix = matrix.copy()
    values = [np.nanmean(replace_zero_with_nan(copy_matrix)[:i+1,:i+1]) for i in range(copy_matrix.shape[0])]
    
    if per_task:
        return values
    else:
        return values[-1]
    

def AC_per_task(position=None, **kwargs):
    values = [average_compatibility(position[:i+1,:i+1], position=position[:i+1,:i+1]) for i in range(1,position.shape[0])]
    return values

