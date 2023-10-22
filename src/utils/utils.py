import numpy as np
import scipy.sparse as sp 


def load_data(matrix_file):
    file_type = matrix_file.split('.')[-1]
    if file_type == 'txt':
        data = np.loadtxt(matrix_file)
    elif file_type == 'csv':
        data = np.loadtxt(matrix_file, delimiter=',')
    elif file_type == 'npz':
        data = sp.load_npz(matrix_file)
    else:
        data = np.load(matrix_file) 

    # if file_type != 'npz':
        # print('data size={}'.format(data.shape))
    return data

def projection_barycentric(x, y, coupling, XontoY = True):
    '''
    projection function from SCOT: https://github.com/rsinghlab/SCOT
    '''
    if XontoY:
        #Projecting the first domain onto the second domain
        y_aligned=y
        weights=np.sum(coupling, axis = 0)
        X_aligned=np.matmul(coupling, y) / weights[:, None]
    else:
        #Projecting the second domain onto the first domain
        X_aligned = x
        weights=np.sum(coupling, axis = 0)
        y_aligned=np.matmul(np.transpose(coupling), x) / weights[:, None]

    return X_aligned, y_aligned

def subsampling(data, sample_size):
    linspace = np.linspace(0, data.shape[0] - 1, sample_size, dtype= int)
    data_new = data[linspace]
    return data_new