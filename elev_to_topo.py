import numpy as np
from tqdm import tqdm
import matplotlib.cm as mat1
import matplotlib.colors as mat2


earth = mat1.get_cmap('gist_earth')
rgba = earth([[0.1,0.5],[0.03,1.0]])
print(rgba)

buffer = 4

def convo(data):
    (y,x,z) = np.shape(data)
    cdata = np.zeros((y,x))
    for i in range(buffer,y-buffer):
        for j in range(buffer, x-buffer):
            v = [0,0,0,0,0]
            for ib in range(-buffer,buffer+1):
                for jb in range(-buffer,buffer+1):
                    k = abs(ib) + abs(jb)
                    if(k<=4):
                        v[k] += data[i+ib][j+jb][0]
            cdata[i][j] = (v[0]/2) + (v[1]/16) + (v[2]/64) + (v[3]/192) + (v[4]/256)
    for i in range(buffer,y-buffer):
        for j in range(buffer, x-buffer):
            data[i][j][0] = cdata[i][j]


def e_to_t(data, interval):
    (y,x) = np.shape(data)
    ylen= y-1
    xlen =x-1
    normed = np.zeros((y,x))
    for j in tqdm(range(y)):
        for i in range(x):
            normed[j][i] = data[j][i]//interval
    out = np.zeros((y,x))
    for j in tqdm(range(ylen)):
        if (normed[j][xlen] != normed[j+1][xlen]):
            out[j][xlen] = 1
            out[j + 1][xlen] = 1
        for i in range(xlen):
            k = normed[j][i]
            if(k != normed[j][i+1]):
                out[j][i] = 1
                out[j][i+1] = 1
            if(k != normed[j+1][i]):
                out[j][i] = 1
                out[j+1][i] = 1
    for i in range(xlen):
        if (normed[ylen][i] != normed[ylen][i+1]):
            out[ylen][i] = 1
            out[ylen][i+1] = 1
    return out

def e_to_tF(data, interval):
    (y, x) = np.shape(data)
    ylen = y - 1
    xlen = x - 1
    normed = np.zeros((y, x))
    for j in tqdm(range(y)):
        for i in range(x):
            normed[j][i] = data[j][i] // interval
    out = data
    for j in tqdm(range(ylen)):
        if (normed[j][xlen] != normed[j + 1][xlen]):
            out[j][xlen] = 1
            out[j + 1][xlen] = 1
        for i in range(xlen):
            k = normed[j][i]
            if (k != normed[j][i + 1]):
                out[j][i] = 1
                out[j][i + 1] = 1
            if (k != normed[j + 1][i]):
                out[j][i] = 1
                out[j + 1][i] = 1
    for i in range(xlen):
        if (normed[ylen][i] != normed[ylen][i + 1]):
            out[ylen][i] = 1
            out[ylen][i + 1] = 1
    return out

def eShader(data):
    a =a






