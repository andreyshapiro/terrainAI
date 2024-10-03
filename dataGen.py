import matplotlib.pyplot as plt
import numpy as np
from osgeo import gdal
from tqdm import tqdm
import pickle
import os
gdal.UseExceptions()

runREST = 0
run = 0
dataLen = 200 * 4 * 4
foldername = "to-load1"


def get_sample(size):
    ds = gdal.Open('data_in/in.tif')  # 'data_in/in.tif')
    band = ds.GetRasterBand(1)
    elevation = band.ReadAsArray()

    xlen = len(elevation)
    ylen = len(elevation[0])



    x = np.random.randint(0, xlen-size+1)
    y = np.random.randint(0, ylen-size+1)

    e = elevation[x:x+size,y:y+size]
    ema = np.amax(e)
    emi = np.amin(e)
    e -= emi
    e /= (ema-emi)
    return e


def get_sample_unnormed(size):
    ds = gdal.Open('data_in/in.tif')  # 'data_in/in.tif')
    band = ds.GetRasterBand(1)
    elevation = band.ReadAsArray()

    xlen = len(elevation)
    ylen = len(elevation[0])



    x = np.random.randint(0, xlen-size+1)
    y = np.random.randint(0, ylen-size+1)

    e = elevation[x:x+size,y:y+size]
    return e


def load_elev():
    return np.load(foldername + "/elev.npy")


def save_elev(array):
    np.save(foldername+"/elev.npy", array, allow_pickle=True, fix_imports=True)


def load_water():
    return np.load(foldername + "/water.npy"), np.load(foldername + "/water_dist.npy")


def save_water(array1, array2):
    np.save(foldername+"/water.npy", array1, allow_pickle=True, fix_imports=True)
    np.save(foldername + "/water_dist.npy", array2, allow_pickle=True, fix_imports=True)


def load_veg():
    return np.load(foldername + "/veg.npy")


def save_veg(array):
    np.save(foldername+"/veg.npy", array, allow_pickle=True, fix_imports=True)


def load_fert_maps():
    return (np.load(foldername + "/grass.npy"), np.load(foldername + "/shrub.npy"), np.load(foldername + "/tree.npy"))


def save_fert_maps(array1, array2, array3):
    np.save(foldername+"/grass.npy", array1, allow_pickle=True, fix_imports=True)
    np.save(foldername + "/shrub.npy", array2, allow_pickle=True, fix_imports=True)
    np.save(foldername + "/tree.npy", array3, allow_pickle=True, fix_imports=True)


def load_slope():
    return np.load(foldername + "/slope.npy")


def save_slope(array):
    np.save(foldername+"/slope.npy", array, allow_pickle=True, fix_imports=True)


def load_set_map():
    return np.load(foldername + "/set_map.npy")


def save_set_map(array):
    np.save(foldername+"/set_map.npy", array, allow_pickle=True, fix_imports=True)


# gets struct, village, and, path graph
def load_structs():
    file = open(foldername + "/villages.pickle", 'rb')
    file2 = open(foldername + "/graph.pickle", 'rb')

    v = pickle.load(file)
    g = pickle.load(file2)

    file.close()
    file2.close()
    return (np.load(foldername + "/struct.npy"), v, g)


def save_structs(array, villages, graph):
    np.save(foldername+"/struct.npy", array, allow_pickle=True, fix_imports=True)

    if os.path.exists(foldername+'/villages.pickle'):
        os.remove(foldername+'/villages.pickle')

    if os.path.exists(foldername+'/graph.pickle'):
        os.remove(foldername+'/graph.pickle')

    file = open(foldername + "/villages.pickle", 'ab')
    file2 = open(foldername + "/graph.pickle", 'ab')

    pickle.dump(villages, file)
    file.close()

    pickle.dump(graph, file2)
    file2.close()


def save_ALL(elev, water, water_dist, slope, veg, grass, shrub, tree, set_map, struct, villages, graph):
    save_elev(elev)
    save_water(water, water_dist)
    save_slope(slope)
    save_veg(veg)
    save_fert_maps(grass,shrub,tree)
    save_set_map(set_map)
    save_structs(struct, villages, graph)


def load_ALL():
    elev = load_elev()
    water, water_dist = load_water()
    slope = load_slope()
    veg = load_veg()
    grass, shrub, tree = load_fert_maps()
    set_map = load_set_map()
    struct, villages, graph = load_structs()
    return elev, water, water_dist, slope, veg, grass, shrub, tree, set_map, struct, villages, graph


if runREST:
    ds = gdal.Open('data_in/in.tif')#'data_in/in.tif')
    band = ds.GetRasterBand(1)
    elevation = band.ReadAsArray()
    print(len(elevation))
    brick = 64
    l = len(elevation)
    cnt = l//brick

    cut = brick*cnt

    out_array = elevation[0:cut,0:cut].reshape(cnt,brick,cnt,brick).swapaxes(1,2).reshape(cnt*cnt,brick,brick)

    new_array = np.reshape(out_array[0:dataLen],(dataLen,1,brick,brick))

    np.save('data_out/out_200_64bit', new_array[0:dataLen], allow_pickle=True, fix_imports=True)

if run:
    ha = np.load('data_out/out_200_64bit.npy')
    print(np.shape(ha))
    print(ha.dtype)
    plt.imshow(ha[20][0], cmap='gist_earth')
    plt.show()

