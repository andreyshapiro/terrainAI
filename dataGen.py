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
foldername_loading = "to-load1"
foldername_saving = "to-load1"


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
    return np.load(foldername_loading + "/elev.npy")


def save_elev(array):
    np.save(foldername_saving+"/elev.npy", array, allow_pickle=True, fix_imports=True)


def load_water():
    return np.load(foldername_loading + "/water.npy")


def save_water(array1):
    np.save(foldername_saving+"/water.npy", array1, allow_pickle=True, fix_imports=True)



def load_veg():
    return np.load(foldername_loading + "/veg.npy")


def save_veg(array):
    np.save(foldername_saving+"/veg.npy", array, allow_pickle=True, fix_imports=True)


def load_fert_maps():
    return (np.load(foldername_loading + "/grass.npy"), np.load(foldername_loading + "/shrub.npy"),
            np.load(foldername_loading + "/tree.npy"))


def save_fert_maps(array1, array2, array3):
    np.save(foldername_saving+"/grass.npy", array1, allow_pickle=True, fix_imports=True)
    np.save(foldername_saving + "/shrub.npy", array2, allow_pickle=True, fix_imports=True)
    np.save(foldername_saving + "/tree.npy", array3, allow_pickle=True, fix_imports=True)


def load_slope():
    return np.load(foldername_loading + "/slope.npy"), np.load(foldername_loading + "/water_dist.npy")


def save_slope(array, water_distance):
    np.save(foldername_saving+"/slope.npy", array, allow_pickle=True, fix_imports=True)
    np.save(foldername_saving + "/water_dist.npy", water_distance, allow_pickle=True, fix_imports=True)


def load_set_map():
    return np.load(foldername_loading + "/set_map.npy")


def save_set_map(array):
    np.save(foldername_saving+"/set_map.npy", array, allow_pickle=True, fix_imports=True)


# gets struct, village, and, path graph
def load_structs():
    file = open(foldername_loading + "/villages.pickle", 'rb')
    file2 = open(foldername_loading + "/graph.pickle", 'rb')

    v = pickle.load(file)
    g = pickle.load(file2)

    file.close()
    file2.close()
    return (np.load(foldername_loading + "/struct.npy"), v, g)


def save_structs(array, villages, graph):
    np.save(foldername_saving+"/struct.npy", array, allow_pickle=True, fix_imports=True)

    if os.path.exists(foldername_saving+'/villages.pickle'):
        os.remove(foldername_saving+'/villages.pickle')

    if os.path.exists(foldername_saving+'/graph.pickle'):
        os.remove(foldername_saving+'/graph.pickle')

    file = open(foldername_saving + "/villages.pickle", 'ab')
    file2 = open(foldername_saving + "/graph.pickle", 'ab')

    pickle.dump(villages, file)
    file.close()

    pickle.dump(graph, file2)
    file2.close()


def save_ALL(elev, water, water_dist, slope, veg, grass, shrub, tree, set_map, struct, villages, graph):
    save_elev(elev)
    save_water(water)
    save_slope(slope, water_dist)
    save_veg(veg)
    save_fert_maps(grass,shrub,tree)
    save_set_map(set_map)
    save_structs(struct, villages, graph)


def load_ALL():
    elev = load_elev()
    water = load_water()
    slope, water_dist = load_slope()
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

