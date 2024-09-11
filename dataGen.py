import matplotlib.pyplot as plt
import numpy as np
from osgeo import gdal
from tqdm import tqdm
gdal.UseExceptions()

runREST = 0
dataLen = 200 * 4 * 4

if runREST:
    ds = gdal.Open('data_in/in.tif')#'data_in/in.tif')
    band = ds.GetRasterBand(1)
    elevation = band.ReadAsArray()

    brick = 64
    l = len(elevation)
    cnt = l//brick

    cut = brick*cnt

    out_array = elevation[0:cut,0:cut].reshape(cnt,brick,cnt,brick).swapaxes(1,2).reshape(cnt*cnt,brick,brick)

    new_array = np.reshape(out_array[0:dataLen],(dataLen,1,brick,brick))

    np.save('data_out/out_200_64bit', new_array[0:dataLen], allow_pickle=True, fix_imports=True)


ha = np.load('data_out/out_200_64bit.npy')
print(np.shape(ha))
print(ha.dtype)
plt.imshow(ha[20][0], cmap='gist_earth')
plt.show()

