import numpy as np
from tqdm import tqdm
import matplotlib.cm as mat1
import matplotlib.colors as mat2
import matplotlib.pyplot as plt
import handy_functions
import water_gen
import dataGen

unit_height = handy_functions.unit_height  # feet
unit_length = handy_functions.unit_length  # feet
water_max = handy_functions.water_max // unit_length # tiles

# Note to self: this can be improved by making it one function. And calculating slope via np.gradient
# can speed up water_distance.


# given input elev data and water data, returns data needed for vegitation generation
def create_data(datain,water):

    xlen = len(datain)
    ylen = len(datain[0])
    dataout2 = np.full((xlen,ylen), water_max**2 + 1)

    index_array = []
    for i in range(-water_max,water_max+1):
        ki = np.abs(i)**2
        for j in range(-water_max,water_max+1):
            k = ki + np.abs(j)**2
            if k<=water_max**2:
                index_array.append((i,j,k))


    print("by row, gen_data:")
    for x in tqdm(range(xlen)):
        for y in range(ylen):
            # find squared distance to water
            if water[x][y]>0:
                for (i,j,k) in index_array:
                    xi = x+i
                    yj = y+j
                    if xi>=0 and xi<xlen and yj>=0 and yj<ylen and k < dataout2[xi][yj]:
                        dataout2[xi][yj]=k
    # normalise to given units
    dataout2 *= unit_length*unit_length
    dx, dy = np.gradient(datain)
    dx *= unit_height / unit_length
    dy *= unit_height / unit_length
    slope = np.arctan(np.sqrt(dx * dx + dy * dy))

    return slope, dataout2


# creates a fertility map triplet (tree, shrub, grass)
# trees suffer quadratically more from slope
# grass suffers quadratically more from distance to water.
def create_fertility(datain,water):

    xlen = len(datain)
    ylen = len(datain[0])

    (slope_data, water_data) = create_data(datain, water)

    # max values for each type of plant and factor. Slopes are in radians, water distances are squared.
    tree_slope = 0.75
    shrub_slope = 1
    grass_slope = 1.25
    tree_water = 1500**2  # feet^2
    shrub_water = 750**2  # feet^2
    grass_water = 375**2  # feet^2

    f_tree = np.full((xlen, ylen),0.0)
    f_shrub = np.full((xlen, ylen), 0.0)
    f_grass = np.full((xlen, ylen), 0.0)

    print("by row, gen fert:")
    for x in tqdm(range(xlen)):
        for y in range(ylen):
            if water_data[x][y] == 0:
                sT=sS=sG=wT=wS=wG=0
            else:
                sT = max(0, 1 - slope_data[x][y]/tree_slope)
                sS = max(0, 1 - slope_data[x][y]/shrub_slope)
                sG = max(0, 1 - slope_data[x][y]/grass_slope)

                wT = max(0, 1 - water_data[x][y]/tree_water)
                wS = max(0, 1 - water_data[x][y]/shrub_water)
                wG = max(0, 1 - water_data[x][y]/grass_water)

            f_tree[x][y] = sT**3 * wT
            f_shrub[x][y] = sS * sS * wS * wS
            f_grass[x][y] = sG * wG**3
    return (f_tree, f_shrub, f_grass, slope_data, water_data)


# randomly generates vegetation
def gen_veg(datain,water):
    # Note to self: consider generating batches of batches
    # parameter controlling how frequently batches appear. The higher, the less frequent
    gen_param_tree = 1200
    gen_param_shrub = 1000
    gen_param_grass = 25
    # parameters controlling how many per batch and how much they disperse
    tree_disperse = 80
    tree_num = .04
    shrub_disperse = 40
    shrub_num = .2
    grass_disperse = 30
    grass_num = .1

    xlen = len(datain)
    ylen = len(datain[0])
    veg = np.zeros((xlen,ylen,3))

    (tree, shrub, grass, slope, water_data) = create_fertility(datain, water)

    # first find where patches are to be generated.
    tree_seeds = []
    shrub_seeds = []
    grass_seeds = []

    print("by row, deciding seeds:")
    for x in tqdm(range(xlen)):
        for y in range(ylen):
            r = np.random.random() * gen_param_tree
            if r < tree[x][y]:
                tree_seeds.append((x,y))

            r = np.random.random() * gen_param_shrub
            if r < shrub[x][y]:
                shrub_seeds.append((x, y))

            r = np.random.random() * gen_param_grass
            if r < grass[x][y]:
                grass_seeds.append((x, y))

    print("planting grass:")
    grass_sphere = handy_functions.create_sphere(grass_disperse)
    for (x,y) in tqdm(grass_seeds):
        for (i,j) in grass_sphere:
            xi = x+i
            yj = y+j
            if (xi>=0 and yj>=0 and xi<xlen and yj<ylen
                    and grass[xi][yj]!= 0):
                r = np.random.random()
                if r<grass_num * grass[xi][yj]:
                    veg[xi][yj] = [1,0,0]
    print("planting shrubs:")
    shrub_sphere = handy_functions.create_sphere(shrub_disperse)
    for (x,y) in tqdm(shrub_seeds):
        for (i,j) in shrub_sphere:
            xi = x+i
            yj = y+j
            if (xi>=0 and yj>=0 and xi<xlen and yj<ylen):
                r = np.random.random()
                if r<shrub_num * shrub[xi][yj]:
                    veg[xi][yj] = [0,0,1]
    print("planting trees:")
    tree_sphere = handy_functions.create_sphere(tree_disperse)
    for (x,y) in tqdm(tree_seeds):
        for (i,j) in tree_sphere:
            xi = x+i
            yj = y+j
            if (xi>=0 and yj>=0 and xi<xlen and yj<ylen):
                r = np.random.random()
                if r<tree_num * tree[xi][yj]:
                    veg[xi][yj] = [0,1,0]
    return veg, tree, shrub, grass, slope, water_data

if False:
    datain = dataGen.get_sample_unnormed(2048) #handy_functions.erode_Semi(handy_functions.genSample(256,256),10)
    unit_height = 1
    unit_length = 10

    s = water_gen.spring(datain,1/(256*256), 2048, 2048)
    datain2 = np.copy(datain)
    water = water_gen.draw_water_erode2(datain2,s,14)

    veg, tree, shrub, grass, slope, water_dist = gen_veg(datain2,water)


    plt.imshow(slope,cmap='Greys')
    plt.show()

    plt.imshow(water_dist,cmap='Oranges')
    plt.show()

    plt.imshow(np.concatenate((tree,shrub,grass), axis = 1),cmap='Greens')
    plt.show()


    first_row = np.concatenate((datain, water * 100), axis = 1)
    #second_row = np.concatenate((veg, np.subtract(veg,water*10)), axis = 1)
    plt.imshow(water, cmap='gist_earth')#np.concatenate((first_row, second_row), axis =0),cmap='gist_earth')
    plt.show()

    plt.imshow(veg)#cmap='Dark2_r')
    plt.show()

    for i in range(len(veg)):
        for j in range(len(veg[0])):
            if (veg[i][j] == [0,0,0]).all():
                veg[i][j] = [datain[i][j],datain[i][j],datain[i][j]]

    plt.imshow(veg)#cmap='Dark2_r')
    plt.show()

