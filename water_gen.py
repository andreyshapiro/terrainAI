import numpy as np
from tqdm import tqdm
import matplotlib.cm as mat1
import matplotlib.colors as mat2
import matplotlib.pyplot as plt
import handy_functions
import dataGen


# returns an array of spring locations with a density as input
def spring(datain,density, xlen, ylen):
    #spring gen prop to height for now
    ma = np.amax(datain)
    mi = np.amin(datain)
    prod = density / (ma - mi)

    springs = []
    for x in tqdm(range(xlen)):
        for y in range(ylen):
            if np.random.random() < prod * (datain[x][y] - mi):
               springs.append((x,y))
    return springs


# takes in a list of springs and uses them to create water streams. Unlimited water version
def draw_water(datain, springs):
    xlen = len(datain)
    ylen = len(datain[0])
    water = np.zeros((xlen,ylen))
    for (cx,cy) in tqdm(springs):
        (x,y) = (cx,cy)
        cont = True
        while cont:
            if water[x][y] < 1: water[x][y]=1
            datain[x][y] += .001
            (ax,ay, minVal) = handy_functions.getMinNbr(datain, xlen, ylen, x, y)

            # if we stay put, increase the water level
            if (x,y) == (ax, ay):
                datain[x][y] += .005
                water[x][y] += .01
            elif ax<0:
                # if out of bounds, we stop
                break
            else:
                (x,y) = (ax,ay)
    return water


# same as draw_water but allows the water to erode to hopefully create more realistic paths
# Main Parameters:
#   erode_rate: the rate at which the flowing water erodes nearby tiles
#   water_height: the default height of water
#   water_increase_height: the amount water raises in height if it finds itself in a hole - good for filling in lakes
#   max_momentum: water accumulates momentum over time if it does not erode. This is the max momentum.
#   momentum: increases the rate of erosion
def draw_water_erode(datain, springs):
    xlen = len(datain)
    ylen = len(datain[0])
    water = np.zeros((xlen,ylen))

    erode_rate = .0003
    water_height = .001
    water_increase_height = .005
    max_momentum = 6

    for (cx,cy) in tqdm(springs):
        (x,y) = (cx,cy)
        cont = True
        momentum = 0
        while cont:
            water[x][y] += water_height
            datain[x][y] += water_height
            (ax, ay, minVal) = handy_functions.getMinNbr(datain, xlen, ylen, x, y)

            nbrs = handy_functions.get_valid_nbrs(xlen,ylen,x,y)
            nbr_cnt = 0
            for (nx,ny) in nbrs:
                if water[nx][ny] == 0:
                    nbr_cnt += 1
                    datain[nx][ny] -= erode_rate * momentum
            if nbr_cnt<6:
                # if few eroded neighbors, we speed up
                momentum += 1
            elif nbr_cnt>=7:
                # if many eroded neighbors, we slow down
                momentum -= 1
            if momentum <0: momentum = 0
            elif momentum>max_momentum: momentum=max_momentum

            # if we stay put, increase the water level
            if (x,y) == (ax, ay):
                datain[x][y] += water_increase_height
                water[x][y] += water_increase_height
            elif ax<0:
                # if out of bounds, we stop
                break
            else:
                (x,y) = (ax,ay)
    return water


# same as draw_water_erode breaks momentum to avoid straight lines (by depositing sediment once max momentum is reached)
# runs cycles of water gen and drying to create paths
# Main Parameters and Variables:
#   erode_rate: the rate at which the flowing water erodes nearby tiles
#   water_height: the default height of water
#   water_increase_height: the amount water raises in height if it finds itself in a hole - good for filling in lakes
#   max_momentum: water accumulates momentum over time if it does not erode. This is the max momentum.
#   momentum: increases the rate of erosion
#   direction: the direction water is currently flowing. Ranges [-4,4] and calculated by: dx + 3*dy
def draw_water_erode2(datain, springs, itter):
    xlen = len(datain)
    ylen = len(datain[0])
    water = np.zeros((xlen, ylen))

    erode_rate = .18 / handy_functions.unit_height
    water_height = .2 / handy_functions.unit_height
    water_increase_height = 1 / handy_functions.unit_height
    max_momentum = 10

    for i in tqdm(range(itter)):
        datain = np.subtract(datain,water)
        water = np.zeros((xlen, ylen))

        for (cx,cy) in springs:
            (x,y) = (cx,cy)
            cont = True
            momentum = 0
            direction = 0 # -4 to 4. 0 is same point
            sediment = 0
            while cont:
                # once we surpass max momentum, we will drop off all sediment directly in front of us.
                if momentum > max_momentum:
                    dx = direction % 3
                    if dx == 2: dx = -1
                    dy = (direction-dx)//3

                    tx = x+dx
                    ty = y+dy

                    if tx>=0 and ty>=0 and tx<xlen and ty<ylen:
                        if water[tx][ty]==0:
                            datain[tx][ty] += sediment
                            momentum = 0
                        else:
                            momentum = max_momentum
                        sediment = 0

                water[x][y] += water_height
                datain[x][y] += water_height

                (ax, ay, minVal) = handy_functions.getMinNbr(datain, xlen, ylen, x, y)

                if momentum > 0:
                    nbrs = handy_functions.get_valid_nbrs(xlen,ylen,x,y)

                    for (nx,ny) in nbrs:
                        if water[nx][ny] == 0:
                            datain[nx][ny] -= erode_rate * momentum
                            sediment += erode_rate * momentum

                direction2 = x - ax + 3 * (y - ay)
                if direction == direction2:
                    momentum += 1
                else:
                    momentum -= 1
                    if momentum<0: momentum=0

                # if we stay put, increase the water level and loose all momentum and sediment
                if direction2 == 0:
                    datain[x][y] += water_increase_height
                    water[x][y] += water_increase_height
                    momentum = 0
                    sediment = 0
                elif ax<0:
                    # if out of bounds, we stop
                    break
                else:
                    (x,y) = (ax,ay)
                direction = direction2
    return water


if False:
    datain = dataGen.get_sample(2048) #handy_functions.erode_Semi(handy_functions.genSample(256,256),10)
    s = spring(datain, 1/(256*256), 2048, 2048)
    datain2 = np.copy(datain)
    water = draw_water_erode2(datain2,s,25)

    hill = handy_functions.hillshade(datain, 0,30)
    hill2 = handy_functions.hillshade(datain2, 0, 30)
    first_row = np.concatenate((datain, datain2), axis=1)
    second_row = np.concatenate((datain, water*10), axis=1)

    plt.imshow(water, cmap='gist_earth')
    plt.show()

    handy_functions.plot_hillshade(first_row, np.concatenate((hill,hill2),axis=1))

    plt.imshow(np.concatenate((first_row,second_row),axis=0),cmap='gist_earth')
    plt.show()



