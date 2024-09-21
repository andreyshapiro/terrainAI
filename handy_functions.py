import numpy as np
from tqdm import tqdm
import matplotlib.cm as mat1
import matplotlib.colors as mat2
import matplotlib.pyplot as plt
#import earthpy.spatial as es
import dataGen

buffer = 4
unit_length = 33 # feet
unit_height = 1 # feet
water_max = 300 # feet

# Returns a list of valid neighbors (a,b)
def get_valid_nbrs(xlen, ylen, x, y):
    out = []
    xB = x>0
    xS = x<xlen-1
    yB = y>0
    yS = y<ylen-1
    if xB:
        out.append((x-1,y))
        if yB: out.append((x-1,y-1))
        if yS: out.append((x-1,y+1))
    if xS:
        out.append((x+1, y))
        if yB: out.append((x+1, y-1))
        if yS: out.append((x+1, y+1))
    if yB: out.append((x,y-1))
    if yS: xB:out.append((x,y+1))

    return out

# Creates a sphere of indecies centered at (0,0)
def create_sphere(radius):
    out = []
    r2 = radius**2
    for i in range(-radius,radius+1):
        for j in range(-radius,radius+1):
            if i**2 + j**2 <= r2: out.append((i,j))
    return out

# Runs simple local convolution
def convo(data):
    (y,x) = np.shape(data)
    cdata = np.zeros((y,x))
    for i in tqdm(range(buffer,y-buffer)):
        for j in range(buffer, x-buffer):
            v = [0,0,0,0,0]
            for ib in range(-buffer,buffer+1):
                for jb in range(-buffer,buffer+1):
                    k = abs(ib) + abs(jb)
                    if(k<=4):
                        v[k] += data[i+ib][j+jb]
            cdata[i][j] = (v[0]/2) + (v[1]/16) + (v[2]/64) + (v[3]/192) + (v[4]/256)
    for i in range(buffer,y-buffer):
        for j in range(buffer, x-buffer):
            data[i][j] = cdata[i][j]

# Turns an elevation map into a contour map at specified intervals via marching squares algorithm
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

# Second attempt at turning elevation map to contour map (UNFINISHED)
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

# Gets min neighbor (of 9) with off-screen being -infty (in which case it returns (-1,-1,-1).
# Option to add modifier to central square
def getMinNbr(data, xlen, ylen, x ,y, *args):

    if (x==0 or y==0 or x==xlen-1 or y==ylen-1):
        return (-1,-1,-1)
    else:

        minVal = data[x][y]
        if len(args)>0:
            minVal += args[0]

        (ax, ay) = (x, y)
        if data[x-1][y]<minVal:
            minVal = data[x-1][y]
            (ax, ay) = (x-1,y)
        if data[x][y-1]<minVal:
            minVal = data[x][y-1]
            (ax, ay) = (x,y-1)
        if data[x+1][y]<minVal:
            minVal = data[x+1][y]
            (ax, ay) = (x+1,y)
        if data[x][y+1]<minVal:
            minVal = data[x][y+1]
            (ax, ay) = (x,y+1)
        if data[x+1][y+1]<minVal:
            minVal = data[x+1][y+1]
            (ax, ay) = (x+1,y+1)
        if data[x-1][y+1]<minVal:
            minVal = data[x-1][y+1]
            (ax, ay) = (x-1,y+1)
        if data[x+1][y-1]<minVal:
            minVal = data[x+1][y-1]
            (ax, ay) = (x+1,y-1)
        if data[x-1][y-1]<minVal:
            minVal = data[x-1][y-1]
            (ax, ay) = (x-1,y-1)
        return (ax, ay, minVal)

# Simulates erosion via many random droplets. Droplet will collect sediment, at each step it will collect
# a fraction of it's remaining capacity and evaporate a bit. This continues until the droplet evaporates or
# gets stuck in a hole.
# Main Parameters:
#   collection_rate: fraction of remaining capacity (volume of the droplet) to be collectd at each step
#   size: starting volume of the droplet
#   evap_rate: volume of the droplet will decrease by this much at each step - any spillover of sediment will
# be deposited immediately.
def erode_Random(datain, itter):
    xlen = len(datain)
    ylen = len(datain[0])

    data=np.copy(datain)

    collection_rate = .5
    size = .0003
    evap_rate = size/40 # should divide size evenly

    for i in tqdm(range(itter)):
        water = size
        sediment = 0

        x = np.random.randint(0, xlen)
        y = np.random.randint(0, ylen)

        while(water>0):
            # evaporate
            water -= evap_rate
            if water >= sediment:
                cap = water - sediment
            else:
                # deposit spillover
                data[x][y] += sediment - water
                sediment = water
                cap = 0

            # collect sediment
            collect = cap * collection_rate
            collect2 = collect / 10
            collect_count = 2  # automatically pick up two on the spot since no conflicts
            data[x][y] -= 2 * collect2

            # collect half as much from neighbors
            if x > 0:
                data[x - 1][y] -= collect2
                collect_count += 1
                if y > 0:
                    data[x - 1][y - 1] -= collect2
                    collect_count += 1
                if y < ylen - 1:
                    data[x - 1][y + 1] -= collect2
                    collect_count += 1
            if x < xlen - 1:
                data[x + 1][y] -= collect2
                collect_count += 1
                if y > 0:
                    data[x + 1][y - 1] -= collect2
                    collect_count += 1
                if y < ylen - 1:
                    data[x + 1][y + 1] -= collect2
                    collect_count += 1
            if y > 0:
                data[x][y - 1] -= collect2
                collect_count += 1
            if y < ylen - 1:
                data[x][y - 1] -= collect2
                collect_count += 1

            sediment += collect_count * collect2

            # step
            (ax, ay, minVal) = getMinNbr(data, xlen, ylen, x, y)
            if ax<0:
                # if out of bounds exit loop
                water = 0
            if (x,y)==(ax,ay):
                # if droplet is in a hole, deposit all sediment and quit
                data[x][y] += sediment
                water = 0
            # update step
            (x,y) = (ax,ay)
    return data

# Simulates erosion via simultaneous rain droplets (UNFINISHED)
def erode_Simul(datain, itter):
    xlen = len(datain)
    ylen = len(datain[0])

    data = np.copy(datain)

    collection_rate = .4
    size = .0005
    size_grid = np.full(shape=(xlen, ylen), fill_value=size)
    e_r = size/10
    evap_rate = np.full(shape=(xlen, ylen), fill_value=e_r)  # should devide size evenly
    zer = np.zeros((xlen, ylen))

    for i in tqdm(range(itter)):
        water = np.copy(size_grid)
        water2 = np.copy(size_grid)
        waterCount = xlen*ylen
        sediment = np.copy(zer)
        sediment2 = np.copy(sediment)
        cap = size

        c=0
        o=0
        d=0
        m=0
        PRINT = False

        while (waterCount > 0):
            # evaporate
            d_temp = np.add(data, water2, sediment2)
            water = np.subtract(water2, evap_rate)
            water2 = np.copy(zer)
            sediment = np.copy(sediment2)
            sediment2 = np.copy(zer)
            for x in range(xlen):
                for y in range(ylen):
                    if water[x][y]<=0 and water[x][y]>-e_r:
                        water[x][y] = 0
                        waterCount -= 1
                        c += 1
                        if PRINT: print("evaporated", c, "out", o)
                    elif water[x][y]<0:
                        water[x][y] = 0
                        continue
                    if water[x][y] >= sediment[x][y]:
                        cap = water[x][y] - sediment[x][y]
                    else:
                        # deposit
                        data[x][y] += sediment[x][y] - water[x][y]
                        sediment[x][y] = water[x][y]
                        cap = 0

                    # collect sediment
                    collect = cap * collection_rate
                    sediment[x][y] += collect
                    data[x][y] -= collect

                    # step
                    (ax, ay, minVal) = getMinNbr(d_temp, xlen, ylen, x, y)
                    if ax < 0:
                        # if out of bounds remove water
                        #water[x][y] = 0
                        #sediment[x][y]=0
                        waterCount -= 1
                        o+= 1
                        if PRINT: print("out", o,m, "evaporated", c, d)
                    elif (x,y)==(ax,ay):
                        #if stuck in place, safely drain water and keep sediment
                        waterCount -= 1
                        data[x][y] += sediment[x][y]
                        d+=1
                        if PRINT: print("evaporated", c,d, "out", o, m)
                    else:
                        # update step
                        #check for merging
                        if water2[ax][ay]>0 and water[x][y]>0:
                            waterCount -= 1
                            m+=1
                            if PRINT: print("merged",m, "other",o,c,d)

                        #nned to check that we aren't causing water to flow upwards due to other water
                        # if we are, need to calculate how much of the water is transfered
                        down_control = 1
                        t = water[x][y] + sediment[x][y] + data[x][y]  # above ground value

                        if data[x][y]<minVal:


                            # fraction of water and sediment to be transfered
                            # (half of the portion above the other water)
                            down_control = (1 + (data[x][y] - minVal)/t)/2
                            # if split occurs, need to ensure that we add to waterCount
                            if water2[x][y] == 0 and water2[ax][ay]==0:
                                waterCount+=1
                            water2[x][y] += water[x][y] * (1-down_control)
                            sediment2[x][y] += sediment[x][y] * (1-down_control)

                        water2[ax][ay] += water[x][y] * down_control
                        sediment2[ax][ay] += sediment[x][y] * down_control

    print("quiting!")
    return data

# Simulates erosion via few semi-simultaneous rain droplets over several iterations.
# Droplets are semi-simultaneous in that their paths are determined sequentially but
# can still interact with each other's heights, thereby allowing wide riverbeds to form.
# Main Parameters:
#   water_Volume: how many water droplets are generated semi-simultaneously.
#   collection_rate: fraction of remaining capacity (volume of the droplet).
#   size: starting volume of the droplet.
#   stepLen: maximum length of a path a droplet can take. This really just functions
# as a failsafe against infinite loops.
# Note: water_Volume and size may be set dynamically. Generally we want to run a few batches
# of large water_Volume followed by many iterations of small water_Volume
def erode_Semi(datain,itter):
    xlen = len(datain)
    ylen = len(datain[0])

    # This is the elevation data of each square including sediment and any stable water
    data = np.copy(datain)

    water_Volume = 1 #65536

    # Note to self: consider making the collection rate larger than size
    # (.04,.0001) for good rivers, (.04,.0005) WV 20.
    collection_rate = .1
    size = .001
    stepLen = 1200

    reps = itter #131072 * 2 - 2 #itter*ylen*xlen//160#//2000

    l = 1
    # 2^16

    for i in tqdm(range(reps)):

        # dynamic water_Volume
        # Note to self: currently this is configured for 256x256
        # it may be better to set a dynamic allocation more explicitly.
        # if i>l and l<16:
        #     l = l * 2
        #     if l>= 16:
        #         water_Volume = 1
        #     else:
        #         water_Volume=water_Volume // 2

        # degenerate case with one droplet
        # this is essentially erode_Random but without evaporation
        if(water_Volume == 1):
            sediment = 0

            # rain-fall
            x = np.random.randint(0, xlen)
            y = np.random.randint(0, ylen)

            cont = True
            step = 0

            while (cont):

                # collect sediment
                if sediment < size:
                    collect = collection_rate * (size - sediment)
                    collect2 = collect / 10
                    collect_count = 2 # automatically pick up two since no conflicts

                    # collect half as much from neighbors
                    if x > 0:
                        data[x - 1][y] -= collect2
                        collect_count += 1
                        if y > 0:
                            data[x - 1][y - 1] -= collect2
                            collect_count += 1
                        if y < ylen - 1:
                            data[x - 1][y + 1] -= collect2
                            collect_count += 1
                    if x < xlen - 1:
                        data[x + 1][y] -= collect2
                        collect_count += 1
                        if y > 0:
                            data[x + 1][y - 1] -= collect2
                            collect_count += 1
                        if y < ylen - 1:
                            data[x + 1][y + 1] -= collect2
                            collect_count += 1
                    if y > 0:
                        data[x][y - 1] -= collect2
                        collect_count += 1
                    if y < ylen - 1:
                        data[x][y - 1] -= collect2
                        collect_count += 1

                    sediment += collect_count * collect2
                    data[x][y] += collect_count * collect2

                # step
                (ax, ay, minVal) = getMinNbr(data, xlen, ylen, x, y, -sediment)
                if ax < 0:
                    # if out of bounds exit loop
                    # the tile looses its sediment, water never becomes stable and is lost
                    data[x][y] -= sediment
                    break
                if (x, y) == (ax, ay) or step > stepLen:
                    # if droplet is in a hole or ran out of movement,
                    # deposit all sediment (done implicitly)
                    # and stop water movement (done by adding size of water)
                    data[x][y] += size
                    break
                # update step
                data[x][y] -= sediment
                data[ax][ay] += sediment
                (x, y) = (ax, ay)

                step += 1
        else:
            # The case where water_Volume is not 1

            water = np.zeros((xlen, ylen))
            sediment = 0

            # rain-fall
            for j in range(water_Volume):

                x = np.random.randint(0, xlen)
                y = np.random.randint(0, ylen)

                step = 0

                while(True):

                    # collect sediment
                    if sediment<size:
                        collect = collection_rate * (size - sediment)
                        collect2 = collect / 10
                        collect_count = 0

                        if water[x][y] == 0: collect_count += 2

                        # collect half as much from neigbors
                        if x>0:
                            if water[x-1][y] == 0:
                                data[x-1][y] -= collect2
                                collect_count+=1
                            if y>0:
                                if water[x - 1][y-1] == 0:
                                    data[x - 1][y-1] -= collect2
                                    collect_count+=1
                            if y<ylen-1:
                                if water[x - 1][y+1] == 0:
                                    data[x - 1][y+1] -= collect2
                                    collect_count+=1
                        if x<xlen-1:
                            if water[x + 1][y] == 0:
                                data[x+1][y] -= collect2
                                collect_count+=1
                            if y>0:
                                if water[x + 1][y-1] == 0:
                                    data[x + 1][y-1] -= collect2
                                    collect_count+=1
                            if y<ylen-1:
                                if water[x + 1][y+1] == 0:
                                    data[x + 1][y+1] -= collect2
                                    collect_count+=1
                        if y>0:
                            if water[x][y-1] == 0:
                                data[x][y-1] -= collect2
                                collect_count+=1
                        if y<ylen-1:
                            if water[x][y + 1] == 0:
                                data[x][y-1] -= collect2
                                collect_count += 1

                        sediment += collect_count * collect2
                        data[x][y] += collect_count * collect2

                    # step
                    (ax, ay, minVal) = getMinNbr(data, xlen, ylen, x, y, -sediment)
                    if ax < 0:
                        # if out of bounds exit loop

                        # the tile looses its sediment, water never becomes stable and is lost
                        data[x][y] -= sediment
                        break
                    if (x, y) == (ax, ay) or step>stepLen:
                        # if droplet is in a hole or ran out of movement,
                        # deposit all sediment (done implicitly)
                        # and stop water movement (done by adding size of water)
                        data[x][y] += size
                        water[x][y] += size

                        break

                    # update step
                    data[x][y] -= sediment
                    data[ax][ay] += sediment
                    (x, y) = (ax, ay)

                    step += 1

            # at the end of the rainfall, all the water dries and
            # leaves only ground elevation with new sediment
            data = np.subtract(data, water)

    return data

# Shade an elevation map according to gradient - helps to make the image look 3d
# code taken from this site: https://www.neonscience.org/resources/learning-hub/tutorials/create-hillshade-py
def hillshade(array, azimuth, angle_altitude):
    azimuth = 360.0 - azimuth

    x, y = np.gradient(array)
    slope = np.pi / 2. - np.arctan(np.sqrt(x * x + y * y))
    aspect = np.arctan2(-x, y)
    azm_rad = azimuth * np.pi / 180.  # azimuth in radians
    alt_rad = angle_altitude * np.pi / 180.  # altitude in radians

    shaded = np.sin(alt_rad) * np.sin(slope) + np.cos(alt_rad) * np.cos(slope) * np.cos((azm_rad - np.pi / 2.) - aspect)

    return (shaded + 1) / 2


# Given an array and its hillshade, it plots them over each other
def plot_hillshade(array, shaded_array):
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    im1 = plt.imshow(array, cmap='terrain')
    cbar = plt.colorbar()
    cbar.set_label('Elevation, m', rotation=270, labelpad=20)
    im2 = plt.imshow(shaded_array, cmap='Greys_r', alpha=0.8)  # plt.colorbar()
    plt.show()

# creates an elevation map with simple hills for testing use
def genSample(xlen, ylen):
    out = np.zeros((xlen,ylen))
    numHills = 30
    hillWidth = 30
    frac = .5/hillWidth
    for i in range(numHills):
        (x,y) = (np.random.randint(0,xlen),np.random.randint(0,ylen))
        for j in range(hillWidth):
            for k in range(hillWidth):
                d = hillWidth - np.sqrt(((k**2) + (j**2)))
                if d > 0:
                    if (x+j<xlen) and (y+k<ylen):
                        out[x+j][y+k] += frac * d
                    if j!= 0 and (x-j>=0):
                        if y+k<ylen:
                            out[x - j][y + k] += frac * d
                        if k!= 0 and (y-k>=0):
                            out[x - j][y - k] += frac * d
                    if k!= 0 and x+j<xlen and (y-k>=0):
                        out[x + j][y - k] += frac * d
    return out


if False:
    print("eroding")

    tem = dataGen.get_sample(256)#genSample(256,256)

    k = erode_Semi(tem, 100000)

    hill = hillshade(tem, 0, 30)
    hill2 = hillshade(k, 0, 30)

    plot_hillshade(np.concatenate((tem,k),axis=1),np.concatenate((hill,hill2),axis=1))

    out = np.concatenate((tem, k, np.subtract(k,tem)),axis = 1)
    plt.imshow(out,cmap='gist_earth')
    plt.show()

    out[0][0] = -1
    out[0][1] = 1
    plt.imshow(np.subtract(k,tem),cmap='gist_earth')
    plt.show()


