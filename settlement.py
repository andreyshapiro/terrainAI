import numpy as np
from tqdm import tqdm
import matplotlib.cm as mat1
import matplotlib.colors as mat2
import matplotlib.pyplot as plt
import handy_functions
import water_gen
import dataGen
import veg_gen

# Overview:
# 1) reason for settlement maps
#       Trade: water, fish (lakes), game (trees),
#       Agriculture: water, Farming (grass), Herding (shrub)
# 2) settlement seeds:
#       nucleated: flat, agriculture or trade -> high speciallization
#                    growth in number of buildings/specilization
#       disperced: agriculture, mountinous, -> low professionalism
#                    growth in size of buildings
#       linear: valley floors, river, water body, road. Agriculture
# 3) initial cluster
#       seed population, divide into houses, plant houses and their mode of resource
# 4) types of buildings, events: https://knightstemplar.co/from-taverns-to-towers-a-glimpse-into-medieval-city-constructs/
#       residence: house, large farm house, townhouse, mannor
#       authority: temple, town hall, constabulary, armory, guild houses
#       specialization: tavern, theater (simple stage, amphetheater, enclosed theater), mill (water and wind),
#           blacksmith, cobler, leatherworks, bakeries, textiles, apothocary, herbalist, carpetner, mason,
#           precius metal smith, pottery/clayworks, glass works, stables/horses, messenger system, fletcher
#       primary: farm, dock, warf, herding wall, canal, barn, granary, warehouse,fishmonger,
#       public works: well, library, market square, fountain, hospital/almhouse
#       defence: walls (wood, stone, sizes), mound, tower, gate, moat
#
#     events: plague, famine, drought, flood, fire, harsh winter, strong storms, rain, invation, occupation,
# 5) growth, building, road, population
#       populations grows steadily, events drop it
#       buildings grow with population along with them come specializations, events destroy buildings, other buildings
# destroy buildings.
#       roads built with cluster, new road made every time a building is built not adjacent to one. Use dextra algorithm
# with harsh punishment for gradient. Roads are paved by proximity to important places (may not wory about paving)
#


def settlement_eval(elev, slope, water, water_dist, tree, shrub, grass):
    xlen = len(elev)
    ylen = len(elev[0])

    sphere_radius = 1000 // handy_functions.unit_length  # 1000 ft radius
    building_radius = sphere_radius // 8

    water_factor = np.zeros((xlen,ylen,1))
    slope_factor = np.zeros((xlen,ylen,1))
    fish = np.zeros((xlen,ylen))

    building_sphere = handy_functions.create_sphere(building_radius)
    print("evaluating settlement factors")

    # here we find the average of slopes in a smaller radius to use for building
    slope_stack = []
    slope_stackMain = []

    cnt_Max = 2 ** 26 // (xlen * ylen)  # a safe memory limit for my device, after which we need to sum the stack
    cnt = 0
    broken = 0
    for (i, j) in tqdm(building_sphere):
        if i > 0:
            if j > 0:
                slope_stack.append(np.pad(slope, ((i, 0), (j, 0)), mode='edge')[:-i, :-j])
            else:
                slope_stack.append(np.pad(slope, ((i, 0), (0, -j)), mode='edge')[:-i, -j:])
        else:
            if j > 0:
                slope_stack.append(np.pad(slope, ((0, -i), (j, 0)), mode='edge')[-i:, :-j])
            else:
                slope_stack.append(np.pad(slope, ((0, -i), (0, -j)), mode='edge')[-i:, -j:])
        cnt += 1
        if cnt > cnt_Max:
            slope_stackMain.append(np.sum(slope_stack, axis=0))
            stack = []
            broken += 1
            print("slope stack broken", broken)
            cnt = 0
    if slope_stack: slope_stackMain.append(np.sum(slope_stack, axis=0))
    slope_building = np.sum(slope_stackMain, axis=0) / len(building_sphere)

    w_max = handy_functions.water_max
    w_max_q = w_max//4
    water_dist2 = np.sqrt(water_dist)

    water_thresh = [w_max_q, w_max_q*2, w_max_q*3, w_max]
    slope_thresh = np.array([5/180,15/180,25/180,35/180]) * np.pi
    fish_thresh = np.array([20, 10, 5, 1]) / handy_functions.unit_height

    for x in tqdm(range(xlen)):
        for y in range(ylen):

            w = water_dist2[x][y]
            s = slope_building[x][y]

            if w==0:
                water_factor[x][y] = [0.0]
                fish[x][y] = 1.0
                ww = water[x][y]
                for t in fish_thresh:
                    if ww>=t:
                        break
                    else:
                        fish[x][y]/=2
            else:
                water_factor[x][y] = [1.0]
                for t in water_thresh:
                    if w<t:
                        break
                    else:
                        water_factor[x][y] /= 1.35
            if s>slope_thresh[3]:
                slope_factor[x][y] = [0.0]
            else:
                slope_factor[x][y] = [1.0]
                for t in slope_thresh:
                    if s<t:
                        break
                    else:
                        slope_factor[x][y]/=2

    sphere_radius = 1000 // handy_functions.unit_length  # 1000 ft radius
    sphere = handy_functions.create_sphere(sphere_radius)
    print("summing settlement factors; sphere size:", len(sphere))


    stack = []
    stackMain = []
    grand_array = np.dstack((grass,shrub,tree,fish))

    cnt_Max = 2**26 // (xlen*ylen) # a safe memory limit for my device, after which we need to sum the stack
    cnt = 0
    broken = 0
    for (i,j) in tqdm(sphere):
        if i>0:
            if j>0:
                stack.append(np.pad(grand_array, ((i,0),(j,0),(0,0)), mode='edge')[:-i,:-j,:])
            else:
                stack.append(np.pad(grand_array, ((i, 0), (0, -j), (0, 0)), mode='edge')[:-i, -j:, :])
        else:
            if j>0:
                stack.append(np.pad(grand_array, ((0, -i), (j, 0), (0, 0)), mode='edge')[-i:, :-j, :])
            else:
                stack.append(np.pad(grand_array, ((0, -i), (0, -j), (0, 0)), mode='edge')[-i:, -j:, :])
        cnt+=1
        if cnt>cnt_Max:
            stackMain.append(np.sum(stack, axis = 0))
            stack = []
            broken += 1
            print("stack broken", broken)
            cnt=0
    if stack: stackMain.append(np.sum(stack, axis = 0))
    suitability_map = np.sum(stackMain, axis = 0)
    suitability_map[:,:,3] *= np.sqrt(suitability_map[:,:,3])
    suitability_map[:, :, 2] = np.power(suitability_map[:, :, 2], 7/8)
    suitability_map[:, :, 1] /= 2 #np.sqrt(suitability_map[:, :, 1]) * sphere_radius
    suitability_map[:, :, 0] *= 1.25 # np.sqrt(suitability_map[:, :, 0]) * sphere_radius
    suitability_map *= water_factor * slope_factor / len(sphere)

    return suitability_map



datain = dataGen.get_sample_unnormed(2048)  # handy_functions.erode_Semi(handy_functions.genSample(256,256),10)

s = water_gen.spring(datain, 1 / (256 * 256), 2048, 2048)
datain2 = np.copy(datain)
water = water_gen.draw_water_erode2(datain2, s, 14)

veg, tree, shrub, grass, slope, water_dist = veg_gen.gen_veg(datain2, water)

set_map = settlement_eval(datain2, slope, water, water_dist, tree, shrub, grass)


plt.imshow(slope, cmap='Greys')
plt.show()

plt.imshow(np.sqrt(water_dist), cmap='Oranges')
plt.show()

plt.imshow(np.concatenate((tree, shrub, grass), axis=1), cmap='Greens')
plt.show()

# second_row = np.concatenate((veg, np.subtract(veg,water*10)), axis = 1)
plt.imshow(water, cmap='gist_earth')  # np.concatenate((first_row, second_row), axis =0),cmap='gist_earth')
plt.show()

plt.imshow(veg*[.50,1.0,1.0])  # cmap='Dark2_r')
plt.show()



plt.imshow(np.delete(set_map, 1, 2))  # cmap='Dark2_r')
plt.show()

plt.imshow(np.delete(set_map, 3, 2))  # cmap='Dark2_r')
plt.show()





            
            
