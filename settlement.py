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
#           precious metal smith, pottery/clayworks, glass works, stables/horses, messenger system, fletcher
#       primary: farm, dock, warf, herding wall, canal, barn, granary, warehouse,fishmonger,
#       public works: well, library, market square, fountain, hospital/almshouse, bridge, ferry
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

class village_seed():
    def __init__(self,x,y):
        self.center = (x,y)
        self.radius = 500 // handy_functions.unit_length
        self.population = 25
        self.wealth = 25
        self.spent = 0
        self.building_list = []
        self.families_list = []
        self.roads_list = []
        self.type = 1 # 3 is dispersed
        self.industries = [0,0,0,0]  # farming, herding, game, fishing
        self.A_T = [0,0]  # Agriculture and Trade

    def __str__(self):
        p = ((str(self.center) + " " + str(self.radius) + " Type: "+ str(self.type) +'\n[Population = ' +
              str(self.population) + ", Wealth: " + str(self.wealth) + "]\n") + str(self.A_T) + str(self.industries) +
             "\n Families: " + str(len(self.families_list))) + ", Buildings: " + str(len(self.building_list))
        return p
class family():
    def __init__(self, num):
        self.size = num

class building():
    def __init__(self, x, y, type):
        self.location =  [(x,y)]
        self.type = type
        self.size = 1
        self.cost = 5
        # 1-99:Residence; 100-199:Authority; 200-299:Specialization; 300-399:Primary; 400-499:Public Works; 500+:Defence
        # 1: House
        # 400: Meeting Spot
    def collide(self, x, y):
        for (a,b) in self.location:
            if (a,b) == (x,y): return True
        return False
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

# takes in a settlement evaluation map and gives the desired number of seeds for villages
# a seed contains: city center coords, population, # of houses, type (nucleated, linear, or dispersed),
# 4 industry ratings (farming, herding, hunting, fishing), 2 corresponding type ratings (agriculture vs trade), and
# starting radius.
def settlement_seeds(settlement_map, water, slope, num):
    # sum sett_map for each square, then divide by 4 * ( 2000 / unit_length)^2 then roll against that number to get proposed
    # seed. Then in order from the highest score to the lowest set those as the seeds and then halving the score of any seed
    # within 2000 ft of the current seed. If we run out of proposed seeds, do the process again, ensuring new additions are also
    # halved if within 2000 ft of a pre-existing seed.

    xlen = len(settlement_map)
    ylen = len(settlement_map[0])

    rad_sq = (2000 / handy_functions.unit_length) ** 2

    # average of squares -> areas with one clear good industry are better than areas with a bit of everything
    sumsett = np.sum(settlement_map * settlement_map, axis=2) / ((1000.0 / handy_functions.unit_length)**2)
    final_seeds = []

    seeds = []
    while len(final_seeds)<num:
        if seeds:
            seeds.sort()
            (ka,xa,ya) = seeds.pop()

            final_seeds.append((ka,xa,ya) )
            for i in range(len(seeds)):
                (k,x,y) = seeds[i]
                # Note to self: this could be more dynamic. Maybe too close = bad,
                # but close enough is good (being near other villages)
                dist_sq = (x-xa)**2 + (y-ya)**2
                if dist_sq < rad_sq:
                    seeds[i] = (k/4,x,y)
                #elif dist_sq < rad_sq * 49:
                #    seeds[i] = (k * 2,x,y)
            # for every seed we add, remove one two from the bottom
            if seeds: seeds.pop(0)
            if seeds: seeds.pop(0)
        else:
            print("preparing ", num, " seeds. ", len(final_seeds), " done.")
            for x in tqdm(range(xlen)):
                for y in range(ylen):
                    if np.random.random() < sumsett[x][y]: seeds.append((sumsett[x][y], x, y))
            # checking against pre-defined seeds
            for j in range(len(final_seeds)):
                (ka, xa, ya) = final_seeds[j]
                for i in range(len(seeds)):
                    (k, x, y) = seeds[i]
                    if x==xa and y==ya:
                        seeds.pop(i)
                    elif (x-xa)**2 + (y-ya)**2< rad_sq:
                        seeds[i] = (k/2,x,y)
            print(len(seeds), " potential seeds.")


    villages = []
    for (k,x,y) in final_seeds:
        v = village_seed(x,y)
        v.industries = settlement_map[x][y]
        v.A_T = [v.industries[0]**2 + v.industries[1]**2, v.industries[2]**2 + v.industries[3]**2]
        if v.A_T[0] > v.A_T[1]: v.type = 3
        v.radius *= v.type
        v.wealth *= 1+v.A_T[1]
        v.population *= 1+v.A_T[0]
        v.spent = 25  # 5 houses

        p = v.population
        families = []
        for i in [5,4,3,2,1]:
            pp = p // i
            p -= pp
            families.append(family(pp))
        v.families_list = families

        buildings = [building(x,y,400)]  # Meeting square
        for i in [5,4,3,2,1]:
            while 1:
                bx = np.random.randint(x-v.radius//2,x+v.radius//2)
                by = np.random.randint(y-v.radius//2,y+v.radius//2)
                if bx<0 or by<0 or bx>=xlen or by>=ylen or water[bx][by]>0 or slope[bx][by]>0.68: continue
                t = 0
                for b in buildings:
                    if b.collide(bx,by):
                        t=1
                        break
                if t==0: break
            buildings.append(building(bx,by,1))

        # Note to self: implement road-building here

        budget = v.wealth - v.spent
        if budget>=6:
            buildings[0].type = 404  # Amphitheatre
            v.spent += 6
        elif budget>=4:
            buildings[0].type = 403  # Meeting House
            v.spent += 4
        elif budget>=3:
            buildings[0].type = 402  # Meeting Square
            v.spent += 3
        elif budget>=1:
            buildings[0].type = 401  # Bell
            v.spent += 1
        v.building_list = buildings
        villages.append(v)


    # TEMPORARY
    radCOLOR = [.5,.5,.5,.5]
    homeCOLOR = [.8,.8,.8,.8]
    meetCOLOR = [.5,1,1,1]
    for v in villages:
        (x,y) = v.center
        sphere = handy_functions.create_sphere(v.radius)
        for (i,j) in sphere:
            xi = x+i
            yj = y+j
            if xi>=0 and yj>=0 and xi<xlen and yj<ylen and water[xi][yj]<=0: settlement_map[xi][yj] *= radCOLOR

        for b in v.building_list:
            if b.type >= 400: col = meetCOLOR
            else: col = homeCOLOR
            for (a,b) in b.location:
                settlement_map[a][b] = col

    return villages


len_in = 2048
datain = dataGen.get_sample_unnormed(len_in)  # handy_functions.erode_Semi(handy_functions.genSample(256,256),10)

s = water_gen.spring(datain, 1 / (256 * 256), len_in, len_in)
datain2 = np.copy(datain)
water = water_gen.draw_water_erode2(datain2, s, 14)

veg, tree, shrub, grass, slope, water_dist = veg_gen.gen_veg(datain2, water)

set_map = settlement_eval(datain2, slope, water, water_dist, tree, shrub, grass)

final_seeds = settlement_seeds(set_map, water, slope, 6)

for v in final_seeds:
    print(v)

plt.imshow(slope, cmap='Greys')
plt.show()

plt.imshow(np.sqrt(water_dist), cmap='Oranges')
plt.show()

plt.imshow(np.concatenate((tree, shrub, grass), axis=1), cmap='Greens')
plt.show()

# second_row = np.concatenate((veg, np.subtract(veg,water*10)), axis = 1)
plt.imshow(water, cmap='gist_earth')  # np.concatenate((first_row, second_row), axis =0),cmap='gist_earth')
plt.show()

plt.imshow(veg)  # cmap='Dark2_r')
plt.show()



plt.imshow(np.delete(set_map, 1, 2))  # cmap='Dark2_r')
plt.show()

plt.imshow(np.delete(set_map, 3, 2))  # cmap='Dark2_r')
plt.show()





            
            
