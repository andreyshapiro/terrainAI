import numpy as np
from tqdm import tqdm
import matplotlib.cm as mat1
import matplotlib.colors as mat2
import matplotlib.pyplot as plt
import handy_functions
import water_gen
import dataGen
import veg_gen
import bisect

aspect = handy_functions.unit_height / handy_functions.unit_length
h_val = 32
path_factor = 8
TIME_STEP = 10 #years

Building_Dict = {}
# id, [name, path-like, color]
L = [[0, ("Empty",0,(0,0,0))],
     [1, ("House",1,(.7,.7,1))], [3, ("Manner",1,(.73,.73,1))], [-1, ("Path", -1, (0.92, 0.67, 0.31))], [-10, ("Bridge", -2, (0.92, 0.67, 0.31))],
     [400, ("Meeting Spot", -1, (.7,1,.8))], [401, ("Bell", -1, (.7, 1, .8 ))], [402, ("Meeting Square", -1, (.7, 1, .8 ))],
     [403, ("Meeting House", 1, (.7, 1, 1))], [404, ("Amphitheater", -1, (.7, 1, .8 ))],
     [410, ("Well", 1, (.2,.2,1))],
     [200, ("Dock", -2, (0.82, 0.77, 0.15))]]
for i in range(len(L)):
    Building_Dict[L[i][0]] = []
    Building_Dict[L[i][0]].append(L[i][1])

# Overview:
# !!! 1) reason for settlement maps
# !!!      Trade: water, fish (lakes), game (trees),
# !!!      Agriculture: water, Farming (grass), Herding (shrub)
# !!! 2) settlement seeds:
# !!!      nucleated: flat, agriculture or trade -> high speciallization
# !!!                   growth in number of buildings/specilization
# !!!      disperced: agriculture, mountinous, -> low professionalism
# !!!                   growth in size of buildings
# !!!      linear: valley floors, river, water body, road. Agriculture
# !!! 3) initial cluster
# !!!      seed population, divide into houses, plant houses, and their mode of resource
#  4) types of buildings, events: https://knightstemplar.co/from-taverns-to-towers-a-glimpse-into-medieval-city-constructs/
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
# !!!      roads built with cluster, new road made every time a building is built not adjacent to one. Use A* algorithm
# !!! with harsh punishment for gradient.
#   Roads are paved by proximity to important places (may not worry about paving)
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

class travel_graph():
    def __init__(self, xlen, ylen, bottom_left):
        self.bottom_left = bottom_left
        self.shape = (xlen, ylen)  # this implicitly defines the nodes
        self.edgesLR = np.full((xlen-1,ylen),1.0)
        self.edgesDU = np.full((xlen,ylen-1),1.0)
        self.edges01 = np.full((xlen-1,ylen-1), 1.414)
        self.edges10 = np.full((xlen - 1, ylen - 1), 1.414)

        self.path_edgesLR = np.full((xlen - 1, ylen), False)
        self.path_edgesDU = np.full((xlen, ylen - 1), False)
        self.path_edges01 = np.full((xlen - 1, ylen - 1), False)
        self.path_edges10 = np.full((xlen - 1, ylen - 1), False)


    def get_edges(self, m, n):
        es = []
        xlen, ylen = self.shape
        if m>0:
            es.append((m-1,n,self.edgesLR[m-1][n]))
            if n>0:
                es.append((m-1,n-1, self.edges01[m-1][n-1]))
            if n<ylen-1:
                es.append((m - 1, n + 1, self.edges10[m - 1][n]))
        if m<xlen-1:
            es.append((m+1,n,self.edgesLR[m][n]))
            if n>0:
                es.append((m+1,n-1, self.edges10[m][n-1]))
            if n<ylen-1:
                es.append((m + 1, n + 1, self.edges10[m][n]))
        if n>0:
            es.append((m,n-1,self.edgesDU[m][n-1]))
        if n<ylen-1:
            es.append((m,n+1,self.edgesDU[m][n]))
        return es

    # takes in a LOCAL indexed path
    def update_path(self, path):
        (a,b) = path[0]
        for (x,y) in path[1:]:
            dx = x-a
            dy = y-b
            if dx == 0:
                corner = min(y,b)
                if not self.path_edgesDU[x][corner]:
                    self.path_edgesDU[x][corner] = True
                    self.edgesDU[x][corner] /= path_factor
            elif dy == 0:
                corner = min(x, a)
                if not self.path_edgesLR[corner][y]:
                    self.path_edgesLR[corner][y] = True
                    self.edgesLR[corner][y] /= path_factor
            elif dx-dy == 0:
                corner = min(x,a)
                corner2 = min(y,b)
                if not self.path_edges01[corner][corner2]:
                    self.path_edges01[corner][corner2] = True
                    self.edges01[corner][corner2] /= path_factor
            elif dx*dy == -1:
                corner = min(x, a)
                corner2 = min(y, b)
                if not self.path_edges10[corner][corner2]:
                    self.path_edges10[corner][corner2] = True
                    self.edges10[corner][corner2] /= path_factor

            (a,b) = (x,y)


# given a graph, and two points, it will find the shortest path between them and an updated graph.
def path_maker(graph, datain, x1, y1, x2, y2):
    # first we find the nodes the points correspond to
    (w, z) = graph.bottom_left
    (m1, n1)= (x1-w,y1-z)
    (m2, n2) = (x2-w,y2-z)

    height_goal = datain[x2][y2]

    # now we find the shortest path
    xlen, ylen = graph.shape
    max_v = xlen*ylen*256
    graphcopy = np.full((xlen,ylen,4), (max_v, 0, -1,-1)) #g, n/open/closed, parent_x, parent_y)
    print("finished initiating graphcopy")

    node_stack = [(0,m1,n1,0,0)]  # (f, x, y, g, h)

    print_val = 10
    print_val2 = 15
    closed = 0
    while 1:
        (f, m, n, g, _) = node_stack.pop(0)
        if g>print_val:
            print("g:", g, "h:", f-g)
            print(len(node_stack))
            print_val+= 500
        # value is always strictly improved so update graphcopy and add new directions
        if (m, n) == (m2, n2): break

        graphcopy[m][n][1] = 2 #close (m,n)
        closed += 1
        if closed>print_val2:
            print("closed", closed)
            print_val2 += 10000

        # then we add the new edges reached (rather, the nodes they reach at an improved rate, and the directions)
        es = graph.get_edges(m,n)
        for (e1,e2,c) in es:
            if graphcopy[e1][e2][1] == 2: continue # location is closed -> skip it.
            if g+c < graphcopy[e1][e2][0]:
                if graphcopy[e1][e2][1] == 1:  # if location open, then find and delete it before adding it in again
                    k = 0
                    while 1:
                        if node_stack[k][1:3] == (e1,e2):
                            node_stack.pop(k)
                            break
                        k+= 1
                graphcopy[e1][e2] = (g+c, 1, m, n)

                # calculate the heuristic: difference in altitudes + # of tiles
                h = np.sqrt(np.abs(e1-m2)**2 + np.abs(e2-n2)**2) + np.abs(datain[w+e1][z+e2] - height_goal) * aspect * h_val / path_factor

                # add in the new node, and sort according to h+g2
                bisect.insort(node_stack,(c+g+h,e1,e2,c+g,h))
        if not node_stack:
            print("out of options! Broke at", m,n)
            break

    print("while loop exited")
    # once done, retrace steps:
    outpath = [(m2,n2)]
    (_,_,x,y) = graphcopy[m2][n2]
    while (x,y) != (m1,n1) and (x,y) != (-1,-1):
        # add the tile to the path
        outpath.append((x,y))
        (_,_,x,y) = graphcopy[x][y]
    if (x,y) != (-1,-1):
        outpath.append((x,y))
    else: print("final destination not reached!")

    outpath = np.array(outpath) + (w,z)
    return (graphcopy[m2][n2][0], outpath) # a pair: value, directions


# given a graph, updates its edges
def graph_maker(graph, datain, structure, water):
    (a,b) = graph.bottom_left
    xlen, ylen = graph.shape
    max_v = xlen * ylen * 256
    for xi in range(xlen-1):
        x = a+xi
        for yj in range(ylen-1):
            y = b+yj
            # need to set any paths as such in the graph
            if Building_Dict[structure[x][y]][0][1]<0:
                if Building_Dict[structure[x+1][y]][0][1]:
                    graph.path_edgesLR[xi][yj] = True
                if Building_Dict[structure[x][y+1]][0][1]<0:
                    graph.path_edgesDU[xi][yj] = True
                if Building_Dict[structure[x+1][y+1]][0][1]<0:
                    graph.path_edges01[xi][yj] = True
            if Building_Dict[structure[x+1][y]][0][1] <0 and Building_Dict[structure[x][y+1]][0][1] < 0:
                graph.path_edges10[xi][yj] = True


            # if either contain water and no bridge: if the water is deep - impassible, if shallow then cost 100
            # if there is a proper building (structure value>0) - impassible
            if Building_Dict[structure[x][y]][0][1]>0 or (water[x][y] > 1 and Building_Dict[structure[x][y]][0][1]>=-1):
                graph.edgesLR[xi][yj] = max_v
                graph.edgesDU[xi][yj] = max_v
                graph.edges01[xi][yj] = max_v
            else:
                if Building_Dict[structure[x+1][y]][0][1]>0>0 or (water[x + 1][y] > 1 and Building_Dict[structure[x+1][y]][0][1]>=-1):
                    graph.edgesLR[xi][yj] = max_v
                elif (water[x + 1][y] > 0 and Building_Dict[structure[x+1][y]][0][1]>=-1) or (water[x][y] > 0  and Building_Dict[structure[x][y]][0][1]>=-1):
                    graph.edgesLR[xi][yj] = 250
                    if graph.path_edgesLR[xi][yj]: graph.edgesLR[xi][yj] /= path_factor
                else:
                    # no water or buildings: we measure height change 1+slope^2
                    graph.edgesLR[xi][yj] = (1 + np.abs(datain[x][y]-datain[x+1][y]) * aspect) ** h_val
                    if graph.path_edgesLR[xi][yj]: graph.edgesLR[xi][yj] /= path_factor
                if Building_Dict[structure[x][y+1]][0][1]>0 or (water[x][y+1] > 1 and structure[x][y+1]>=-1):
                    graph.edgesDU[xi][yj] = max_v
                elif (water[x][y+1] > 0 and structure[x][y+1]>=-1) or (water[x][y] > 0  and Building_Dict[structure[x][y+1]][0][1]>=-1):
                    graph.edgesDU[xi][yj] = 250
                    if graph.path_edgesDU[xi][yj]: graph.edgesDU[xi][yj] /= path_factor
                else:
                    # no water or buildings: we measure height change 1+slope^2
                    graph.edgesDU[xi][yj] = (1 + np.abs(datain[x][y] - datain[x][y+1]) * aspect) ** h_val
                    if graph.path_edgesDU[xi][yj]: graph.edgesDU[xi][yj] /= path_factor
                if Building_Dict[structure[x+1][y+1]][0][1]>0 or (water[x+1][y+1] > 1 and Building_Dict[structure[x+1][y+1]][0][1]>=-1):
                    graph.edges01[xi][yj] = max_v
                elif (water[x+1][y+1] > 0 and Building_Dict[structure[x+1][y+1]][0][1]>=-1) or (water[x][y] > 0 and Building_Dict[structure[x][y]][0][1]>=-1):
                    graph.edges01[xi][yj] = 250
                    if graph.path_edges01[xi][yj]: graph.edges01[xi][yj] /= path_factor
                else:
                    # no water or buildings: we measure height change 1+slope^2
                    graph.edges01[xi][yj] *= (1 + np.abs(datain[x][y] - datain[x+1][y+1]) * aspect) ** h_val
                    if graph.path_edges01[xi][yj]: graph.edges01[xi][yj] /= path_factor
            if (Building_Dict[structure[x][y+1]][0][1]>0 or Building_Dict[structure[x+1][y]][0][1]>0 or
                    (water[x][y+1] > 1 and Building_Dict[structure[x][y+1]][0][1]>=-1) or (water[x+1][y] > 1 and Building_Dict[structure[x+1][y]][0][1]>=-1)):
                graph.edges10[xi][yj] = max_v
            elif (water[x][y+1] > 0 and Building_Dict[structure[x][y+1]][0][1]>=-1) or (water[x+1][y] > 0  and Building_Dict[structure[x+1][y]][0][1]>=-1):
                graph.edges10[xi][yj] = 250
                if graph.path_edges10[xi][yj]: graph.edges10[xi][yj] /= path_factor
            else:
                # no water or buildings: we measure height change 1+slope^2
                graph.edges10[xi][yj] *= (1 + np.abs(datain[x][y+1] - datain[x+1][y]) * aspect) ** h_val
                if graph.path_edges10[xi][yj]: graph.edges10[xi][yj] /= path_factor

    for xi in range(xlen-1):
        x = a+xi
        y = b+ylen-1

        # again, need to update path flags.
        if (Building_Dict[structure[x][y]][0][1] <= -1) and (Building_Dict[structure[x+1][y]][0][1] <= -1):
            graph.path_edgesLR[xi][ylen-1] = True

        if (Building_Dict[structure[x][y]][0][1] > 0 or Building_Dict[structure[x+1][y]][0][1]>0 or
                (water[x][y] > 1 and Building_Dict[structure[x][y]][0][1] >= -1) or (water[x + 1][y] > 1 and Building_Dict[structure[x+1][y]][0][1]>=-1)):
            graph.edgesLR[xi][ylen-1] = max_v

        elif water[x+1][y] > 0 and Building_Dict[structure[x+1][y]][0][1]>=-1:
            graph.edgesLR[xi][ylen-1] = 250
            if graph.path_edgesLR[xi][ylen - 1]: graph.edgesLR[xi][ylen - 1] /= path_factor
        else:
            graph.edgesLR[xi][ylen-1] = (1 + np.abs(datain[x][y] - datain[x+1][y]) * aspect) ** h_val
            if graph.path_edgesLR[xi][ylen - 1]: graph.edgesLR[xi][ylen - 1] /= path_factor
    for yj in range(ylen -1):
        x = a + xlen -1
        y = b + yj

        # again, need to update path flags.
        if (Building_Dict[structure[x][y]][0][1] <= -1) and (Building_Dict[structure[x][y+1]][0][1] <= -1):
            graph.path_edgesDU[xlen-1][yj] = True

        if (Building_Dict[structure[x][y]][0][1] > 0 or Building_Dict[structure[x][y+1]][0][1] > 0 or
                (water[x][y] > 1 and Building_Dict[structure[x][y]][0][1] >= -1) or (water[x][y+1] > 1 and Building_Dict[structure[x][y+1]][0][1] >= -1)):
            graph.edgesDU[xlen-1][yj] = max_v
        elif water[x][y + 1] > 0 and Building_Dict[structure[x][y+1]][0][1] >= -1:
            graph.edgesDU[xlen-1][yj] = 250
            if graph.path_edgesDU[xlen-1][yj]: graph.edgesDU[xlen-1][yj] /= path_factor
        else:
            graph.edgesDU[xlen-1][yj] = (1 + np.abs(datain[x][y] - datain[x][y + 1]) * aspect) ** h_val
            if graph.path_edgesDU[xlen - 1][yj]: graph.edgesDU[xlen - 1][yj] /= path_factor
    return


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


# takes in coordinates, a struct map, and water map, and tells us if the given location is suitable for docks
def validator_Docks(xlen, ylen, struct, water, x, y):
    if water[x][y] > 0 and struct[x][y] == 0:
        nbrs = handy_functions.get_valid_nbrs(xlen, ylen, x, y)
        grnd = 0
        wtr = 0
        for (m, n) in nbrs:
            if water[m][n] > 0:
                wtr += 1
            else:
                grnd += 1
        if grnd > 1 and wtr > 4: return True
    return False


# takes in the struct list, the center, number of tiles to place, max radius to check for, and a function
# "validator" which takes in a coordinate and tells us if it's a suitable spot.
# Function returns a list a places to place the buildings
def building_placer(shape, center, no_tiles, max_tiles, validator):
    xlen, ylen = shape
    out = []
    (a,b) = center
    while no_tiles > 0:
        k = 1
        while k < max_tiles:
            for i in range(-k, k + 1):
                if a+i>=0 and a+i<xlen:
                    if b-k>=0:
                        if validator(a+i, b-k):
                            out.append((a+i,b-k))
                            k = max_tiles
                            break
                    if b+k<ylen:
                        if validator(a + i, b + k):
                            out.append((a + i, b + k))
                            k = max_tiles
                            break
                if b + i >= 0 and b + i < ylen:
                    if a-k >= 0:
                        if validator(a - k, b + i):
                            out.append((a - k, b + i))
                            k = max_tiles
                            break
                    if a+k < xlen:
                        if validator(a + k, b + i):
                            out.append((a + k, b + i))
                            k = max_tiles
                            break
            k += 1
        no_tiles -= 1
    return out


# takes in a village class, the structure map, path graph, relevant info (datain, water, tree, shrub, grass) and simulates a
# time_step, updating the village, structure map, and graph if needed.
def village_time_step(village, struct, graph, datain, water, tree, shrub, grass):
    # order: population, wealth growth, event occurance, determine needs and order, build up to available wealth
    # how to determine needs? any buildings destroyed by events? over population?
    #   trade specialization improvements
    #   number of people -> public works, defence, professions, authority
    return 1


# takes in a settlement evaluation map and gives the desired number of seeds for villages
# a seed contains: city center coords, population, # of houses, type (nucleated, linear, or dispersed),
# 4 industry ratings (farming, herding, hunting, fishing), 2 corresponding type ratings (agriculture vs trade), and
# starting radius.
def settlement_seeds(settlement_map, datain, water, slope, num):
    # sum sett_map for each square, then divide by 4 * ( 2000 / unit_length)^2 then roll against that number to get proposed
    # seed. Then in order from the highest score to the lowest set those as the seeds and then halving the score of any seed
    # within 2000 ft of the current seed. If we run out of proposed seeds, do the process again, ensuring new additions are also
    # halved if within 2000 ft of a pre-existing seed.

    xlen = len(settlement_map)
    ylen = len(settlement_map[0])
    shape = (xlen,ylen)

    rad_sq = (2000 / handy_functions.unit_length) ** 2

    # average of squares -> areas with one clear good industry are better than areas with a bit of everything
    sumsett = np.sum(settlement_map * settlement_map, axis=2) / ((500.0 / handy_functions.unit_length)**2)
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


    # now we will initialise the structure map and draw paths between and within villages
    struct = np.zeros((xlen, ylen))
    for v in villages:
        for b in v.building_list:
            for (x, y) in b.location:
                struct[x][y] = b.type

    # Note to self: need to plant primary income structures right away
    # farmland: find tiles with slope<0.15 within 500 ft of a river. 871200 sq ft * farming score -> rounded down
    # docks: find closest water tile (to center) with a shore and 5 neigboring water tiles. If fishing is primary, find 2

    for v in villages:
        (a,b) = v.center
        if v.A_T[0]>= v.A_T[1]:  # if agricultural, we generate farmland
            no_tiles = 871200 * v.industries[0] // (handy_functions.unit_length ** 2)


        else:  # if trade-based we create 0,1,or 2 docks
            if v.industries[3]>v.industries[2]: no_tiles = 2
            elif v.industries[3]> 0.1: no_tiles = 1
            else: continue

            # this will find a nearby suitable spot for a dock by going outward in concentric circles:
            # Note to self: currently this runs the risk of going out of bounds - we either need to implement a check or
            # require that village centers are close enough inside the boundaries that this isn't a problem. Also, the
            # max radius of 20 is arbitrary and needs to be changed later.
            docks = building_placer(shape, (a,b), no_tiles, 20, lambda i, j: validator_Docks(xlen, ylen, struct, water, i, j))
            for (i, j) in docks:
                struct[i][j] = 200  # Note to self, figure out what number id this should be
                v.building_list.append(building(i, j, 200))
            while False:#no_tiles>0:
                k  = 1
                while k<20:
                    for i in range(-k,k+1):
                        if water[a+i][b-k] > 0 and struct[a+i][b-k] == 0:
                            nbrs = handy_functions.get_valid_nbrs(xlen,ylen,a+i,b-k)
                            grnd = 0
                            wtr = 0
                            for (m,n) in nbrs:
                                if water[m][n] > 0: wtr+=1
                                else: grnd+= 1
                            if grnd>1 and wtr>4:
                                struct[a+i][b-k] = 200 # Note to self, figure out what number id this should be
                                v.building_list.append(building(a+i,b-k, 200))
                                k = 20
                                break
                        if water[a+i][b+k] > 0 and struct[a+i][b+k] == 0:
                            nbrs = handy_functions.get_valid_nbrs(xlen, ylen, a + i, b + k)
                            grnd = 0
                            wtr = 0
                            for (m, n) in nbrs:
                                if water[m][n] > 0:
                                    wtr += 1
                                else:
                                    grnd += 1
                            if grnd > 1 and wtr > 4:
                                struct[a + i][b + k] = 200  # Note to self, figure out what number id this should be
                                v.building_list.append(building(a + i, b + k, 200))
                                k = 20
                                break
                        if water[a-k][b+i] > 0 and struct[a-k][b+i] == 0:
                            nbrs = handy_functions.get_valid_nbrs(xlen, ylen, a - k, b + i)
                            grnd = 0
                            wtr = 0
                            for (m, n) in nbrs:
                                if water[m][n] > 0:
                                    wtr += 1
                                else:
                                    grnd += 1
                            if grnd > 1 and wtr > 4:
                                struct[a - k][b + i] = 200  # Note to self, figure out what number id this should be
                                v.building_list.append(building(a - k, b + i, 200))
                                k = 20
                                break
                        if water[a+k][b+i] > 0 and struct[a+k][b+i] == 0:
                            nbrs = handy_functions.get_valid_nbrs(xlen, ylen, a + k, b + i)
                            grnd = 0
                            wtr = 0
                            for (m, n) in nbrs:
                                if water[m][n] > 0:
                                    wtr += 1
                                else:
                                    grnd += 1
                            if grnd > 1 and wtr > 4:
                                struct[a + k][b + i] = 200  # Note to self, figure out what number id this should be
                                v.building_list.append(building(a + k, b + i, 200))
                                k = 20
                                break
                    k+=1
                no_tiles -= 1

    # TEMPORARY
    temp = False
    if temp:
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

    g = travel_graph(xlen, ylen, (0, 0))
    graph_maker(g, datain, struct, water)

    # within each village, connect all houses and structures to meeting square
    for v in villages:
        (v1,v2) = v.center
        for b in v.building_list:
            if b.type < 400:
                (x,y) = b.location[0]
                val, dir = path_maker(g, datain, x, y, v1, v2)
                print("got there at ", val)
                g.update_path(dir)
                for (a,b) in dir:
                    if struct[a][b] == 0: struct[a][b] = -1

    print("inner paths done")

    # Note to self: there is probably a better way to do this considering the fact that the number of villages is small
    # ideally we would do something like v1~v2, v1~v3, v2~v3, v1~v4, v2~v4, v3~v4, ...
    for i in range(len(villages)):
        (v1, v2) = villages[i].center
        for j in range(i+1, len(villages)):
            (v3,v4) = villages[j].center
            if np.sqrt((v1-v3)**2 + (v2-v4)**2) < 10000 / handy_functions.unit_length:
                val, dir = path_maker(g, datain, v1, v2, v3, v4)
                print("got there at ", val)
                g.update_path(dir)
                for (a, b) in dir:
                    if struct[a][b] == 0: struct[a][b] = -1

    print("outer paths done")


    return villages, struct, g  # returns the list of village seeds, the structure map, and the path graph


# a tool that takes in a struct and water map, along with a base image, and paints on it the water and structures
def struct_painter(img, struct, water):
    for x in range(len(struct)):
        for y in range(len(struct[0])):
            if struct[x][y] != 0:
                img[x][y] = Building_Dict[struct[x][y]][0][2]
            elif water[x][y] > 0: img[x][y] = (0, 0, 1)

# a temporary testing function for path-finding
def path_tester(len_in):
    datain = dataGen.get_sample_unnormed(len_in)
    s = water_gen.spring(datain, ((2048/len_in)**2) / ( 256 * 256), len_in, len_in)
    datain2 = np.copy(datain)
    water = water_gen.draw_water_erode2(datain2, s, 14)

    struct = np.zeros((len_in,len_in))

    g = travel_graph(len_in, len_in, (0,0))
    graph_maker(g, datain2, struct, water)

    v, dir = path_maker(g, datain, 50, 100, 500, 450)
    print("got there at ", v)

    (a,b) = g.bottom_left

    showim = handy_functions.hillshade(datain, 0, 30)
    showim3 = np.dstack((showim,showim,showim))

    for (x,y) in dir:
        showim3[x][y] = [0.92, 0.67, 0.31]

    for x in range(len_in):
        for y in range(len_in):
            if water[x][y]>0: showim3[x][y]=[0,0,1]

    plt.imshow(showim3)
    plt.show()

    g.update_path(dir + (a,b))

    v, dir = path_maker(g, datain, 50, 500, 500, 50)
    print("got there at ", v)

    for (x,y) in dir:
        showim3[x][y] = [0.92, 0.67, 0.31]

    plt.imshow(showim3)
    plt.show()

    g.update_path(dir + (a, b))


# a temporary testing function for building paths in and between villages
def village_path_tester(len_in):
    datain = dataGen.get_sample_unnormed(len_in)  # handy_functions.erode_Semi(handy_functions.genSample(256,256),10)
    s = water_gen.spring(datain, 24 / (256 * 256), len_in, len_in)
    datain2 = np.copy(datain)
    water = water_gen.draw_water_erode2(datain2, s, 24)

    veg, tree, shrub, grass, slope, water_dist = veg_gen.gen_veg(datain2, water)

    set_map = settlement_eval(datain2, slope, water, water_dist, tree, shrub, grass)

    final_seeds, struct, g = settlement_seeds(set_map, datain2, water, slope, 5)

    for v in final_seeds:
        print(v)
    showim = handy_functions.hillshade(datain2, 100, 30)
    showim3 = np.dstack((showim, showim, showim))

    struct_painter(showim3, struct, water)

    plt.imshow(showim3)
    plt.show()


village_path_tester(512)

if False:
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





            
            
