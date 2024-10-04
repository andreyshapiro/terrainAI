import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import handy_functions
import dataGen
import water_gen
import veg_gen
import settlement


# first we ask if they want to load data or generate new data (for now this does nothing, we just assume it's [g])

while 1:
    i = input("Load [l] or Generate [g]?")
    if i == 'l' or i == 'g':
        break
    else: print("please input l or g")

save = False
k = input("Automatic Saving? [y] or [n]")
if k == "y":
    save = True
    targetfolder = input("where would you like to save to?")
    dataGen.foldername_saving = targetfolder


# elev, water, fert_maps, veg, set_map, village, all
skip = 0
if i == 'l':
    filepath = input("where would you like to load from?")
    dataGen.foldername_loading = filepath
    i = input("up to what stage would you like to load data? \n "
              "[e] - elevation \n [w] - water \n [f] - fertility_maps (includes water distance and slope) \n [veg] - vegetation"
              "\n [s] - set_map \n [vil] - village_seeds (includes struct and path graph) \n [a] - ALL")
    if i=='e':
        skip = 1
    elif i=='w':
        skip = 2
    elif i=='f':
        skip = 3
    elif i=='veg':
        skip = 4
    elif i=='s':
        skip = 5
    elif i=='vil':
        skip = 6
    elif i=='a':
        skip = 100

if skip<=0:
    # second we would ask user to specify parameters or use default
    # (geographical area, size, aridity, grass, shrub, tree, num of villages, village event factors)
    # for now most of this is skipped

    size = 512
    i = input("size? (512 default)")
    if i.isdigit(): size = int(i)
    else: print("Invalid size input, using default")

    if size>= 4096: print("This will take a very long time, proceed at your own risk")
    elif size>=2048: print("Warning, this size input may take a while, particularly in the settlement_eval stage")

if skip<1:
    # next, we ask them to choose if they want to use real data or AI generated. For now, we assume it's real data.
    gen_type = 1
    i = input("Use Real [r] or Model-generated terrain [m]?")
    if i == 'm': gen_type = 0

    data = []
    while 1:
        if gen_type:
            data = dataGen.get_sample_unnormed(size)
        else:
            data = dataGen.get_sample_unnormed(size)  # to be replaced with a call to the model
        hill = handy_functions.hillshade(data, 0, 30)
        handy_functions.plot_hillshade(data,hill)
        i = input("happy with elevation map? [y] or [n]")
        if i == 'y': break
        else: print("trying again")

    if save:
        dataGen.save_elev(data)
else:
    data = dataGen.load_elev()
    size = len(data)

if skip<2:
    # now we generate the water
    water = []
    while 1:
        data2 = np.copy(data)
        factor = 1
        i = input("input spring factor (higher means more springs - default 1)")
        if i.isdigit(): factor = int(i)
        springs = water_gen.spring(data2, factor / (256 * 256), size, size)
        water = water_gen.draw_water_erode2(data2, springs, 14)
        plt.imshow(water,cmap='gist_earth')
        plt.show()
        i = input("happy with water generation? [y] or [n]")
        if i == 'y':
            break
        else:
            print("trying again")
    data = data2

    if save:
        dataGen.save_water(water)
else:
    water = dataGen.load_water()

if skip<3:
    # next we generate fertility maps
    tree, shrub, grass, slope, water_data = veg_gen.create_fertility(data,water)

    print("Presenting resulting fertility maps")
    plt.imshow(np.concatenate((tree, shrub, grass), axis=1), cmap='Greens')
    plt.show()

    print("Presenting resulting water_distance map")
    plt.imshow(np.sqrt(water_data), cmap='Oranges')
    plt.show()

    if save:
        dataGen.save_slope(slope, water_data)
        dataGen.save_fert_maps(grass, shrub, tree)
else:
    grass, shrub, tree = dataGen.load_fert_maps()
    slope, water_data = dataGen.load_slope()

if skip<4:
    # next, generate vegetation map
    veg = []
    while 1:
        veg = veg_gen.gen_veg_with_pregen(data, tree, shrub, grass, slope, water_data)
        plt.imshow(veg)
        plt.show()
        i = input("happy with veg generation (red = grass, green = trees, blue = shrubs)? [y] or [n]")
        if i == 'y':
            break
        else:
            print("trying again")

    if save:
        dataGen.save_veg(veg)
else:
    veg = dataGen.load_veg()

if skip<5:
    # next generate the settlement_map
    set_map = settlement.settlement_eval(data, slope, water, water_data, veg, grass)

    print("Presenting resulting settlement_maps \n Left Hand Side: red = farming, green = game, blue = fishing      | "
          "Right Hand Side: red = farming, green = herding, blue = game")
    plt.imshow(np.concatenate((np.delete(set_map, 1, 2), np.delete(set_map, 3, 2)), axis=1))
    plt.show()

    if save:
        dataGen.save_set_map(set_map)
else:
    set_map = dataGen.load_set_map()

if skip<6:
    # next we generate village seeds (along with path graph and struct)
    while 1:
        village_num = 6
        i = input("number of villages? (6 default)")
        if i.isdigit():
            village_num = int(i)
        else:
            print("Invalid village_num input, using default")

        village_seeds, struct, graph = settlement.settlement_seeds(set_map, data, water, water_data, slope, village_num)
        for v in village_seeds:
            print(v)

        showim = handy_functions.hillshade(data, 0, 30)
        showim3 = np.dstack((showim, showim, showim))

        settlement.struct_painter(showim3, struct, water)

        plt.imshow(showim3)
        plt.show()

        i = input("happy with village generation? [y] or [n]")
        if i == 'y':
            break
        else:
            print("trying again")
    if save:
        dataGen.save_structs(struct, village_seeds, graph)
else:
    struct, village_seeds, graph = dataGen.load_structs()









