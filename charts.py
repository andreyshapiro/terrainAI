Events = {}
# id, [name,]
e = [[0, "Nothing"],
     [1, "Plague"],  # kills 5-25% of the population --- (population/city radius) will contribute, public works decrease
     [2, "Drought"],  # kills % according to investment: Fi*35 + Hi*15 + Gi*20 --- public works mitigate, so does v.W
     [3, "Flood"],    # destroy all buildings 40-300 ft from water --- suitability_map[:,:,4] contributes to this
     [4, "Fire"],     # start at a random building, spread with a prob to nearby buildings (dep on distance and type)
     [5, "Harsh Winter"],  # kills
     [6, "Strong Storm"],  # kills % according to investment: Fi*10 + Si*10, removed double the % of wealth --- public works decrease
     [7, "Earthquake"],  # 5-15% of buildings destroyed. For each 1-5 population dies.
     [8, "Invasion"],  # kills 5-15% of the population, removes 5-25% of wealth. --- defence buildings prevent
     [9, "Occupation"],  # removes 15% of wealth --- an invasion has to have occurred
     [10, "Riot"],  # kills 5%, removes 5% of wealth. --- authority buildings decrease.
     [11, "Famine"]] #

for i in range(len(e)):
    Events[e[i][0]] = []
    Events[e[i][0]].append(e[i][1])

# id, probability
Base_Events = [(1, .1), (2, .12), (3, .05), (4, .12), (5, .15), (6, .03), (7, .05), (8, .05), (9, 0), (10, .05), (11, .08)]