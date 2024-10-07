This is a project that aims to help generate:
  1) Realistic Terrain using a diffusion model
  2) Post-process this terrain by generating water (with errosion)
  3) Generate Vegitation (using water and slope data)
  4) Plant and systematically grow Villages (using perviusly generated components). The villages include inter- and intra-connecting roads.
  5) The Villages will have a recorded history and will grow/shrink with time, giving each one it's own unique character determined by it's foundation, location, industry, population, wealth,
        event history, and growth history


To use: run final_product. If you want to save the generated data for reuse later, there are files (to-load1 and to-load3) which are meant for that. If you want to later use this data, simply indicate this
when running final_product and tell it which file you want to load from.

**If you want to use real data as opposed to model-generated data (currently this is the only option):** you need to add a folder "data_in" and to it download "in.tif". This should be a 1 degree x 1 degree
elevation map with 1/3 arcsecond resolution.

**how to download in.tif:**
  1) go to https://apps.nationalmap.gov/downloader/
  2) under "data" select "Elevation Products (3DEP)"
  3) select "1/3 arc-second DEM" and click "show" to see coverage.
  4) At the top, under "area of interest" select "Extent" and on the map drag a square around an area you're interested in.
  5) Click on the blue "Search Products" button. This will generate all the 1 degree x 1 degree maps that intersect the area you've selected. Mousing over them will show on the map their footprint.
  6) Find the one you want and click "Download Link (TIF)" and save in the "data_in" folder as "in.tif" - it may take a while, it's about 400 MB.
