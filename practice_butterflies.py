from huggingface_hub import login
from diffusers import DiffusionPipeline, DDIMScheduler
import matplotlib.pyplot as plt
import numpy as np
from diffusers import DDPMPipeline
from huggingface_hub import get_full_repo_name
from huggingface_hub import HfApi, create_repo
from osgeo import gdal
import handy_functions

model_id = "elevdata-v1"

login('')

hub_model_id = get_full_repo_name(model_id)

stable = 0

stepNUM = 1
outNumC = 1
outNumR = 1
outNum= outNumC * outNumR


if True:
    scheduler = DDIMScheduler.from_pretrained(f'andreyshapiro/{model_id}')
    scheduler.set_timesteps(num_inference_steps=stepNUM)

    pipeline = DiffusionPipeline.from_pretrained(
        f'andreyshapiro/{model_id}', scheduler=scheduler)

    images = []
    for i in range(outNum):
        images.append(pipeline(num_inference_steps=stepNUM, output_type= np.array)[0][0])



    k = images[0]

    tem = []
    for i in range(outNumC):
        tem.append(images[i])
        for j in range(outNumR-1):
            tem[i]=np.concatenate((tem[i],images[(i*outNumR)+j+1]))

    for i in range(outNumC-1):
        tem[0] = np.concatenate((tem[0], tem[i+1]), axis=1)


    if stable == 1:
        tem[0][0][0] = 0
        tem[0][0][1] = 1

    print("semi")

    temp = np.reshape(tem[0], (len(tem[0]),len(tem[0][0])))

#temp = handy_functions.genSample(256,256)

k = handy_functions.erode_Semi(temp, 10)

out = np.concatenate((temp, k, np.subtract(k,temp)),axis = 1)
plt.imshow(out,cmap='gist_earth')
plt.show()

plt.imshow(np.subtract(k,temp),cmap='gist_earth')
plt.show()

#plt.imshow(tem[0],cmap='gist_earth')
#plt.show()
#for i in range(outNum):
#    plt.subplot(2,2,i+1)    # the number of images in the grid
#    plt.axis("off")
#    plt.imshow(images[i],cmap='gist_earth')
#plt.tight_layout()
#plt.show()

print("eroding")

k = handy_functions.erode_Random(temp, 1000000)

plt.imshow(np.concatenate((temp, k, np.subtract(k,temp)),axis = 1),cmap='gist_earth')
plt.show()

#plt.imshow(k, cmap='gist_earth')
#plt.show()


#plt.imshow(np.subtract(tem[0],k), cmap='gist_earth')
#plt.show()



