from huggingface_hub import login
from diffusers import DiffusionPipeline, DDIMScheduler
import matplotlib.pyplot as plt
import numpy as np
from diffusers import DDPMPipeline
from huggingface_hub import get_full_repo_name
from huggingface_hub import HfApi, create_repo
from osgeo import gdal
import elev_to_topo

model_id = "elevdata-v1"

#enter login here login('')

hub_model_id = get_full_repo_name(model_id)

stepNUM = 100

scheduler = DDIMScheduler.from_pretrained(f'andreyshapiro/{model_id}')
scheduler.set_timesteps(num_inference_steps=stepNUM)

pipeline = DiffusionPipeline.from_pretrained(
    f'andreyshapiro/{model_id}', scheduler=scheduler)

images = pipeline(num_inference_steps=stepNUM, output_type= np.array)


k = images[0][0]

plt.imshow(images[0][0], cmap='gist_earth')
plt.show()


elev_to_topo.convo(k)

plt.imshow(k, cmap='gist_earth')
plt.show()



