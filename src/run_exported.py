import io
import pickle
import zlib

import cv2
import torch
from matplotlib import pyplot as plt

exported_path = "../checkpoints/depth_pro_jit.pt"

print('Loading model...')
with open(exported_path, 'rb') as f:
    data = pickle.loads(zlib.decompress(f.read()))

image_path = "../data/example.jpg"

img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)

print('Initializing model...')
model = torch.jit.load(io.BytesIO(data.pop('model')), 'cuda').half().eval()
img_size = data.pop('img_size')
image = torch.from_numpy(cv2.resize(img, img_size[::-1]).transpose(2, 0, 1)).cuda().half()[None] / 255.

print('Running inference...')
with torch.inference_mode():
    depth = model(image)

depth_np = depth[0, 0].cpu().float().numpy()

print('Plotting results...')
dpi = 100
for figimage in [depth_np, image[0].cpu().float().permute(1, 2, 0).numpy()]:
    fig = plt.figure(figsize=(depth_np.shape[1] / dpi, depth_np.shape[0] / dpi), dpi=dpi)
    fig.figimage(figimage)
    plt.show()
print('Done.')
