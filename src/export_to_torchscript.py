import os
import pickle
import zlib

import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt

import depth_pro
from depth_pro.depth_pro import DEFAULT_MONODEPTH_CONFIG_DICT as config


class Wrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model.forward(x * 2 - 1)[0]


image_path = "../data/example.jpg"

img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)

# Load model and preprocessing transform
config.checkpoint_uri = "../checkpoints/depth_pro.pt"
model, transform = depth_pro.create_model_and_transforms(config)
model.eval()
wrapper = Wrapper(model)
img_size = (model.img_size,) * 2
image = transform(cv2.resize(img, img_size))[None] * .5 + .5

# Run inference.
with torch.inference_mode():
    depth = wrapper(image)

depth_np = depth[0, 0].cpu().numpy()

traced_model = torch.jit.trace(wrapper, image)

exported_path = os.path.splitext(config.checkpoint_uri)[0] + '_jit.pt'
dumped_model = traced_model.save_to_buffer()
b = {
    'model': dumped_model,
    'torch': torch.__version__,
    'img_size': tuple(image.shape[2:4]),
}
dump = pickle.dumps(b, protocol=pickle.HIGHEST_PROTOCOL)
compressed = zlib.compress(dump)

with open(exported_path, 'wb') as f:
    f.write(compressed)

with torch.inference_mode():
    depth_jit = traced_model(image)

depth_np_jit = depth_jit[0, 0].cpu().numpy()

errors = np.abs(depth_np - depth_np_jit)
max_error = errors.max()
mean_error = errors.mean()
std_error = errors.std()
print(f"Max error: {max_error}, mean error: {mean_error}, std error: {std_error}")

dpi = 80
for figimage in [depth_np, depth_np_jit, image[0].cpu().permute(1, 2, 0).numpy()]:
    fig = plt.figure(figsize=(depth_np.shape[1] / dpi, depth_np.shape[0] / dpi), dpi=dpi)
    fig.figimage(figimage)
    plt.show()
