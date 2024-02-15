import contextlib
import itertools
import torch
from torchvision import transforms

import os
import imageio
from PIL import Image


from .model import D4nERdy
from .postprocessing import PostProcessing


home = os.path.expanduser('~')

sted_data_prefix = '/path/to/sted-data'

# Number of samples per group
groups = {'climp':9, 'control':14, 'rtn':12}

in_channels = 1
out_channels = 1

model = D4nERdy(in_channels, out_channels)
process = PostProcessing()
model.load_state_dict(torch.load('path/to/your/model.pth'))

def get_prob_map():

    # Adjust path to your image
    imgpath = 'climp12_er_mean.png'

    image = Image.open(imgpath)
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    image = transform(image)

    # Add batch dimension
    image = image.unsqueeze(0)

    # Forward pass through the model
    model.eval()
    with torch.no_grad():
        output = model(image.cuda())

    # Convert the output to probabilities by applying the sigmoid activation
    output_probs = torch.sigmoid(output)

    # Convert tensors to numpy arrays for visualization
    image_np = image.squeeze().numpy()
    output_probs_np = output_probs.cpu().squeeze().numpy()

    norm = (output_probs_np - output_probs_np.min()) / (output_probs_np.max() - output_probs_np.min())

    prob_map = process.postprocessing(norm)

    return prob_map

# get the probability map per input in each group
def prob_map_runner():
    prob_map_data = {}
    for group in groups:
        for frame in range(1, groups[group]+1):
            prob_map = get_prob_map(group, frame)
            prob_map_data[f'{group}{frame}'] = prob_map
    return prob_map_data