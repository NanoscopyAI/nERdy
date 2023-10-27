import contextlib
import itertools
import torch
from torchvision import transforms

import os
import imageio
from PIL import Image


from model import NerdyNet
from postprocessing import PostProcessing


home = os.path.expanduser('~')

sted_data_prefix = f'{home}/MIAL/data/sted-data/vess_enh_unet'

groups = ['climp', 'control', 'rtn']

in_channels = 1
out_channels = 1


model = NerdyNet(in_channels, out_channels)
process = PostProcessing()
model.load_state_dict(torch.load('/localhome/asa420/unet_oct17.pth'))

def get_prob_map():
    imgpath = f'{sted_data_prefix}/{group}/images/sted_{group}{i}_er_mean.png'

    image = Image.open(imgpath)
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    image = transform(image)
    image = image.unsqueeze(0)  # Add batch dimension

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
