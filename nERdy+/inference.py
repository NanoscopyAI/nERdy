import torch
from torchvision import transforms

import os
from PIL import Image

from model import D4nERdy
from postprocessing import postprocessing

import matplotlib.pyplot as plt


home = os.path.expanduser('~')

sted_data_prefix = '/path/to/sted-data'

# Number of samples per group
groups = {'climp': 9, 'control': 14, 'rtn': 12}

in_channels = 1
out_channels = 1

model = D4nERdy(in_channels, out_channels).cuda()
model.load_state_dict(torch.load('path/to/your/model.pth'))

# Uncomment the following line in case using the model from this repository
# model.load_state_dict(torch.load('nERdy+/NNet_groupy_p4m_v2_VecAdam.pth'))


def get_prob_map(group, frame):
    """
    Get the probability map for a given group and frame.

    Args:
        group (str): The group name.
        frame (int): The frame number.

    Returns:
        prob_map (numpy.ndarray): The probability map.
    """
    
    # Adjust path to your image
    # Test with climp12_er_mean.png from the repository.
    imgpath = f'{group}{frame}_er_mean.png'

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

    # prob_map = process.postprocessing(norm)
    prob_map = postprocessing(norm)

    return prob_map

# get the probability map per input in each group
def prob_map_runner():
    """
    Generates probability maps for each input in each group.

    Returns:
        prob_map_data (dict): A dictionary containing the probability maps for each input in each group.
    """
    prob_map_data = {}

    # Iterate over each group
    for group in groups:
        # Iterate over each frame in the group
        for frame in range(1, groups[group]+1):
            # Get the probability map for the current group and frame
            prob_map = get_prob_map(group, frame)

            # Store the probability map in the dictionary
            prob_map_data[f'{group}{frame}'] = prob_map

    return prob_map_data

# Visualize the probability map for climp12
def visualize_prob_map():
    """
    Visualize the probability map for climp12.
    """
    prob_map = get_prob_map('climp', 12)
    plt.imshow(prob_map, cmap='gray')
    plt.show()

# visualize_prob_map()
