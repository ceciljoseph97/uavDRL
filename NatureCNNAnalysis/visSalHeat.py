import gym
import torch as th
import torch.nn as nn
import numpy as np
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torchinfo import summary
from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as vutils
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
import os
import argparse
from torchviz import make_dot
import torch.nn.functional as F

activations = {}

def get_activation(name):
    def hook(module, input, output):
        activations[name] = output.detach()
    return hook

def preprocess_image(image_path, input_type, transform=None):
    image = Image.open(image_path).convert('RGB')

    if input_type == 'depth_map':
        image = image.convert('L')
        image = image.resize((50, 50))
        image = np.array(image, dtype=np.float32) / 255.0
        image = th.tensor(image).unsqueeze(0).unsqueeze(0)
    elif input_type == 'rgb_image':
        image = transform(image).unsqueeze(0)
    elif input_type == 'depth_image':
        image = image.convert('L')
        image = image.resize((150, 50))
        image = np.array(image, dtype=np.float32) / 255.0
        image = th.tensor(image).unsqueeze(0).unsqueeze(0)
    else:
        raise ValueError(f"Unsupported input type: {input_type}")

    return image

def generate_saliency_map(model, input_tensor, device, input_image, output_file):
    input_tensor.requires_grad_()
    model.eval()
    model.zero_grad()
    output = model(input_tensor)
    score = output.max()
    score.backward()
    saliency = input_tensor.grad.abs().squeeze().cpu().detach().numpy()
    plt.figure(figsize=(10, 5))
    plt.suptitle("Input Image and Saliency Map", fontsize=16)

    plt.subplot(1, 2, 1)
    plt.title("Input Image")
    plt.imshow(np.array(input_image))
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title("Saliency Map")
    if saliency.ndim == 2:
        plt.imshow(saliency, cmap='coolwarm')
    elif saliency.ndim == 3:
        plt.imshow(np.max(saliency, axis=0), cmap='coolwarm')
    else:
        raise ValueError(f"Unexpected saliency map shape: {saliency.shape}")

    plt.axis('off')
    plt.savefig(f'{output_file}_saliency.png')
    plt.show()


def save_combined_activations(activations, output_file):
    resized_activations = []
    layer_names = ['conv1', 'conv2', 'conv3']

    target_shape = activations['conv1'][0].shape[-2:]

    for layer_name in layer_names:
        act = activations[layer_name][0].cpu()

        act = (act - act.min()) / (act.max() - act.min())

        resized_act = F.interpolate(
            act.unsqueeze(0), size=target_shape, mode="bilinear", align_corners=False
        ).squeeze(0)

        resized_activations.append((resized_act, layer_name))
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Activations from Convolutional Layers", fontsize=16)

    for idx, (resized_act, layer_name) in enumerate(resized_activations):
        grid = vutils.make_grid(resized_act.unsqueeze(1), nrow=8, normalize=True, scale_each=True)
        axes[idx].imshow(grid.permute(1, 2, 0).cpu().numpy())
        axes[idx].set_title(f"{layer_name.upper()} Activations")
        axes[idx].axis('off')

    plt.savefig(f'{output_file}_activations.png')
    plt.show()

def generate_heatmap(activations, input_image, output_file, cmap='jet'):
    target_shape = activations['conv1'][0].shape[-2:]

    resized_activations = []
    for layer_name in ['conv1', 'conv2', 'conv3']:
        act = activations[layer_name][0].cpu()
        act = (act - act.min()) / (act.max() - act.min())
        resized_act = F.interpolate(
            act.unsqueeze(0), size=target_shape, mode="bilinear", align_corners=False
        ).squeeze(0)

        resized_activations.append(resized_act)

    combined_act = th.cat(resized_activations, dim=0).mean(dim=0).numpy()
    heatmap_resized = F.interpolate(
        th.tensor(combined_act).unsqueeze(0).unsqueeze(0),
        size=input_image.size[::-1],
        mode='bilinear',
        align_corners=False
    ).squeeze().numpy()

    plt.figure(figsize=(10, 5))
    plt.suptitle("Heatmap Overlay", fontsize=16)
    plt.subplot(1, 2, 1)
    plt.title("Input Image")
    plt.imshow(input_image)
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.title("Heatmap Overlay")
    plt.imshow(input_image, alpha=0.5)
    plt.imshow(heatmap_resized, cmap=cmap, alpha=0.6)
    plt.colorbar(label="Activation Intensity")
    plt.axis('off')
    plt.savefig(f'{output_file}_heatmap.png')
    plt.show()



class NatureCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 512):
        super(NatureCNN, self).__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=2, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Flatten()
        )
        with th.no_grad():
            sample_input = th.as_tensor(observation_space.sample()[None]).float()
            n_flatten = self.cnn(sample_input).shape[1]
        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim), 
            nn.BatchNorm1d(features_dim),
            nn.ReLU()
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize CNN activations and saliency maps.')
    parser.add_argument('--model', type=str, required=True, help='Path to the trained model.')
    parser.add_argument('--image', type=str, help='Path to a single image file.')
    parser.add_argument('--input_type', type=str, required=True, choices=['depth_map', 'rgb_image', 'depth_image'], help='Type of input image.')
    parser.add_argument('--output_file', type=str, required=True, help='Base filename for saving visualizations.')

    args = parser.parse_args()

    if args.input_type == 'rgb_image':
        observation_shape = (3, 50, 50)
    elif args.input_type == 'depth_map':
        observation_shape = (1, 50, 50)
    elif args.input_type == 'depth_image':
        observation_shape = (1, 50, 150)
    else:
        raise ValueError("Invalid input type")

    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    observation_space = gym.spaces.Box(low=0, high=255, shape=observation_shape, dtype=np.float32)
    model = NatureCNN(observation_space, features_dim=512).to(device)
    summary(model, input_size=(1, *observation_space.shape))

    model.cnn[0].register_forward_hook(get_activation('conv1'))
    model.cnn[3].register_forward_hook(get_activation('conv2'))
    model.cnn[6].register_forward_hook(get_activation('conv3'))

    model.eval()
    writer = SummaryWriter('runs/model_visualization')

    transform = transforms.Compose([
        transforms.Resize((50, 50)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    if args.image:
        if os.path.isfile(args.image):
            input_image = Image.open(args.image)
            sample_input = preprocess_image(args.image, args.input_type, transform).to(device)

            generate_saliency_map(model, sample_input, device, input_image, args.output_file)
            save_combined_activations(activations, args.output_file)
            generate_heatmap(activations, input_image, args.output_file, cmap='jet')

        else:
            print(f"Image file {args.image} does not exist.")
    else:
        print("No image provided.")

