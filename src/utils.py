import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import torch.nn as nn
import seaborn as sns

def generate_plots(list_of_dirs, legend_names, save_path):
    """ Generate plots according to log 
    :param list_of_dirs: List of paths to log directories
    :param legend_names: List of legend names
    :param save_path: Path to save the figs
    """
    assert len(list_of_dirs) == len(legend_names), "Names and log directories must have same length"
    data = {}
    for logdir, name in zip(list_of_dirs, legend_names):
        json_path = os.path.join(logdir, 'results.json')
        assert os.path.exists(os.path.join(logdir, 'results.json')), f"No json file in {logdir}"
        with open(json_path, 'r') as f:
            data[name] = json.load(f)
    
    for yaxis in ['train_accs', 'valid_accs', 'train_losses', 'valid_losses']:
        fig, ax = plt.subplots()
        for name in data:
            ax.plot(data[name][yaxis], label=name)
        ax.legend()
        ax.set_xlabel('epochs')
        ax.set_ylabel(yaxis.replace('_', ' '))
        fig.savefig(os.path.join(save_path, f'{yaxis}.png'))

def plot_gradient_flow(grad_dict, title, save_path):
    # Process layer names and data
    layers = [k.replace('.weight', '').replace('.bias', '') for k in grad_dict.keys()]
    num_epochs = len(next(iter(grad_dict.values())))
    data = np.array([grad_dict[k] for k in grad_dict.keys()])
    
    # Compute layer-wise means and standard deviations
    means = [np.mean(grad_dict[k]) for k in grad_dict.keys()]
    stds = [np.std(grad_dict[k]) for k in grad_dict.keys()]
    
    # Create a figure with 1 row and 2 columns, setting the relative widths
    fig, axes = plt.subplots(
        1, 2, 
        figsize=(16, len(layers) * 0.4 + 3),
        gridspec_kw={'width_ratios': [70, 30]}
    )
    
    # Plot heatmap on the left subplot (remove the colorbar/sidebar with cbar=False)
    ax0 = axes[0]
    sns.heatmap(
        data, 
        ax=ax0, 
        xticklabels=range(1, num_epochs+1), 
        yticklabels=layers, 
        cmap='viridis', 
        cbar=False
    )
    ax0.set_xlabel('Epoch')
    ax0.set_title('Gradient Flow Heatmap')
    # Ensure y-axis tick labels appear on the left side (default)
    ax0.yaxis.tick_left()
    ax0.yaxis.set_label_position("left")
    
    # Plot horizontal bar plot on the right subplot
    ax1 = axes[1]
    y_positions = np.arange(len(layers))
    ax1.barh(y_positions, means, xerr=stds, capsize=5, color='skyblue')
    ax1.set_yticks(y_positions)
    # Remove y-axis tick labels from the layer-wise plot
    ax1.set_yticklabels([])
    ax1.invert_yaxis()  # Display the first layer at the top
    ax1.set_xlabel('Gradient Norm')
    ax1.set_title('Layer-wise Gradient Flow')
    
    # Remove extra top and bottom margins:
    bar_height = 0.8  # Default bar height in matplotlib barh
    ax1.set_ylim((len(layers) - 1) + bar_height / 2, 0 - bar_height / 2)
    
    # Set the overall title for the combined plot
    fig.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_path)
    plt.close()


def generate_gradient_flow_plots(list_of_dirs, legend_names):
    assert len(list_of_dirs) == len(legend_names), "Names and log directories must have same length"
    
    for logdir, name in zip(list_of_dirs, legend_names):
        json_path = os.path.join(logdir, 'results.json')
        assert os.path.exists(json_path), f"No results.json file in {logdir}"
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Assume that the json contains a key 'train_grad_norms' that is a dictionary
        # mapping layer names to gradient norm lists.
        grad_norms = data['train_grad_norms']
        
        # Split the gradients into weights and biases, excluding some layers from the process.
        weight_grads = {k: v for k, v in grad_norms.items() 
                        if 'weight' in k and 'bias' not in k and 'bn' not in k and 'norm' not in k}
        bias_grads   = {k: v for k, v in grad_norms.items() 
                        if 'bias' in k and 'bn' not in k and 'norm' not in k}
        
        # Generate combined plot for weights.
        weight_combined_title = f"Gradient Flow of Weights for {name}"
        weight_combined_save = os.path.join(logdir, f"gradient_flow_weight.png")
        plot_gradient_flow(weight_grads, weight_combined_title, weight_combined_save)
        
        # Generate combined plot for biases.
        bias_combined_title = f"Gradient Flow of Bias for {name}"
        bias_combined_save = os.path.join(logdir, f"gradient_flow_bias.png")
        plot_gradient_flow(bias_grads, bias_combined_title, bias_combined_save)


def seed_experiment(seed):
    """Seed the pseudorandom number generator, for repeatability.

    Args:
        seed (int): random seed
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        # Prefer deterministic CuDNN behavior for repeatability.
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

def to_device(tensors, device):
    if isinstance(tensors, torch.Tensor):
        return tensors.to(device=device)
    elif isinstance(tensors, dict):
        return dict(
            (key, to_device(tensor, device)) for (key, tensor) in tensors.items()
        )
    elif isinstance(tensors, list):
        return list(
            (to_device(tensors[0], device), to_device(tensors[1], device)))
    else:
        raise NotImplementedError("Unknown type {0}".format(type(tensors)))

def cross_entropy_loss(logits: torch.Tensor, labels: torch.Tensor):
    """ Return the mean loss for this batch
    :param logits: [batch_size, num_class]
    :param labels: [batch_size]
    :return loss 
    """
    logits = logits - logits.max(dim=1, keepdim=True)[0]
    exps = torch.exp(logits).sum(dim=1, keepdim=True)
    probs = logits - torch.log(exps)
    losses = -probs[torch.arange(logits.shape[0]), labels]
    return losses.mean()

def compute_accuracy(logits: torch.Tensor, labels: torch.Tensor):
    """ Compute the accuracy of the batch """
    acc = (logits.argmax(dim=1) == labels).float().mean()
    return acc

def save_model(model: nn.Module, logdir: str) -> None:
    """ Save the model's state dictionary """
    os.makedirs(logdir, exist_ok=True)
    save_path = os.path.join(logdir, "model.pth")
    torch.save(model, save_path)        
    print(f"Model saved to {save_path}")

def load_model(logdir: str, map_location=None) -> nn.Module:
    """ Load the model's state dictionary """
    load_path = os.path.join(logdir, "model.pth")
    # If map_location is None, will load onto the devices originally saved from. 
    model = torch.load(load_path, map_location=map_location, weights_only=False)
    print(f"Model loaded from {load_path}")
    return model