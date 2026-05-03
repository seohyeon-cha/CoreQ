import torch
import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
from mpl_toolkits.mplot3d import Axes3D  # needed for 3D plots


def plot_delta_x_2d_3d(
    per_channel_mae_per_layer,  # list of tensors [hidden_size]
    layer_mae,                  # list of floats
    layer_fro,                  # list of floats
    method_name: str,
    output_path_2d: str,
    output_path_3d: str,
):
    """
    per_channel_mae_per_layer: list length = num_layers,
        each element is a 1D tensor of shape [hidden_size] with MAE per channel.
    layer_mae: list of scalar MAE per layer.
    layer_fro: list of scalar Frobenius norm per layer.
    """

    num_layers = len(per_channel_mae_per_layer)
    hidden_size = per_channel_mae_per_layer[0].numel()

    # ---------- 2D plot: MAE and Frobenius norm per layer ----------
    layers = np.arange(num_layers)

    plt.figure(figsize=(8, 5))
    plt.plot(layers, layer_mae, marker='o', label='MAE(X_f, X_q)', linewidth=2, markersize=5)
    plt.plot(layers, layer_fro, marker='s', label='Frobenius norm ||X_f - X_q||_F', linewidth=2, markersize=5)
    plt.xlabel('Layer index', fontsize=14)
    plt.ylabel('Error', fontsize=14)
    plt.title(f'Activation error vs. layer ({method_name})', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tick_params(labelsize=12)
    plt.tight_layout()
    
    # Create directory if it doesn't exist
    dir_path = os.path.dirname(output_path_2d)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)
    
    plt.savefig(output_path_2d, dpi=300)
    plt.close()

    # ---------- 3D plot: per-channel MAE surface ----------
    # Stack to [num_layers, hidden_size]
    # Convert to float32 before numpy conversion (handles BFloat16)
    mae_matrix = torch.stack(per_channel_mae_per_layer, dim=0).float().numpy()

    L, C = mae_matrix.shape
    X, Y = np.meshgrid(np.arange(L), np.arange(C))  # X: layer, Y: channel

    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(
        X, Y, mae_matrix.T,  # transpose so that Y-axis is channel
        rstride=1, cstride=1, linewidth=0, antialiased=True
    )
    ax.set_xlabel('Layer index', fontsize=14)
    ax.set_ylabel('Channel index', fontsize=14)
    ax.set_zlabel('MAE |X_f - X_q|', fontsize=14)
    ax.set_title(f'Per-channel MAE surface ({method_name})', fontsize=16, fontweight='bold')
    ax.tick_params(labelsize=11)
    plt.tight_layout()
    
    # Create directory if it doesn't exist
    dir_path = os.path.dirname(output_path_3d)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)
    
    plt.savefig(output_path_3d, dpi=300)
    plt.close()


def compute_and_save_layer_norms(layer_fro_list, save_path: str, layer_mae_list=None):
    """
    Save Frobenius norms and MAE per layer to a pickle file.
    This can be used later to compare different alphas / methods on a single plot.
    
    Args:
        layer_fro_list: List of Frobenius norms per layer
        save_path: Path to save the pickle file
        layer_mae_list: Optional list of MAE per layer
    """
    data = {
        'layer_fro': layer_fro_list,
    }
    if layer_mae_list is not None:
        data['layer_mae'] = layer_mae_list
    with open(save_path, 'wb') as f:
        pickle.dump(data, f)


def plot_module_mae(module_mae, method_name: str, output_path_heatmap: str, output_path_lines: str):
    """
    Plot module-wise MAE heatmap and line plots.
    
    Args:
        module_mae: Dictionary mapping module names to lists of MAE values per layer.
                   e.g., {'self_attn.q_proj': [0.01, 0.02, ...], ...}
        method_name: Name of the quantization method
        output_path_heatmap: Path to save the heatmap plot
        output_path_lines: Path to save the line plot
    """
    modules = list(module_mae.keys())
    num_modules = len(modules)
    num_layers = len(module_mae[modules[0]])

    mae_mat = np.zeros((num_modules, num_layers))
    for i, m in enumerate(modules):
        mae_mat[i] = np.array(module_mae[m])

    # Heatmap of MAE for each module and layer
    plt.figure(figsize=(8, 6))
    plt.imshow(mae_mat, aspect='auto', origin='lower')
    cbar = plt.colorbar(label='MAE |X_f - X_q|')
    cbar.ax.tick_params(labelsize=16)
    cbar.set_label('MAE |X_f - X_q|', fontsize=16)
    plt.xticks(np.arange(num_layers), np.arange(num_layers), fontsize=16)
    plt.yticks(np.arange(num_modules), modules, fontsize=16)
    plt.xlabel('Layer index', fontsize=16)
    plt.ylabel('Module', fontsize=16)
    plt.title(f'Module-wise MAE heatmap ({method_name})', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Create directory if it doesn't exist
    dir_path = os.path.dirname(output_path_heatmap)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)
    
    plt.savefig(output_path_heatmap, dpi=300)
    plt.close()

    # Module-wise MAE vs layer
    plt.figure(figsize=(9, 5))
    layers = np.arange(num_layers)

    for m in modules:
        plt.plot(layers, module_mae[m], marker='o', label=m, linewidth=2, markersize=5)

    plt.xlabel('Layer index', fontsize=16)
    plt.ylabel('MAE |X_f - X_q|', fontsize=16)
    plt.title(f'Module-wise MAE vs layer ({method_name})', fontsize=16, fontweight='bold')
    plt.legend(fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tick_params(labelsize=16)
    plt.tight_layout()
    
    # Create directory if it doesn't exist
    dir_path = os.path.dirname(output_path_lines)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)
    
    plt.savefig(output_path_lines, dpi=300)
    plt.close()


def plot_unified_mae(mae_data_dict, output_path: str, title: str = "Unified MAE comparison across alphas",
                     calibration_data_dict=None):
    """Plot MAE from multiple runs (different α values) on a single unified plot.

    Args:
        mae_data_dict: Dictionary mapping labels to lists of MAE per layer.
                      e.g., ``{'alpha=0.25': [0.01, 0.02, ...]}``.
        output_path: Path to save the unified plot.
        title: Title for the plot.
        calibration_data_dict: Optional dictionary (deprecated, kept for compatibility).
    """
    plt.figure(figsize=(8, 6))
    layers = None

    def sort_key(label):
        if label.startswith('alpha='):
            try:
                return (0, float(label.split('=')[1]))
            except ValueError:
                return (0, float('inf'))
        return (1, label)

    sorted_labels = sorted(mae_data_dict.keys(), key=sort_key)
    
    # Plot MAE for each label
    for label in sorted_labels:
        mae_list = mae_data_dict[label]
        if layers is None:
            layers = np.arange(len(mae_list))
        plt.plot(layers, mae_list, marker='o', linestyle='-', linewidth=2.5, markersize=6,
                label=label)
    
    plt.xlabel('Layer index', fontsize=16)
    plt.ylabel('MAE |X_f - X_q|', fontsize=16)
    plt.title(title, fontsize=16, fontweight='bold')
    plt.legend(fontsize=12, loc='best', ncol=2)
    plt.grid(True, alpha=0.3)
    plt.tick_params(labelsize=16)
    plt.tight_layout()
    
    # Create directory if it doesn't exist
    dir_path = os.path.dirname(output_path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)
    
    plt.savefig(output_path, dpi=300)
    plt.close()


def load_mae_from_pickle(pickle_path: str):
    """
    Load MAE data from a pickle file.
    
    Args:
        pickle_path: Path to the pickle file
        
    Returns:
        Dictionary with 'layer_mae' key containing list of MAE per layer
    """
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)
    return data
