import matplotlib.pyplot as plt
import numpy as np

def plot_generation_with_input(
    input_samples, 
    outputs,
    save_path: str = None
    ):

    fig, ax = plt.subplots(len(input_samples), 2)

    plt.axis('off')

    for i in range(len(input_samples)):
        ax[i][0].imshow(input_samples[i].permute(1, 2, 0).cpu().detach().numpy())
        ax[i][1].imshow(outputs[i].permute(1, 2, 0).cpu().detach().numpy())
        ax[i][0].axis("off")
        ax[i][1].axis("off")

    ax[0][0].set_title("Input")
    ax[0][1].set_title("Output")

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

def plot_generation_with_title(
    sample_list: list, 
    title_list: list,
    save_path: str = None):

    fig, ax = plt.subplots(int(len(sample_list)//3)+1, 3)

    plt.axis('off')

    for i in range(len(sample_list)):
        ax[i//3][i%3].imshow(sample_list[i].permute(1, 2, 0).cpu().detach().numpy())
        ax[i//3][i%3].set_title(title_list[i])
        ax[i//3][i%3].axis("off")

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

def plot_causal_adjacency_matrix(
    adjacency_matrix
    ):
    fig, ax = plt.subplots(figsize=(6, 5))
    cax = ax.imshow(adjacency_matrix, cmap='viridis')
    fig.colorbar(cax)

    nrows, ncols = adjacency_matrix.shape
    for i in range(nrows):
        for j in range(ncols):
            ax.text(j, i, f'{adjacency_matrix[i, j]:.2f}',
                    ha='center', va='center', color='white', fontsize=12)

    ax.set_title("Causal Adjacency Matrix")
    ax.set_xticks(np.arange(ncols))
    ax.set_yticks(np.arange(nrows))
    ax.set_xticklabels([f'C^{j}' for j in range(ncols)])
    ax.set_yticklabels([f'C^{i}' for i in range(nrows)])
    plt.show()



