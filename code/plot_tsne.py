import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
from MulticoreTSNE import MulticoreTSNE as TSNE


def plot_tsne(train_images_view, plot_name, label_1, label_2, epoch):
    train_images_view = train_images_view.detach().numpy()
    embeddings = TSNE(n_jobs=2).fit_transform(train_images_view.reshape(-1, 784))
    x = embeddings[:, 0]
    y = embeddings[:, 1]
    x = pd.DataFrame(x, columns=['x'])
    y = pd.DataFrame(y, columns=['y'])

    number_of_data = int(x.shape[0] / 2)

    x.reset_index(drop=True, inplace=True)
    y.reset_index(drop=True, inplace=True)

    plt.scatter(x.iloc[0:number_of_data], y.iloc[0:number_of_data], c="red", label=label_1)
    plt.scatter(x.iloc[number_of_data:], y.iloc[number_of_data:], c="green", label=label_2)
    plt.legend()
    #plt.show()
    plt.savefig(f'plots/{plot_name}_{str(epoch)}.png')
    plt.close()
