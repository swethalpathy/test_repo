import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
from sklearn.manifold import TSNE


# Importing train and test data
#train = pd.read_csv(r"C:\Users\91944\Downloads\neo4j-community-3.5.4\import\Titanic\train_kmeans.csv")


# t-SNE
def t_sne(xx, yy):
    #yy = train['Survived']
    #xx = train.drop(columns=['Survived'])
    tsne = TSNE(random_state=123).fit_transform(xx)

    # choose a color palette with seaborn.
    num_classes = len(np.unique(yy))
    palette = np.array(sns.color_palette("hls", num_classes))

    # create a scatter plot.
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    sc = plt.scatter(tsne[:, 0], tsne[:, 1], lw=0, s=40, c=palette[yy.astype(np.int)])
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    plt.axis('off')
    plt.axis('tight')

    # add the labels for each digit corresponding to the label
    txts = []
    for i in range(num_classes):
        # Position of each label at median of data points.

        xtext, ytext = np.median(tsne[yy == i, :], axis=0)
        txt = plt.text(xtext, ytext, str(i), fontsize=24)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])
        txts.append(txt)
    plt.show()







