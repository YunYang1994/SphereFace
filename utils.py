import imageio
import os
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt

def visualize(embedding, label, epoch=0, acc=0., picname=''):

    batch_size, embedding_dim = embedding.shape
    if embedding_dim == 2:
        """
        visualize embedding in 2D
        """
        fig,ax = plt.subplots()
        X, Y = embedding[:,0], embedding[:,1]
        ax.set_xlim(X.min(), X.max())
        ax.set_ylim(Y.min(), Y.max())
        for x,y,l in zip(X,Y,label):
            c = cm.rainbow(int(255 *l/ 9))
            ax.text(x, y, l, color=c)
            # plt.plot(x,y, '.', c=c)
            plt.title("epoch: %2d   accuracy: %.4f" %(epoch+1, acc))
        plt.axis('off')
        plt.legend()
        plt.tight_layout()
        plt.savefig(picname)

    if embedding_dim == 3:
        """
        visualize embedding in 3D
        """
        fig = plt.figure(); ax = Axes3D(fig)
        X, Y, Z = embedding[:, 0], embedding[:, 1], embedding[:, 2]
        for x, y, z, s in zip(X, Y, Z, label):
            c = cm.rainbow(int(255*s/9)); ax.text(x, y, z, s, color=c)
            # c = cm.rainbow(int(255*s/9)); ax.text(x, y, z, s, backgroundcolor=c)
        ax.set_xlim(X.min(), X.max()); ax.set_ylim(Y.min(), Y.max()); ax.set_zlim(Z.min(), Z.max())
        plt.title("accuracy: %.4f" %acc)
        plt.legend()
        plt.tight_layout()
        plt.savefig(picname)

def create_gif(gif_name, path, duration = 0.2):
    '''
    creat GIF
    '''

    frames = []
    pngFiles = os.listdir(path)
    image_list = [os.path.join(path, str(i)+'.jpg') for i in range(len(pngFiles))]
    for image_name in image_list:
        frames.append(imageio.imread(image_name))
    imageio.mimsave(gif_name, frames, 'GIF', duration = duration)

    return

if __name__ == "__main__":
    # path = './image/1/'
    path = './image/2/'
    # gif_name = './image/modified_softmax_loss.gif'
    gif_name = './image/angular_softmax_loss.gif'
    create_gif(gif_name, path)




