import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import PillowWriter, FuncAnimation

# Chaine cinematique standard HumanML3D/SMPL, 22 joints (0-indexed)
# jambe droite / jambe gauche / colonne+tete / bras droit / bras gauche
BONES = [(0,2),(2,5),(5,8),(8,11),
         (0,1),(1,4),(4,7),(7,10),
         (0,3),(3,6),(6,9),(9,12),(12,15),
         (9,14),(14,17),(17,19),(19,21),
         (9,13),(13,16),(16,18),(18,20)]

def main(npy_path, out_gif):
    seq = np.load(npy_path)
    if seq.ndim == 4:
        seq = seq.squeeze(0)
    print(f"Frames chargees: {seq.shape}")

    fig = plt.figure(figsize=(5,5))
    ax = fig.add_subplot(111, projection='3d')

    def update(i):
        ax.cla()
        pts = seq[i]
        xs, ys, zs = pts[:,0], pts[:,1], pts[:,2]
        ax.scatter(xs, zs, ys, c='green', s=15)
        for a, b in BONES:
            ax.plot([xs[a], xs[b]], [zs[a], zs[b]], [ys[a], ys[b]], c='darkgreen')
        ax.set_xlim(seq[:,:,0].min(), seq[:,:,0].max())
        ax.set_ylim(seq[:,:,2].min(), seq[:,:,2].max())
        ax.set_zlim(seq[:,:,1].min(), seq[:,:,1].max())
        ax.set_title(f"frame {i}/{len(seq)}")

    anim = FuncAnimation(fig, update, frames=len(seq), interval=50)
    anim.save(out_gif, writer=PillowWriter(fps=15))
    print(f"Sauvegarde: {out_gif}")

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
