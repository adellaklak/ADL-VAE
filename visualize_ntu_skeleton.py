import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import PillowWriter

def read_ntu_skeleton(file_path):
    with open(file_path, 'r') as f:
        num_frame = int(f.readline())
        frames = []
        for _ in range(num_frame):
            num_body = int(f.readline())
            bodies = []
            for _ in range(num_body):
                f.readline()  # body info line (bodyID, tracking flags, etc.)
                num_joint = int(f.readline())
                joints = []
                for _ in range(num_joint):
                    vals = f.readline().split()
                    x, y, z = float(vals[0]), float(vals[1]), float(vals[2])
                    joints.append((x, y, z))
                bodies.append(joints)
            frames.append(bodies)
    return frames

# Connexions standard du squelette NTU 25-joints (indices 0-based)
BONES = [(0,1),(1,20),(2,20),(3,2),(4,20),(5,4),(6,5),(7,6),(8,20),(9,8),
         (10,9),(11,10),(12,0),(13,12),(14,13),(15,14),(16,0),(17,16),
         (18,17),(19,18),(21,22),(22,7),(23,24),(24,11)]

def main(skeleton_path, out_gif):
    frames = read_ntu_skeleton(skeleton_path)
    # garde uniquement le corps 0 sur les frames où il est présent
    seq = np.array([f[0] for f in frames if len(f) > 0])
    print(f"Frames chargées: {seq.shape}")

    fig = plt.figure(figsize=(5,5))
    ax = fig.add_subplot(111, projection='3d')

    def update(i):
        ax.cla()
        pts = seq[i]
        xs, ys, zs = pts[:,0], pts[:,1], pts[:,2]
        ax.scatter(xs, zs, ys, c='red', s=15)
        for a, b in BONES:
            ax.plot([xs[a], xs[b]], [zs[a], zs[b]], [ys[a], ys[b]], c='blue')
        ax.set_xlim(seq[:,:,0].min(), seq[:,:,0].max())
        ax.set_ylim(seq[:,:,2].min(), seq[:,:,2].max())
        ax.set_zlim(seq[:,:,1].min(), seq[:,:,1].max())
        ax.set_title(f"frame {i}/{len(seq)}")

    from matplotlib.animation import FuncAnimation
    anim = FuncAnimation(fig, update, frames=len(seq), interval=50)
    anim.save(out_gif, writer=PillowWriter(fps=15))
    print(f"Sauvegardé: {out_gif}")

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
