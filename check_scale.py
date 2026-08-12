import numpy as np

def read_ntu_skeleton(file_path):
    with open(file_path, 'r') as f:
        num_frame = int(f.readline())
        frames = []
        for _ in range(num_frame):
            num_body = int(f.readline())
            bodies = []
            for _ in range(num_body):
                f.readline()
                num_joint = int(f.readline())
                joints = [tuple(map(float, f.readline().split()[:3])) for _ in range(num_joint)]
                bodies.append(joints)
            frames.append(bodies)
    return np.array([b[0] for b in frames if len(b) > 0])

NTU_FROM_H3D = {
    0: 0, 1: 6, 2: 12, 3: 15, 4: 16, 5: 18, 6: 20, 7: 20,
    8: 17, 9: 19, 10: 21, 11: 21, 12: 1, 13: 4, 14: 7, 15: 10,
    16: 2, 17: 5, 18: 8, 19: 11, 20: 9, 21: 20, 22: 20, 23: 21, 24: 21,
}

real = read_ntu_skeleton('/srv/storage/stars@storage3.sophia.grid5000.fr/alakhlef/SK_zsl/Neuron/data/nturgbd_raw/nturgb+d_skeletons/S014C003P027R001A012.skeleton')
print("REEL   -- x:[{:.2f},{:.2f}] y:[{:.2f},{:.2f}] z:[{:.2f},{:.2f}]".format(
    real[:,:,0].min(), real[:,:,0].max(), real[:,:,1].min(), real[:,:,1].max(), real[:,:,2].min(), real[:,:,2].max()))

gen22 = np.load('/srv/storage/stars@storage3.sophia.grid5000.fr/alakhlef/SK_zsl/MotionGPT3/results/motgpt/debug--MoT_vae_trainemb_cls1_cfgrand_2optim_lr2_lr1/samples_2026-08-08-03-41-21/0_out.npy')
if gen22.ndim == 4:
    gen22 = gen22.squeeze(0)
gen25 = np.zeros((gen22.shape[0], 25, 3), dtype=np.float32)
for ntu_j, h3d_j in NTU_FROM_H3D.items():
    gen25[:, ntu_j, :] = gen22[:, h3d_j, :]
print("GENERE -- x:[{:.2f},{:.2f}] y:[{:.2f},{:.2f}] z:[{:.2f},{:.2f}]".format(
    gen25[:,:,0].min(), gen25[:,:,0].max(), gen25[:,:,1].min(), gen25[:,:,1].max(), gen25[:,:,2].min(), gen25[:,:,2].max()))
