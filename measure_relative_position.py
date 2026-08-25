import numpy as np

d = np.load('/srv/storage/stars@storage3.sophia.grid5000.fr/alakhlef/SK_zsl/Neuron/data/ntu60/NTU60_CS.npz')
x_test = d['x_test'].reshape(-1, 300, 2, 25, 3)
y_test = d['y_test'].argmax(axis=1)

TWO_PERSON_CLASSES = [49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59]
NAMES = {51: "pushing", 56: "touch pocket", 58: "walking towards", 59: "walking apart"}

all_offsets = []
print(f"{'classe':<20} {'dist moy':>10} {'dist std':>10} {'offset X':>10} {'offset Y':>10} {'offset Z':>10}")
for c in TWO_PERSON_CLASSES:
    mask = y_test == c
    seqs = x_test[mask]
    offsets = []
    for s in seqs:
        body0, body1 = s[:, 0, :, :], s[:, 1, :, :]
        valid = (np.abs(body0).sum(axis=(1,2)) > 1e-6) & (np.abs(body1).sum(axis=(1,2)) > 1e-6)
        if valid.sum() == 0: continue
        b0v, b1v = body0[valid], body1[valid]
        root_offset = b1v[:, 0, :] - b0v[:, 0, :]   # position racine B - racine A, par frame
        offsets.append(root_offset.mean(axis=0))
    if not offsets: continue
    offsets = np.array(offsets)
    dist = np.linalg.norm(offsets, axis=-1)
    all_offsets.append(offsets)
    name = NAMES.get(c, f"classe{c}")
    print(f"{name:<20} {dist.mean():>10.3f} {dist.std():>10.3f} {offsets[:,0].mean():>10.3f} {offsets[:,1].mean():>10.3f} {offsets[:,2].mean():>10.3f}")

all_offsets = np.concatenate(all_offsets, axis=0)
print(f"\n>>> Offset moyen global (toutes classes 2p): X={all_offsets[:,0].mean():.3f} Y={all_offsets[:,1].mean():.3f} Z={all_offsets[:,2].mean():.3f}")
print(f">>> Distance moyenne globale: {np.linalg.norm(all_offsets, axis=-1).mean():.3f}")
