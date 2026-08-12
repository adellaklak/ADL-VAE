import sys
sys.path.insert(0, '/srv/storage/stars@storage3.sophia.grid5000.fr/alakhlef/SK_zsl/MotionGPT3')
import numpy as np
import torch

exec(open('/srv/storage/stars@storage3.sophia.grid5000.fr/alakhlef/SK_zsl/MotionGPT3/compare_embeddings.py').read().split("# ---------- 6.")[0])

# ---------- Sanity check : un vrai echantillon x_test doit retomber pres de son centroide ----------
d = np.load('/srv/storage/stars@storage3.sophia.grid5000.fr/alakhlef/SK_zsl/Neuron/data/ntu60/NTU60_CS.npz')
x_test, y_onehot = d['x_test'], d['y_test']
y_test = y_onehot.argmax(axis=1)

target_class = 42  # falling down
idx = np.where(y_test == target_class)[0][0]
sample = x_test[idx]  # (300, 150) -- garder tel quel, deja zero-pade


sample = sample.reshape(1, 300, 2, 25, 3).transpose(0, 4, 1, 3, 2).astype(np.float32)  # (1,3,T,25,2)
x = torch.from_numpy(sample)

with torch.no_grad():
    _, emb = model(x)
emb = emb.squeeze(0).numpy()

sims = {c: cosine(emb, centroids[c]) for c in classes}
ranked = sorted(sims.items(), key=lambda kv: -kv[1])
print(f"\nsim au centroide {target_class} (vraie classe): {sims[target_class]:.3f}")
print(f"meilleure classe predite: {ranked[0][0]} (sim={ranked[0][1]:.3f})")
print(f"rang de la vraie classe: {[c for c,_ in ranked].index(target_class)+1}/{len(classes)}")
