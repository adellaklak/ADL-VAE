import numpy as np
import torch
import torch.nn as nn
from torch.nn import Module, Parameter
import math, glob, os, json

# ---------- 1. Shift pur (copie de shift_pure.py, sans dependance CUDA) ----------
class Shift(Module):
    def __init__(self, channel, stride, init_scale=3):
        super(Shift, self).__init__()
        self.stride = stride
        self.xpos = Parameter(torch.zeros(channel))
        self.ypos = Parameter(torch.zeros(channel))
        self.xpos.data.uniform_(-1e-8, 1e-8)
        self.ypos.data.uniform_(-init_scale, init_scale)

    def forward(self, input):
        N, C, H, W = input.shape
        s = self.stride
        Ho = H // s
        dev, dt = input.device, input.dtype
        xpos = self.xpos.to(dev, dt)
        ypos = self.ypos.to(dev, dt)
        h_out = torch.arange(Ho, device=dev, dtype=dt) * s
        w_out = torch.arange(W, device=dev, dtype=dt)
        src_h = (h_out.view(1, Ho, 1) + ypos.view(C, 1, 1)).expand(C, Ho, W)
        src_w = (w_out.view(1, 1, W) + xpos.view(C, 1, 1)).expand(C, Ho, W)
        gh = src_h / max(H - 1, 1) * 2 - 1
        gw = src_w / max(W - 1, 1) * 2 - 1
        grid = torch.stack([gw, gh], dim=-1)
        inp = input.reshape(N * C, 1, H, W)
        grid_nc = grid.unsqueeze(0).expand(N, C, Ho, W, 2).reshape(N * C, Ho, W, 2)
        out = torch.nn.functional.grid_sample(inp, grid_nc, mode='bilinear',
                                               padding_mode='zeros', align_corners=True)
        return out.reshape(N, C, Ho, W)

# ---------- 2. shift_gcn.py, copie exacte, import CUDA remplace + np.int patche ----------
def conv_init(conv):
    nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    nn.init.constant_(conv.bias, 0)

def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)

class tcn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1):
        super(tcn, self).__init__()
        pad = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1),
                              padding=(pad, 0), stride=(stride, 1))
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        conv_init(self.conv)
        bn_init(self.bn, 1)

    def forward(self, x):
        return self.bn(self.conv(x))

class Shift_tcn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1):
        super(Shift_tcn, self).__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.bn2 = nn.BatchNorm2d(in_channels)
        bn_init(self.bn2, 1)
        self.relu = nn.ReLU(inplace=True)
        self.shift_in = Shift(channel=in_channels, stride=1, init_scale=1)
        self.shift_out = Shift(channel=out_channels, stride=stride, init_scale=1)
        self.temporal_linear = nn.Conv2d(in_channels, out_channels, 1)
        nn.init.kaiming_normal_(self.temporal_linear.weight, mode='fan_out')

    def forward(self, x):
        x = self.bn(x)
        x = self.shift_in(x)
        x = self.temporal_linear(x)
        x = self.relu(x)
        x = self.shift_out(x)
        x = self.bn2(x)
        return x

class Shift_gcn(nn.Module):
    def __init__(self, in_channels, out_channels, A, coff_embedding=4, num_subset=3):
        super(Shift_gcn, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if in_channels != out_channels:
            self.down = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1), nn.BatchNorm2d(out_channels))
        else:
            self.down = lambda x: x
        self.Linear_weight = Parameter(torch.zeros(in_channels, out_channels, requires_grad=True))
        nn.init.normal_(self.Linear_weight, 0, math.sqrt(1.0 / out_channels))
        self.Linear_bias = Parameter(torch.zeros(1, 1, out_channels, requires_grad=True))
        nn.init.constant_(self.Linear_bias, 0)
        self.Feature_Mask = Parameter(torch.ones(1, 25, in_channels, requires_grad=True))
        nn.init.constant_(self.Feature_Mask, 0)
        self.bn = nn.BatchNorm1d(25 * out_channels)
        self.relu = nn.ReLU()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        index_array = np.empty(25 * in_channels).astype(np.int64)
        for i in range(25):
            for j in range(in_channels):
                index_array[i * in_channels + j] = (i * in_channels + j + j * in_channels) % (in_channels * 25)
        self.shift_in = Parameter(torch.from_numpy(index_array), requires_grad=False)
        index_array = np.empty(25 * out_channels).astype(np.int64)
        for i in range(25):
            for j in range(out_channels):
                index_array[i * out_channels + j] = (i * out_channels + j - j * out_channels) % (out_channels * 25)
        self.shift_out = Parameter(torch.from_numpy(index_array), requires_grad=False)

    def forward(self, x0):
        n, c, t, v = x0.size()
        x = x0.permute(0, 2, 3, 1).contiguous()
        x = x.view(n * t, v * c)
        x = torch.index_select(x, 1, self.shift_in)
        x = x.view(n * t, v, c)
        x = x * (torch.tanh(self.Feature_Mask) + 1)
        x = torch.einsum('nwc,cd->nwd', (x, self.Linear_weight)).contiguous()
        x = x + self.Linear_bias
        x = x.view(n * t, -1)
        x = torch.index_select(x, 1, self.shift_out)
        x = self.bn(x)
        x = x.view(n, t, v, self.out_channels).permute(0, 3, 1, 2)
        x = x + self.down(x0)
        return self.relu(x)

class TCN_GCN_unit(nn.Module):
    def __init__(self, in_channels, out_channels, A, stride=1, residual=True):
        super(TCN_GCN_unit, self).__init__()
        self.gcn1 = Shift_gcn(in_channels, out_channels, A)
        self.tcn1 = Shift_tcn(out_channels, out_channels, stride=stride)
        self.relu = nn.ReLU()
        if not residual:
            self.residual = lambda x: 0
        elif in_channels == out_channels and stride == 1:
            self.residual = lambda x: x
        else:
            self.residual = tcn(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x):
        return self.relu(self.tcn1(self.gcn1(x)) + self.residual(x))

class DummyGraph:
    """A n'est jamais utilise numeriquement dans Shift_gcn -- stub minimal."""
    def __init__(self, **kwargs):
        self.A = np.zeros((3, 25, 25))

class Model(nn.Module):
    def __init__(self, num_class=60, num_point=25, num_person=2, in_channels=3):
        super(Model, self).__init__()
        self.graph = DummyGraph()
        A = self.graph.A
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)
        self.l1 = TCN_GCN_unit(3, 64, A, residual=False)
        self.l2 = TCN_GCN_unit(64, 64, A)
        self.l3 = TCN_GCN_unit(64, 64, A)
        self.l4 = TCN_GCN_unit(64, 64, A)
        self.l5 = TCN_GCN_unit(64, 128, A, stride=2)
        self.l6 = TCN_GCN_unit(128, 128, A)
        self.l7 = TCN_GCN_unit(128, 128, A)
        self.l8 = TCN_GCN_unit(128, 256, A, stride=2)
        self.l9 = TCN_GCN_unit(256, 256, A)
        self.l10 = TCN_GCN_unit(256, 256, A)
        self.fc = nn.Linear(256, num_class)
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / num_class))
        bn_init(self.data_bn, 1)

    def forward(self, x):
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)
        x = self.l1(x); x = self.l2(x); x = self.l3(x); x = self.l4(x); x = self.l5(x)
        x = self.l6(x); x = self.l7(x); x = self.l8(x); x = self.l9(x); x = self.l10(x)
        c_new = x.size(1)
        x = x.view(N, M, c_new, -1)
        x = x.mean(3).mean(1)
        return self.fc(x), x

# ---------- 3. Retargeting 22 (HumanML3D/SMPL) -> 25 (NTU) ----------
NTU_FROM_H3D = {
    0: 0,   1: 6,   2: 12,  3: 15,  4: 16,  5: 18,  6: 20,  7: 20,
    8: 17,  9: 19,  10: 21, 11: 21, 12: 1,  13: 4,  14: 7,  15: 10,
    16: 2,  17: 5,  18: 8,  19: 11, 20: 9,  21: 20, 22: 20, 23: 21, 24: 21,
}

def retarget_22_to_25(seq22):
    T = seq22.shape[0]
    seq25 = np.zeros((T, 25, 3), dtype=np.float32)
    for ntu_j, h3d_j in NTU_FROM_H3D.items():
        seq25[:, ntu_j, :] = seq22[:, h3d_j, :]
    seq25 = seq25 - seq25[:, 0:1, :]  # recentrage racine, coherent avec le sanity check
    return seq25

def to_shiftgcn_input(seq25, max_T=300):
    # seq25: (T, 25, 3) deja recentre sur racine -> zero-pad a 300 frames (convention NTU60_CS.npz)
    T = min(seq25.shape[0], max_T)
    x = np.zeros((1, 3, max_T, 25, 2), dtype=np.float32)
    x[0, :, :T, :, 0] = seq25[:T].transpose(2, 0, 1)  # C,T,V
    return torch.from_numpy(x)

# ---------- 4. Chargement modele + poids geles ----------
CKPT = '/srv/storage/stars@storage3.sophia.grid5000.fr/alakhlef/SK_zsl/DescVAE/sk_feats/shift_12_r/weights/best.pt'
model = Model(num_class=60, num_point=25, num_person=2, in_channels=3)
state = torch.load(CKPT, map_location='cpu')
state = state.get('state_dict', state) if isinstance(state, dict) else state
missing, unexpected = model.load_state_dict(state, strict=False)
print(f"[chargement poids] missing={len(missing)} unexpected={len(unexpected)}")
if len(missing) > 5:
    print("  premiers missing:", missing[:5])
model.eval()

# ---------- 5. Centroides reels (deja precalcules) ----------
ztest = np.load('/srv/storage/stars@storage3.sophia.grid5000.fr/alakhlef/SK_zsl/DescVAE/sk_feats/shift_12_r/ztest.npy')
z_label = np.load('/srv/storage/stars@storage3.sophia.grid5000.fr/alakhlef/SK_zsl/DescVAE/sk_feats/shift_12_r/z_label.npy')
classes = sorted(np.unique(z_label).tolist())
centroids = {c: ztest[z_label == c].mean(0) for c in classes}

def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

# ---------- 6. Boucle sur les squelettes generes ----------
RESULTS_DIR = '/srv/storage/stars@storage3.sophia.grid5000.fr/alakhlef/SK_zsl/MotionGPT3/results/motgpt/debug--MoT_vae_trainemb_cls1_cfgrand_2optim_lr2_lr1/samples_2026-08-08-03-41-21'
MAP_CSV = '/srv/storage/stars@storage3.sophia.grid5000.fr/alakhlef/SK_zsl/MotionGPT3/tmp_prompts/unseen_variants_map.csv'

rows = []
with open(MAP_CSV) as f:
    for line in f.readlines()[1:]:
        idx, cls, name, variant = line.strip().split(',')
        rows.append((int(idx), int(cls), name, variant))

results = []
with torch.no_grad():
    for idx, true_cls, name, variant in rows:
        npy_path = f"{RESULTS_DIR}/{idx}_out.npy"
        if not os.path.exists(npy_path):
            continue
        if true_cls not in centroids:
            print(f"[skip] classe {true_cls} ({name}) = unseen ss=5 uniquement, pas ss=12 -- a traiter separement")
            continue
        seq = np.load(npy_path)
        if seq.ndim == 4:
            seq = seq.squeeze(0)
        seq25 = retarget_22_to_25(seq)
        x = to_shiftgcn_input(seq25)
        _, emb = model(x)
        emb = emb.squeeze(0).numpy()

        sims = {c: cosine(emb, centroids[c]) for c in classes}
        ranked = sorted(sims.items(), key=lambda kv: -kv[1])
        pred_cls = ranked[0][0]
        results.append({
            'idx': idx, 'true_class': true_cls, 'name': name, 'variant': variant,
            'sim_to_true': sims[true_cls], 'predicted_class': pred_cls,
            'sim_to_predicted': ranked[0][1], 'correct': pred_cls == true_cls,
            'rank_of_true': [c for c, _ in ranked].index(true_cls) + 1,
        })

print(f"\n{'idx':>3} {'classe':>6} {'variante':<20} {'nom':<18} {'sim->vraie':>10} {'predite':>7} {'sim->predite':>12} {'rang vraie':>10} {'OK':>4}")
for r in results:
    print(f"{r['idx']:>3} {r['true_class']:>6} {r['variant']:<20} {r['name']:<18} "
          f"{r['sim_to_true']:>10.3f} {r['predicted_class']:>7} {r['sim_to_predicted']:>12.3f} "
          f"{r['rank_of_true']:>10} {'OUI' if r['correct'] else 'non':>4}")

n_correct = sum(r['correct'] for r in results)
print(f"\n>>> {n_correct}/{len(results)} correctement classees (top-1 parmi les 12 centroides unseen)")

with open('compare_embeddings_results.json', 'w') as f:
    json.dump(results, f, indent=2)
