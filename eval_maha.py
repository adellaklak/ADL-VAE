"""
Regle de decision alternative sur un run deja entraine (zero parametre appris).
Compare cosine-prototype vs Mahalanobis diagonale :
  d(z,k) = sum_j ((z_j - mu_kj)/sigma_kj)^2,  sigma = exp(0.5*logvar)
Usage : python eval_maha.py --ss 5 --wdir results/basefull_5_r
"""
import argparse
import numpy as np, torch, torch.nn.functional as F
from model import Encoder
from data_cnn60 import NTUDataLoaders

p = argparse.ArgumentParser()
p.add_argument('--ntu', type=int, default=60)
p.add_argument('--ss', type=int, required=True)
p.add_argument('--st', type=str, default='r')
p.add_argument('--le', type=str, default='ViT-B/32')
p.add_argument('--tm', type=str, default='lb_ad_md')
p.add_argument('--wdir', type=str, required=True)
p.add_argument('--epoch', type=int, default=16999)
p.add_argument('--latent_size', type=int, default=100)
p.add_argument('--gpu', type=int, default=0)
a = p.parse_args()

dev = torch.device(f'cuda:{a.gpu}'); torch.cuda.set_device(dev)

tfl = [torch.from_numpy(np.load(f'./text_feats/{a.le}/{m}_{a.ntu}.npy')) for m in a.tm.split('_')]
tf = torch.concat(tfl, dim=-1); dim_t = tf.size(-1)
te_all = tf / torch.norm(tf, dim=1, keepdim=True).repeat([1, dim_t])
unseen_inds = np.sort(np.load(f'label_splits/{a.st}u{a.ss}.npy'))
ut = te_all[unseen_inds, :].float().to(dev)

base = f'{a.wdir}/{a.le}/{a.tm}'
se = Encoder([256, a.latent_size]).to(dev)
txt = Encoder([dim_t, a.latent_size]).to(dev)
se.load_state_dict(torch.load(f'{base}/se_{a.epoch}.pth.tar', map_location=dev)['state_dict'])
txt.load_state_dict(torch.load(f'{base}/te_{a.epoch}.pth.tar', map_location=dev)['state_dict'])
se.eval(); txt.eval()

ld = NTUDataLoaders(f'sk_feats/shift_{a.ss}_r', 'max', 1).get_val_loader(64, 0)
smu, y = [], []
with torch.no_grad():
    for inp, t in ld:
        m, _ = se(inp.to(dev)); smu.append(m.cpu()); y.append(t)
smu = torch.cat(smu).to(dev); y = torch.cat(y).numpy()
u = torch.from_numpy(unseen_inds)

def report(name, pred):
    pg = u[pred.cpu()].numpy(); acc = (pg == y).mean()
    per = {int(c): float((pg[y == c] == c).mean()) for c in unseen_inds}
    print(f'{name:<20} ZSL={acc:.2%} | ' + ' '.join(f'{c}:{v:.2f}' for c, v in per.items()))
    return acc

with torch.no_grad():
    tmu, tlv = txt(ut)
    a_cos = report('cosine-prototype', (F.normalize(smu,dim=1) @ F.normalize(tmu,dim=1).t()).argmax(1))
    sig = torch.exp(0.5 * tlv)
    d = ((smu.unsqueeze(1) - tmu.unsqueeze(0)) / sig.unsqueeze(0)).pow(2).sum(-1)
    a_mah = report('mahalanobis-diag', d.argmin(1))
print(f'\n>>> [{a.wdir} ss={a.ss}] gain Maha vs cosine : {100*(a_mah-a_cos):+.2f} pts')
