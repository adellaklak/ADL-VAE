# Text-to-motion augmentation: flow matching gele sur sequence_encoder fige (idee
# MotionGPT3: separer autoencodeur/generation) + contrastif, puis classifieur
# UNIFIE (60 classes, seen reels + unseen synthetiques dans le meme espace latent).
import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from data_cnn60 import NTUDataLoaders
from model import Encoder, MLP, FlowGenerator, info_nce, weights_init

parser = argparse.ArgumentParser()
parser.add_argument('--ss', type=int, required=True)
parser.add_argument('--st', type=str, default='r')
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--wdir', type=str, required=True)
parser.add_argument('--le', type=str, default='ViT-B/32')
parser.add_argument('--ntu', type=int, default=60)
parser.add_argument('--tm', type=str, required=True)
parser.add_argument('--gpu', type=str, required=True)
parser.add_argument('--latent_size', type=int, default=100)
parser.add_argument('--frozen_ckpt', type=str, required=True)
parser.add_argument('--flow_epochs', type=int, default=5000)
parser.add_argument('--cls_epochs', type=int, default=3000)
parser.add_argument('--n_synth_per_class', type=int, default=500)
parser.add_argument('--lambda_contrastive', type=float, default=0.5)
parser.add_argument('--seed', type=int, default=5)
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
torch.manual_seed(args.seed); torch.cuda.manual_seed(args.seed); np.random.seed(args.seed)
device = torch.device("cuda")
vis_emb_input_size = 256

os.makedirs(args.wdir, exist_ok=True)

ntu_loaders = NTUDataLoaders(args.dataset, 'max', 1)
train_loader = ntu_loaders.get_train_loader(64, 0)
zsl_loader = ntu_loaders.get_val_loader(64, 0)
print('Train on %d samples, validate on %d samples' % (ntu_loaders.get_train_size(), ntu_loaders.get_test_size()))

unseen_inds = np.sort(np.load(f'label_splits/{args.st}u{args.ss}.npy'))
seen_inds = np.load(f'label_splits/{args.st}s{args.ntu - args.ss}.npy')

tml = args.tm.split('_')
tfl = [torch.from_numpy(np.load(f'./text_feats/{args.le}/{m}_{args.ntu}.npy')) for m in tml]
text_feat = torch.cat(tfl, dim=-1)
text_emb_input_size = text_feat.size(-1)
text_emb = (text_feat / torch.norm(text_feat, dim=1, keepdim=True)).float().to(device)
print("language embeddings loaded.")

# --- Encodeur squelette GELE (checkpoint config gagnante) ---
sequence_encoder = Encoder([vis_emb_input_size, args.latent_size]).to(device)
sequence_encoder.load_state_dict(torch.load(args.frozen_ckpt)['state_dict'])
sequence_encoder.eval()
for p in sequence_encoder.parameters():
    p.requires_grad = False

flow_gen = FlowGenerator(text_emb_input_size, args.latent_size).to(device)
flow_opt = optim.Adam(flow_gen.parameters(), lr=0.0001)

print("--- Etape 1: entrainement FlowGenerator (texte -> latent squelette gele) ---")
for epoch in range(args.flow_epochs):
    flow_gen.train()
    inputs, target = next(iter(train_loader))
    s = inputs.to(device)
    c = text_emb[target.to(device)]
    with torch.no_grad():
        mu, _ = sequence_encoder(s)   # latent squelette reel (cible)

    fm_loss = flow_gen.flow_matching_loss(mu, c)
    z_gen = flow_gen.sample(c, steps=10)
    nce_loss = info_nce(z_gen, mu)
    loss = fm_loss + args.lambda_contrastive * nce_loss

    flow_opt.zero_grad()
    loss.backward()
    flow_opt.step()

    if epoch % 500 == 0:
        print(f'[flow] epoch {epoch} fm_loss {fm_loss.item():.4f} nce_loss {nce_loss.item():.4f}')

torch.save({'state_dict': flow_gen.state_dict()}, f'{args.wdir}/flow_gen.pth.tar')

print("--- Etape 2: generation de latents synthetiques unseen ---")
flow_gen.eval()
with torch.no_grad():
    synth_latents, synth_labels = [], []
    for u in unseen_inds:
        c_u = text_emb[u].unsqueeze(0).repeat(args.n_synth_per_class, 1)
        z_u = flow_gen.sample(c_u, steps=15)
        synth_latents.append(z_u)
        synth_labels.append(torch.full((args.n_synth_per_class,), int(u)))
    synth_latents = torch.cat(synth_latents, dim=0)
    synth_labels = torch.cat(synth_labels, dim=0)

print("--- Etape 3: classifieur UNIFIE (60 classes, seen reels + unseen synthetiques) ---")
cls = MLP([args.latent_size, args.ntu]).to(device)
cls_opt = optim.Adam(cls.parameters(), lr=0.001)
ce = nn.CrossEntropyLoss().to(device)

all_seen_latents, all_seen_labels = [], []
with torch.no_grad():
    for inputs, target in train_loader:
        s = inputs.to(device)
        mu, _ = sequence_encoder(s)
        all_seen_latents.append(mu)
        all_seen_labels.append(target)
    n_batches_cache = min(len(all_seen_latents), 200)  # cache raisonnable, pas tout le dataset en RAM GPU
    all_seen_latents = torch.cat(all_seen_latents[:n_batches_cache], dim=0)
    all_seen_labels = torch.cat(all_seen_labels[:n_batches_cache], dim=0).to(device)

full_latents = torch.cat([all_seen_latents, synth_latents], dim=0)
full_labels = torch.cat([all_seen_labels, synth_labels.to(device)], dim=0)

for epoch in range(args.cls_epochs):
    cls.train()
    idx = torch.randperm(full_latents.shape[0], device=device)[:256]
    out = cls(full_latents[idx])
    loss = ce(out, full_labels[idx])
    cls_opt.zero_grad(); loss.backward(); cls_opt.step()
    if epoch % 500 == 0:
        acc = (torch.argmax(out, -1) == full_labels[idx]).float().mean().item()
        print(f'[cls] epoch {epoch} loss {loss.item():.4f} batch_acc {acc:.2%}')

print("--- Evaluation ZSL (squelettes unseen reels, classifieur unifie restreint aux unseen) ---")
cls.eval()
unseen_inds_t = torch.from_numpy(unseen_inds).to(device)
correct, total = 0, 0
with torch.no_grad():
    for inputs, target in zsl_loader:
        s = inputs.to(device)
        mu, _ = sequence_encoder(s)
        out = cls(mu)
        out_unseen_only = out[:, unseen_inds_t]
        pred = unseen_inds_t[torch.argmax(out_unseen_only, -1)]
        correct += torch.sum(pred.cpu() == target).item()
        total += target.shape[0]

zsl_acc = correct / total
print(f'zsl_accuracy increased to {zsl_acc:.2%} on cycle final')
np.save(f'{args.wdir}/zsl_acc.npy', np.array([zsl_acc]))
