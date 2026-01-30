from torch import nn
from transformers import DistilBertForMaskedLM, DistilBertConfig
from pathlib import Path

from torchvision.models import resnet50, ResNet50_Weights
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import random
from models.vqvae import FlatVQVAE
import numpy as np
import torch

DATA_DIR = Path(__file__).parent / 'data'
RUNS_DIR = DATA_DIR / 'runs'

VQVAE_WEIGHTS = DATA_DIR / RUNS_DIR / 'vqvae' / 'model_epoch80_flat_vqvae80x80_144x456codebook.pth'
CLASSIFIER_WEIGHTS = DATA_DIR / RUNS_DIR / 'classifier' / 'model_st.pt'

SELECTIVE_TRANSFORMER_WEIGHTS = DATA_DIR / RUNS_DIR / 'selective_transformer' / 'model_st.pt'
RANDOM_TRANSFORMER_WEIGHTS = DATA_DIR / RUNS_DIR / 'random_transformer' / 'model_st.pt'

preprocess = ResNet50_Weights.IMAGENET1K_V2.transforms()

if torch.cuda.is_available():
    DEVICE = 'cuda'
else:
    DEVICE = 'cpu'

# TODO Fill in your imagenet-path, note, we reduced imagenet to classes
IMG_NET_TRAIN = '/your/path/to/Imagenet-100class/train'
IMG_NET_VAL = '/your/path/to/Imagenet-100class/val'


def accuracy_no_reduction(logits, target):
    pred = logits.argmax(dim=1, keepdim=True)
    e = pred.eq(target.view_as(pred))
    return e


def calc_acc(logits, id_b):
    preds = logits.argmax(dim=-1)  # (64, 400)
    acc = (preds == id_b).float().mean()
    return acc


def accuracy(logits, target):
    pred = logits.argmax(dim=1, keepdim=True)
    e = pred.eq(target.view_as(pred)).sum() / target.shape[0]
    return e


def set_cuda_device(cuda_device):
    if torch.cuda.is_available():
        torch.cuda.set_device(cuda_device)
        torch.cuda.empty_cache()


def denormalize(img):
    img = img * torch.tensor([0.5, 0.5, 0.5], device=img.device).view(-1, 1, 1)
    img = img + torch.tensor([0.5, 0.5, 0.5], device=img.device).view(-1, 1, 1)
    return img


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def conc_unique_elements(x: torch.Tensor, y: torch.Tensor, n_elements=1) -> torch.Tensor:
    # Expand dimensions for broadcasting
    x_exp = x.unsqueeze(2)  # (a, n, 1)
    y_exp = y.unsqueeze(1)  # (a, 1, m)
    # Identify elements in y not present in x
    mask = ~(x_exp == y_exp).any(dim=1)  # (a, m)
    idx = torch.argsort(-mask.float(), stable=True)[:, :n_elements]
    to_conc = y.gather(1, idx)
    # Concatenate result
    return torch.cat((x, to_conc), dim=1)


def full_mask(q, indices):
    mask_pattern = torch.ones_like(indices, dtype=torch.bool)
    masked_q = q.clone()  # shallow copy
    masked_q[mask_pattern] = 0  # Assuming 0 is the mask token
    mask_indices = indices.clone()
    mask_indices[~mask_pattern] = -100  # Assuming -100 is the mask label token
    return masked_q, mask_indices, mask_pattern


def vqvae_setup():
    model_vqvae = FlatVQVAE()
    model_vqvae.load_state_dict(torch.load(VQVAE_WEIGHTS, map_location=DEVICE))
    model_vqvae = model_vqvae.to(DEVICE).eval()
    return model_vqvae


def transformer_setup(init_from=None):
    cfg = DistilBertConfig(vocab_size=456, hidden_size=144, sinusoidal_pos_embds=False, n_layers=6, n_heads=4,
                           max_position_embeddings=400)
    model_distil = DistilBertForMaskedLM(cfg).to(DEVICE)
    if init_from is not None:
        model_distil.load_state_dict(torch.load(RUNS_DIR / init_from / 'model_st.pt', map_location=DEVICE))
    model_distil = model_distil.to(DEVICE).eval()
    model_distil.eval()
    return model_distil


def classifier_setup(pretrained=True):
    classifier = resnet50(pretrained=False)
    classifier.fc = nn.Linear(2048, 100)
    if pretrained:
        classifier.load_state_dict(
            torch.load(CLASSIFIER_WEIGHTS, map_location=torch.device(DEVICE), weights_only=False)())
    classifier.to(DEVICE).eval()
    return classifier


def model_setup(init_from=None):
    vqvae = vqvae_setup()
    transformer = transformer_setup(init_from)
    classifier = classifier_setup()
    return vqvae, transformer, classifier


def load_val_data(bs):
    transform_val = transforms.Compose([transforms.Resize((80, 80)), transforms.ToTensor(),
                                        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    dataset_val = datasets.ImageFolder(IMG_NET_VAL, transform=transform_val)
    val_dataloader = DataLoader(dataset_val, batch_size=bs, shuffle=True)
    return val_dataloader


# plot utils
def set_border(ax, correctly_classified, no_color=False):
    for spine in ax.spines.values():
        color = 'green' if correctly_classified else 'red'
        if no_color:
            color = 'black'
        spine.set_edgecolor(color)
        spine.set_linewidth(4)
    ax.set_xticks([])
    ax.set_yticks([])


def hide_extras(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def hide_all_extras(axs):
    for row_id in range(axs.shape[0]):
        for column_id in range(axs.shape[1]):
            hide_extras(axs[row_id][column_id])


def shift_rows_and_columns(axs, rows_to_shift=(), cols_to_shift=(), vertical_shift=0.01, horizontal_shit=0.01):
    cumulative_vertical_shift = 0
    if len(axs.shape) == 1:
        for j in range(axs.shape[0]):
            cumulative_horizontal_shift = 0
            if j in rows_to_shift:
                cumulative_vertical_shift += vertical_shift
            # if axs.shape
            for i in range(axs.shape[0]):
                ax = axs[i]
                pos = ax.get_position()
                if i in cols_to_shift:
                    cumulative_horizontal_shift += horizontal_shit
                ax.set_position(
                    [pos.x0 + cumulative_horizontal_shift, pos.y0 - cumulative_vertical_shift, pos.width, pos.height])
    else:
        cumulative_vertical_shift = 0
        for j in range(axs.shape[0]):
            cumulative_horizontal_shift = 0
            if j in rows_to_shift:
                cumulative_vertical_shift += vertical_shift
            # if axs.shape
            for i in range(axs.shape[1]):
                ax = axs[j, i]
                pos = ax.get_position()
                if i in cols_to_shift:
                    cumulative_horizontal_shift += horizontal_shit
                ax.set_position(
                    [pos.x0 + cumulative_horizontal_shift, pos.y0 - cumulative_vertical_shift, pos.width, pos.height])
