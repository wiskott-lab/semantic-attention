import argparse
from sched import scheduler
import sys
import os
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import DistilBertForMaskedLM, DistilBertConfig
import neptune.new as neptune
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils
from pathlib import Path
import socket
from models.vqvae import FlatVQVAE
from torch.functional import F
import util


def mask_q(quant_b, model_distil, masking_strategy, mask_ratio=0.75, descending=True):
    rows = torch.arange(quant_b.size(0)).unsqueeze(1)
    if masking_strategy == 'random':
        rnd_mask = torch.stack([torch.randperm(quant_b.shape[1]) for _ in range(quant_b.shape[0])]).to(DEVICE)
        pos_to_mask = rnd_mask[:, :int(mask_ratio * quant_b.shape[1])]
    elif masking_strategy == 'selective':
        logits_from_unmasked_img = model_distil(inputs_embeds=quant_b, output_hidden_states=False).logits
        max_conf_unmasked_img, max_index_unmasked_img = torch.max(logits_from_unmasked_img, dim=2)
        arg_sort_max_conf_unmasked_img = torch.argsort(max_conf_unmasked_img, dim=1, descending=descending)
        pos_to_mask = arg_sort_max_conf_unmasked_img[:, :int(mask_ratio * quant_b.shape[1])]
    else:
        raise NameError(f'Unknown masking strategy {masking_strategy}')
    quant_b[rows, pos_to_mask] = 0
    return quant_b, rows, pos_to_mask



def eval_random(model_distil, model_vqvae, classifier, quant_b, id_b, label, sum_eval, masking_strategy, descending,
                masking_ratio=0.75):
    # ce[0], acc[1], ce_m[2], acc_m[3], class_ce[4], class_acc_5
    quant_b, id_b = quant_b.clone(), id_b.clone()
    quant_b, rows, pos_to_mask = mask_q(quant_b, model_distil, masking_strategy, masking_ratio, descending)
    logits = model_distil(inputs_embeds=quant_b, output_hidden_states=False).logits

    sum_eval[0].add_(F.cross_entropy(input=logits.permute(0, 2, 1), target=id_b) * quant_b.shape[0])
    sum_eval[1].add_(util.calc_acc(logits, id_b) * quant_b.shape[0])
    sum_eval[2].add_(F.cross_entropy(input=logits[rows, pos_to_mask].permute(0, 2, 1), target=id_b[rows, pos_to_mask]) * \
                     quant_b.shape[0])
    sum_eval[3].add_(util.calc_acc(logits[rows, pos_to_mask], id_b[rows, pos_to_mask]) * quant_b.shape[0])

    max_conf_per_pos, max_index_per_pos = torch.max(logits, dim=2)
    id_b[rows, pos_to_mask] = max_index_per_pos[rows, pos_to_mask]
    recons_from_max_indices = model_vqvae.decode_code(id_b.reshape(-1, 20, 20).to(DEVICE))
    preprocessed_image = util.preprocess(util.denormalize(recons_from_max_indices))
    class_logits = classifier(preprocessed_image)

    sum_eval[4].add_(F.cross_entropy(input=class_logits, target=label) * quant_b.shape[0])
    sum_eval[5].add_(util.accuracy(logits=class_logits, target=label).item() * quant_b.shape[0])

    id_b[rows, pos_to_mask] = max_index_per_pos[rows, pos_to_mask]
    recons_from_max_indices = model_vqvae.decode_code(id_b.reshape(-1, 20, 20).to(DEVICE))
    class_logits = classifier(recons_from_max_indices)
    sum_eval[4].add_(F.cross_entropy(input=class_logits, target=label) * quant_b.shape[0])
    sum_eval[5].add_(util.accuracy(logits=class_logits, target=label).item() * quant_b.shape[0])



@torch.no_grad()
def eval_model(model_distil, model_vqvae, dataloader, classifier, masking_strategy, masking_ratio=0.5, run=None):
    # ce[0], acc[1], ce_m[2], acc_m[3], class_ce[4], class_acc_5
    sum_sel = torch.zeros(6).to(DEVICE)
    sum_sel_desc = torch.zeros(6).to(DEVICE)
    sum_rnd = torch.zeros(6).to(DEVICE)
    n_inputs = 0
    model_distil.eval()
    for batch_id, (img, label) in enumerate(dataloader):
        img, label = img.to(DEVICE, non_blocking=True), label.to(DEVICE, non_blocking=True)
        quant_b, _, id_b, _, _ = model_vqvae.encode(img)
        quant_b, id_b = torch.flatten(quant_b, start_dim=2).permute(0, 2, 1), torch.flatten(id_b, start_dim=1)
        eval_random(model_distil, model_vqvae, classifier, quant_b, id_b, label, sum_sel, 'selective',False, masking_ratio=masking_ratio)
        eval_random(model_distil, model_vqvae, classifier, quant_b, id_b, label, sum_sel_desc,'selective', True, masking_ratio=masking_ratio)
        eval_random(model_distil, model_vqvae, classifier, quant_b, id_b, label, sum_rnd, 'random', None, masking_ratio=masking_ratio)
        n_inputs += img.shape[0]

    sum_sel = (sum_sel / n_inputs).cpu().numpy()
    sum_sel_desc = (sum_sel_desc / n_inputs).cpu().numpy()
    sum_rnd = (sum_rnd / n_inputs).cpu().numpy()

    if run:
        run["val/rnd_ce_loss"].append(sum_rnd[0])
        run["val/rnd_acc"].append(sum_rnd[1])
        run["val/rnd_ce_loss_masked_only"].append(sum_rnd[2])
        run["val/rnd_acc_masked_only"].append(sum_rnd[3])
        run["val/rnd_class_ce"].append(sum_rnd[4])
        run["val/rnd_class_acc"].append(sum_rnd[5])

        run["val/selective_ce_loss"].append(sum_sel[0])
        run["val/selective_acc"].append(sum_sel[1])
        run["val/selective_ce_loss_masked_only"].append(sum_sel[2])
        run["val/selective_acc_masked_only"].append(sum_sel[3])
        run["val/selective_class_ce"].append(sum_sel[4])
        run["val/selective_class_acc"].append(sum_sel[5])

        run["val/selective_desc_ce_loss"].append(sum_sel_desc[0])
        run["val/selective_desc_acc"].append(sum_sel_desc[1])
        run["val/selective_desc_ce_loss_masked_only"].append(sum_sel_desc[2])
        run["val/selective_desc_acc_masked_only"].append(sum_sel_desc[3])
        run["val/selective_desc_class_ce"].append(sum_sel_desc[4])
        run["val/selective_desc_class_acc"].append(sum_sel_desc[5])
    if masking_strategy == 'selective':
        return sum_sel_desc[2]
    else:
        return sum_rnd[2]


if __name__ == '__main__':
    os.nice(10)  # Adjusts the process priority by +10
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda_device", "-c", help='cuda_device', type=int, default=0)
    parser.add_argument("--lr", "-lr", help='learning rate', type=float, default=0.0003)
    parser.add_argument("--masking_ratio", "-mr", help='learning rate', type=float, default=0.5)

    parser.add_argument("--batch_size", "-bs", help='batch size', type=int, default=128)
    parser.add_argument("--masking_strategy", '-ms', help='masking strat', type=str, default='random')
    parser.add_argument("--train_on_masked_only", "-m", action="store_true", help="train_on_mask_only")
    parser.add_argument("--descending", "-d", action="store_true", help="descending")
    parser.add_argument("--use_resnet_weights", "-r", action="store_true", help="use resnet weights")
    parser.add_argument("--init_from", "-i", help='init_from', type=str, default=None)



    args = parser.parse_args()
    if torch.cuda.is_available():
        DEVICE = 'cuda'
        torch.cuda.set_device(args.cuda_device)
        torch.cuda.empty_cache()
    else:
        DEVICE = 'cpu'

    os.nice(19)
    bs = args.batch_size
    epochs = 10000
    masking_ratio = args.masking_ratio
    masking_strategy = args.masking_strategy
    lr = args.lr
    descending = args.descending
    train_on_masked_only = args.train_on_masked_only
    use_resnet_weights = args.use_resnet_weights


    train_cfg = {'batch_size': bs, 'masking_ratio': masking_ratio, 'masking_strategy': masking_strategy, 'lr': lr,
                 'descending': descending, 'train_on_mask_only': train_on_masked_only,
                 'use_resnet_weights': use_resnet_weights, 'init_from': args.init_from}
    run = neptune.init_run(project="fill_in_project_name", monitoring_namespace='monitoring', # TODO fill in your project name
                           capture_stdout=False, capture_stderr=False, capture_hardware_metrics=False)
    run_id = run["sys/id"].fetch()
    run['cfg'] = train_cfg

    (util.RUNS_DIR / run_id).mkdir(exist_ok=True)
    transform = transforms.Compose([transforms.Resize((80, 80)), transforms.ToTensor(),
                                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

    dataset_train = datasets.ImageFolder(util.IMG_NET_TRAIN, transform=transform)
    dataset_val = datasets.ImageFolder(util.IMG_NET_VAL, transform=transform)

    train_dataloader = DataLoader(dataset_train, batch_size=bs, shuffle=True)
    val_dataloader = DataLoader(dataset_val, batch_size=bs, shuffle=True)

    model_vqvae = util.vqvae_setup()
    model_distil = util.transformer_setup(weight_path=None, init_from=args.init_from)
    classifier = util.classifier_setup(pretrained=True)


    optimizer = torch.optim.AdamW(model_distil.parameters(), lr=lr)
    best_eval_loss = eval_model(model_distil=model_distil, model_vqvae=model_vqvae, dataloader=val_dataloader, run=run,
                                masking_ratio=masking_ratio, classifier=classifier, masking_strategy=masking_strategy)
    for epoch in range(epochs):
        model_distil.train()
        for _, (img, _) in enumerate(train_dataloader):
            img = img.to(DEVICE, non_blocking=True)
            with torch.no_grad():
                quant_b, _, id_b, _, _ = model_vqvae.encode(img)
                quant_b, id_b = torch.flatten(quant_b, start_dim=2).permute(0, 2, 1), torch.flatten(id_b, start_dim=1)
                quant_b, rows, pos_to_mask = mask_q(quant_b, model_distil, masking_strategy, masking_ratio, descending)
            logits = model_distil(inputs_embeds=quant_b, output_hidden_states=False).logits
            if train_on_masked_only:
                logits = logits[rows, pos_to_mask]
                id_b = id_b[rows, pos_to_mask]
            loss = F.cross_entropy(input=logits.permute(0, 2, 1), target=id_b)
            optimizer.zero_grad()  # 1. Clear old gradients
            loss.backward()  # 2. Compute gradients (∂loss/∂params)
            optimizer.step()
            run['train/loss'].append(loss)
        eval_loss = eval_model(model_distil=model_distil, model_vqvae=model_vqvae, dataloader=val_dataloader,
                               run=run, masking_ratio=masking_ratio, classifier=classifier,
                               masking_strategy=masking_strategy)
        if eval_loss < best_eval_loss:
            best_eval_loss = eval_loss
            torch.save(model_distil.state_dict(), f=(util.RUNS_DIR / run_id / 'model_st.pt'))
            run['files/model_st.pt'].upload(str(util.RUNS_DIR / run_id / 'model_st.pt'), wait=True)
