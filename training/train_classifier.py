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
import util

from models.vqvae import FlatVQVAE
from torchvision.models import resnet50, ResNet50_Weights
from torch.functional import F



@torch.no_grad()
def eval_model(model_vqvae, dataloader, run, classifier):
    sum_acc, sum_ce, n_inputs = 0, 0, 0
    classifier.eval()
    for batch_id, (img, label) in enumerate(dataloader):
        img, label = img.to(DEVICE, non_blocking=True), label.to(DEVICE, non_blocking=True)
        img = model_vqvae(img)[0]
        img = util.denormalize(img)
        img = util.preprocess(img)
        logits = classifier(img)
        sum_ce += F.cross_entropy(input=logits, target=label) * img.shape[0]
        sum_acc += util.accuracy(logits=logits, target=label) * img.shape[0]
        n_inputs += img.shape[0]
    sum_acc = (sum_acc / n_inputs).item()
    sum_ce = (sum_ce / n_inputs).item()

    if run:
        run["val/ce"].append(sum_ce)
        run["val/acc"].append(sum_acc)

    return sum_ce



# Define classifier and load saved model(weights)
if __name__ == '__main__':
    os.nice(10)  # Adjusts the process priority by +10
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda_device", "-c", help='cuda_device', type=int, default=0)
    parser.add_argument("--lr", "-lr", help='learning rate', type=float, default=0.0003)
    parser.add_argument("--batch_size", "-bs", help='batch size', type=int, default=32)

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
    lr = args.lr

    train_cfg = {'batch_size': bs, 'lr': lr, 'model_type': 'classifier'}
    run = neptune.init_run(project="fill in your project name", monitoring_namespace='monitoring',
                           capture_stdout=False, capture_stderr=False, capture_hardware_metrics=False)  # TODO fill in your project name
    run_id = run["sys/id"].fetch()
    run['cfg'] = train_cfg

    (util.RUNS_DIR / run_id).mkdir(exist_ok=True)
    transform_train = transforms.Compose([transforms.RandomResizedCrop(80, scale=(0.08, 1.0), ratio=(3 / 4, 4 / 3)),
                                          transforms.RandomHorizontalFlip(p=0.5),
                                          transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    transform_val = transforms.Compose([transforms.Resize((80, 80)), transforms.ToTensor(),
                                        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

    dataset_train = datasets.ImageFolder(util.IMG_NET_TRAIN, transform=transform_train)
    dataset_val = datasets.ImageFolder(util.IMG_NET_VAL, transform=transform_val)

    train_dataloader = DataLoader(dataset_train, batch_size=bs, shuffle=True)
    val_dataloader = DataLoader(dataset_val, batch_size=bs, shuffle=True)

    model_vqvae = util.vqvae_setup()
    classifier = util.classifier_setup(pretrained=False)
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=lr)

    best_eval_loss = eval_model(model_vqvae=model_vqvae, dataloader=val_dataloader, run=run, classifier=classifier)
    for epoch in range(epochs):
        classifier.train()
        for _, (img, target) in enumerate(train_dataloader):
            img, target = img.to(DEVICE, non_blocking=True), target.to(DEVICE, non_blocking=True)
            with torch.no_grad():
                img = model_vqvae(img)[0]
                img = util.denormalize(img)
                img = util.preprocess(img)
            logits = classifier(img)
            loss = F.cross_entropy(input=logits, target=target)
            optimizer.zero_grad()  # 1. Clear old gradients
            loss.backward()  # 2. Compute gradients (∂loss/∂params)
            optimizer.step()
            run['train/loss'].append(loss)
        eval_loss = eval_model(model_vqvae=model_vqvae, dataloader=val_dataloader, run=run, classifier=classifier)
        if eval_loss < best_eval_loss:
            best_eval_loss = eval_loss
            torch.save(classifier.state_dict(), f=(util.RUNS_DIR / run_id / 'model_st.pt'))
            run['files/model_st.pt'].upload(str(util.RUNS_DIR / run_id / 'model_st.pt'), wait=True)
