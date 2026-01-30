import argparse, os
import util
import torch
import torch.nn.functional as F

# run script to generate evaluate masking strategies on trained transformer

def save_files(ce_over_batches, acc_over_batches, mse_over_batches, code_ce_over_batches, code_ce_over_batches_m, code_acc_over_batches, code_acc_over_batches_m, n_inputs, file_name):
    ces = torch.sum(torch.tensor(ce_over_batches), dim=0) / n_inputs
    accs = torch.sum(torch.tensor(acc_over_batches), dim=0) / n_inputs
    mses = torch.sum(torch.tensor(mse_over_batches), dim=0) / n_inputs
    code_ces = torch.sum(torch.tensor(code_ce_over_batches), dim=0) / n_inputs
    code_accs = torch.sum(torch.tensor(code_acc_over_batches), dim=0) / n_inputs
    code_ces_m = torch.sum(torch.tensor(code_ce_over_batches_m), dim=0) / n_inputs
    code_accs_m = torch.sum(torch.tensor(code_acc_over_batches_m), dim=0) / n_inputs

    torch.save(accs, util.DATA_DIR / 'mask_ratio_eval' / f'{file_name}_acc.pt')
    torch.save(ces, util.DATA_DIR / 'mask_ratio_eval' / f'{file_name}_ce.pt')
    torch.save(code_accs, util.DATA_DIR / 'mask_ratio_eval' / f'{file_name}_code_acc.pt')
    torch.save(code_ces, util.DATA_DIR / 'mask_ratio_eval' / f'{file_name}_code_ce.pt')
    torch.save(code_accs_m, util.DATA_DIR / 'mask_ratio_eval' / f'{file_name}_code_acc_m.pt')
    torch.save(code_ces_m, util.DATA_DIR / 'mask_ratio_eval' / f'{file_name}_code_ce_m.pt')
    torch.save(mses, util.DATA_DIR / 'mask_ratio_eval' / f'{file_name}_mse.pt')


@torch.no_grad()
def random_attn_eval(dataloader, vqvae, transformer, classifier, step_size=1, file_name='rnd'):
    (ce_over_batches, acc_over_batches, mse_over_batches, code_ce_over_batches, code_acc_over_batches,
     code_ce_over_batches_m, code_acc_over_batches_m) = [], [], [], [], [], [], []
    n_inputs = 0
    for _, (img, label) in enumerate(dataloader):
        img, label = img.to(util.DEVICE), label.to(util.DEVICE)
        q_batch, _, id_b, _, _ = vqvae.encode(img)
        q_batch, id_b = torch.flatten(q_batch, start_dim=2).permute(0, 2, 1), torch.flatten(id_b, start_dim=1)
        vqvae_out = vqvae.decode(q_batch.permute(0, 2, 1).reshape(-1, 144, 20, 20))
        q_masked, index_masked = q_batch.clone(), id_b.clone()
        rnd_mask = torch.stack([torch.randperm(q_batch.shape[1]) for _ in range(q_batch.shape[0])]).to(util.DEVICE)
        (ce_over_tokens, acc_over_tokens, recon_errors_over_tokens, code_ce_over_tokens, code_acc_over_tokens,
         code_ce_over_tokens_m, code_acc_over_tokens_m)  = [], [], [], [], [], [], []
        rows = torch.arange(q_batch.size(0)).unsqueeze(1)
        n_inputs += q_batch.shape[0]
        for i in range(0, q_batch.shape[1] // step_size + 1):
            pos_to_mask = rnd_mask[:, :i]
            q_masked[rows, pos_to_mask, :] = 0
            logits = transformer(inputs_embeds=q_masked, output_hidden_states=True).logits
            max_conf_per_pos, max_index_per_pos = torch.max(logits, dim=2)
            index_masked[rows, pos_to_mask] = max_index_per_pos[rows, pos_to_mask]

            code_ce_over_tokens.append(F.cross_entropy(input=logits.permute(0, 2, 1), target=id_b).item())
            code_acc_over_tokens.append(util.calc_acc(logits, id_b).item())
            code_ce_over_tokens_m.append(F.cross_entropy(input=logits[rows, pos_to_mask].permute(0, 2, 1), target=id_b[rows, pos_to_mask]).item())
            code_acc_over_tokens_m.append(util.calc_acc(logits[rows, pos_to_mask], id_b[rows, pos_to_mask]))

            recons_from_max_indices = vqvae.decode_code(index_masked.reshape(-1, 20, 20).to(util.DEVICE))
            class_logits = classifier(util.preprocess(util.denormalize(recons_from_max_indices)))
            recon_errors_over_tokens.append(F.mse_loss(util.denormalize(recons_from_max_indices), util.denormalize(vqvae_out)).item())
            ce_over_tokens.append(F.cross_entropy(input=class_logits, target=label).item())
            acc_over_tokens.append(util.accuracy(logits=class_logits, target=label).item())

        ce_over_batches.append((torch.tensor(ce_over_tokens) * q_batch.shape[0]).tolist())
        code_ce_over_batches.append((torch.tensor(code_ce_over_tokens) * q_batch.shape[0]).tolist())
        code_ce_over_batches_m.append((torch.tensor(code_ce_over_tokens_m) * q_batch.shape[0]).tolist())
        code_acc_over_batches.append((torch.tensor(code_acc_over_tokens) * q_batch.shape[0]).tolist())
        code_acc_over_batches_m.append((torch.tensor(code_acc_over_tokens_m) * q_batch.shape[0]).tolist())
        acc_over_batches.append((torch.tensor(acc_over_tokens) * q_batch.shape[0]).tolist())
        mse_over_batches.append((torch.tensor(recon_errors_over_tokens) * q_batch.shape[0]).tolist())

    save_files(ce_over_batches, acc_over_batches, mse_over_batches, code_ce_over_batches, code_ce_over_batches_m, code_acc_over_batches, code_acc_over_batches_m, n_inputs, file_name)



@torch.no_grad()
def add_attn_eval(dataloader, vqvae, transformer, classifier, step_size=1, file_name='add'):
    (ce_over_batches, acc_over_batches, mse_over_batches, code_ce_over_batches, code_acc_over_batches,
     code_ce_over_batches_m, code_acc_over_batches_m) = [], [], [], [], [], [], []
    n_inputs = 0
    for _, (img, label) in enumerate(dataloader):
        img, label = img.to(util.DEVICE), label.to(util.DEVICE)
        q_batch, _, index_repr_batch, _, _ = vqvae.encode(img)
        q_batch, index_repr_batch = (torch.flatten(q_batch, start_dim=2).permute(0, 2, 1),
                                     torch.flatten(index_repr_batch, start_dim=1))
        vqvae_out = vqvae.decode(q_batch.permute(0, 2, 1).reshape(-1, 144, 20, 20))
        q_masked, _, _ = util.full_mask(q_batch, index_repr_batch)
        pos_to_unmask = torch.empty((q_batch.shape[0], 0), dtype=torch.int64).to(util.DEVICE)
        (ce_over_tokens, acc_over_tokens, recon_errors_over_tokens, code_ce_over_tokens, code_acc_over_tokens,
         code_ce_over_tokens_m, code_acc_over_tokens_m) = [], [], [], [], [], [], []
        rows = torch.arange(q_batch.size(0)).unsqueeze(1)
        n_inputs += q_batch.shape[0]
        pos_to_mask_for_val = torch.stack([torch.arange(0, 400) for _ in range(q_batch.shape[0])])
        # for computing logits on masked tokens only, for val only
        for i in range(0, q_batch.shape[1], step_size):
            q_masked[rows, pos_to_unmask] = q_batch[rows, pos_to_unmask]
            logits = transformer(inputs_embeds=q_masked, output_hidden_states=True).logits
            max_conf_per_pos, max_index_per_pos = torch.max(logits, dim=2)
            max_index_per_pos[rows, pos_to_unmask] = index_repr_batch[rows, pos_to_unmask]
            recons_from_max_indices = vqvae.decode_code(max_index_per_pos.reshape(-1, 20, 20).to(util.DEVICE))

            code_ce_over_tokens.append(F.cross_entropy(input=logits.permute(0, 2, 1), target=index_repr_batch).item())
            code_acc_over_tokens.append(util.calc_acc(logits, index_repr_batch).item())

            # for computing logits on masked tokens only, for val only
            pos_to_mask_for_val[rows, pos_to_unmask] = - 1
            tmp_list = pos_to_mask_for_val.tolist()
            cleaned = [[x for x in sub if x != -1] for sub in tmp_list]
            cur_pos_to_mask = torch.tensor(cleaned)


            code_ce_over_tokens_m.append(F.cross_entropy(input=logits[rows, cur_pos_to_mask].permute(0, 2, 1),
                                                         target=index_repr_batch[rows, cur_pos_to_mask]).item())
            code_acc_over_tokens_m.append(util.calc_acc(logits[rows, cur_pos_to_mask], index_repr_batch[rows, cur_pos_to_mask]))

            class_logits = classifier(util.preprocess(util.denormalize(recons_from_max_indices)))
            recon_errors_over_tokens.append(F.mse_loss(util.denormalize(recons_from_max_indices), util.denormalize(vqvae_out)).item())
            ce_over_tokens.append(F.cross_entropy(input=class_logits, target=label).item())
            acc_over_tokens.append(util.accuracy(logits=class_logits, target=label).item())


            sorted_max_conf_per_pos = torch.argsort(max_conf_per_pos, dim=1)
            pos_to_unmask = util.conc_unique_elements(pos_to_unmask, sorted_max_conf_per_pos, n_elements=step_size)


        ce_over_batches.append((torch.tensor(ce_over_tokens) * q_batch.shape[0]).tolist())
        code_ce_over_batches.append((torch.tensor(code_ce_over_tokens) * q_batch.shape[0]).tolist())
        code_ce_over_batches_m.append((torch.tensor(code_ce_over_tokens_m) * q_batch.shape[0]).tolist())
        code_acc_over_batches.append((torch.tensor(code_acc_over_tokens) * q_batch.shape[0]).tolist())
        code_acc_over_batches_m.append((torch.tensor(code_acc_over_tokens_m) * q_batch.shape[0]).tolist())
        acc_over_batches.append((torch.tensor(acc_over_tokens) * q_batch.shape[0]).tolist())
        mse_over_batches.append((torch.tensor(recon_errors_over_tokens) * q_batch.shape[0]).tolist())

    save_files(ce_over_batches, acc_over_batches, mse_over_batches, code_ce_over_batches, code_ce_over_batches_m,
               code_acc_over_batches, code_acc_over_batches_m, n_inputs, file_name)


@torch.no_grad()
def sel_attn_eval(dataloader, vqvae, transformer, classifier, step_size=1, file_name='sel'):
    (ce_over_batches, acc_over_batches, mse_over_batches, code_ce_over_batches, code_acc_over_batches,
     code_ce_over_batches_m, code_acc_over_batches_m) = [], [], [], [], [], [], []
    n_inputs = 0
    for _, (img, label) in enumerate(dataloader):
        img, label = img.to(util.DEVICE), label.to(util.DEVICE)
        q_b, _, index_b, _, _ = vqvae.encode(img)
        q_b, index_b = torch.flatten(q_b, start_dim=2).permute(0, 2, 1), torch.flatten(index_b, start_dim=1)
        vqvae_out = vqvae.decode(q_b.permute(0, 2, 1).reshape(-1, 144, 20, 20))

        logits_from_unmasked_img = transformer(inputs_embeds=q_b, output_hidden_states=True).logits
        max_conf_unmasked_img, max_index_unmasked_img = torch.max(logits_from_unmasked_img, dim=2)
        arg_sort_max_conf_unmasked_img = torch.argsort(max_conf_unmasked_img, dim=1, descending=True)
        rows = torch.arange(index_b.size(0)).unsqueeze(1)
        index_b_gt = index_b.clone()
        n_inputs += q_b.shape[0]
        (ce_over_tokens, acc_over_tokens, recon_errors_over_tokens, code_ce_over_tokens, code_acc_over_tokens,
         code_ce_over_tokens_m, code_acc_over_tokens_m) = [], [], [], [], [], [], []
        for i in range(0, q_b.shape[1] + 1, step_size):
            masked_pos = arg_sort_max_conf_unmasked_img[:, :i]
            q_b[rows, masked_pos] = 0
            logits_from_masked_img = transformer(inputs_embeds=q_b, output_hidden_states=True).logits
            max_conf_masked_img, max_index_masked_img = torch.max(logits_from_masked_img, dim=2)
            index_b[rows, masked_pos] = max_index_masked_img[rows, masked_pos]
            recons_from_max_indices = vqvae.decode_code(index_b.reshape(-1, 20, 20).to(util.DEVICE))  # bx20x20 -> bx3x80x80
            # replace indices at masked positions with most likely indices
            class_logits = classifier(util.preprocess(util.denormalize(recons_from_max_indices)))
            recon_errors_over_tokens.append(F.mse_loss(util.denormalize(recons_from_max_indices), util.denormalize(vqvae_out)).item())
            ce_over_tokens.append(F.cross_entropy(input=class_logits, target=label).item())
            acc_over_tokens.append(util.accuracy(logits=class_logits, target=label).item())

            code_ce_over_tokens.append(F.cross_entropy(input=logits_from_masked_img.permute(0, 2, 1), target=index_b_gt).item())
            code_acc_over_tokens.append(util.calc_acc(logits=logits_from_masked_img, id_b=index_b_gt).item())
            code_ce_over_tokens_m.append(F.cross_entropy(input=logits_from_masked_img[rows, masked_pos].permute(0, 2, 1), target=index_b_gt[rows, masked_pos]).item())
            code_acc_over_tokens_m.append(util.calc_acc(logits=logits_from_masked_img[rows, masked_pos], id_b=index_b_gt[rows, masked_pos]))

        ce_over_batches.append((torch.tensor(ce_over_tokens) * q_b.shape[0]).tolist())
        code_ce_over_batches.append((torch.tensor(code_ce_over_tokens) * q_b.shape[0]).tolist())
        code_ce_over_batches_m.append((torch.tensor(code_ce_over_tokens_m) * q_b.shape[0]).tolist())
        code_acc_over_batches.append((torch.tensor(code_acc_over_tokens) * q_b.shape[0]).tolist())
        code_acc_over_batches_m.append((torch.tensor(code_acc_over_tokens_m) * q_b.shape[0]).tolist())
        acc_over_batches.append((torch.tensor(acc_over_tokens) * q_b.shape[0]).tolist())
        mse_over_batches.append((torch.tensor(recon_errors_over_tokens) * q_b.shape[0]).tolist())

    save_files(ce_over_batches, acc_over_batches, mse_over_batches, code_ce_over_batches, code_ce_over_batches_m,
               code_acc_over_batches, code_acc_over_batches_m, n_inputs, file_name)


if __name__ == "__main__":
    if __name__ == '__main__':
        os.nice(10)  # Adjusts the process priority by +10
        parser = argparse.ArgumentParser()
        parser.add_argument("--cuda_device", "-c", help='cuda_device', type=int, default=0)
        parser.add_argument("--batch_size", "-bs", help='batch size', type=int, default=4)
        parser.add_argument("--trans_id", "-t", help='batch size', type=str, default=None)
        parser.add_argument('--masking_strategy','-ms', help='masking strat', type=str, default=None)


        args = parser.parse_args()
        if torch.cuda.is_available():
            DEVICE = 'cuda'
            torch.cuda.set_device(args.cuda_device)
            torch.cuda.empty_cache()
        else:
            DEVICE = 'cpu'

        val_dataloader = util.load_val_data(args.batch_size)
        vqvae, transformer, classifier = util.model_setup(init_from=args.trans_id)

        if args.masking_strategy == 'random':
            random_attn_eval(transformer=transformer, classifier=classifier, vqvae=vqvae, dataloader=val_dataloader,
                             step_size=1, file_name=f'{args.trans_id}_rnd')
        elif args.masking_strategy == 'additive':
            add_attn_eval(transformer=transformer, classifier=classifier, vqvae=vqvae, dataloader=val_dataloader,
                            step_size=1, file_name=f'{args.trans_id}_add')
        elif args.masking_strategy == 'selective':
            sel_attn_eval(transformer=transformer, classifier=classifier, vqvae=vqvae, dataloader=val_dataloader,
                          step_size=1, file_name=f'{args.trans_id}_sel')
        elif args.masking_strategy == 'all':
            random_attn_eval(transformer=transformer, classifier=classifier, vqvae=vqvae, dataloader=val_dataloader,
                             step_size=1, file_name=f'{args.trans_id}_rnd')
            add_attn_eval(transformer=transformer, classifier=classifier, vqvae=vqvae, dataloader=val_dataloader,
                          step_size=1, file_name=f'{args.trans_id}_add')
            sel_attn_eval(transformer=transformer, classifier=classifier, vqvae=vqvae, dataloader=val_dataloader,
                          step_size=1, file_name=f'{args.trans_id}_sel')
        else:
            raise NameError(f"Unknown masking strategy {args.masking_strategy}")


