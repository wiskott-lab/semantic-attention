import pickle
import torch
import util

# run script to generate reconstructed images and compute various metrics

def save_pkl(file_name, recons, masks, perc, confidences, normed_confidences, correctly_classified, recon_without, correctly_classified_without):
    with open(util.DATA_DIR / 'img_recons' / (file_name + '.pkl'), "wb") as f:
        pickle.dump((recons, masks, perc, confidences, normed_confidences, correctly_classified, recon_without, correctly_classified_without), f)


@torch.no_grad()
def random_plot(img, labels, vqvae, transformer, classifier, step_size=1, img_ids=None, file_name='recons_rnd_attn', plot_every=40):
    q_batch, _, id_b, _, _ = vqvae.encode(img)
    q_batch, id_b = torch.flatten(q_batch, start_dim=2).permute(0, 2, 1), torch.flatten(id_b, start_dim=1)
    q_masked, index_masked = q_batch.clone(), id_b.clone()
    rnd_mask = torch.stack([torch.randperm(q_batch.shape[1]) for _ in range(q_batch.shape[0])]).to(util.DEVICE)
    mask = torch.ones(size=(q_batch.shape[0], q_batch.shape [1]))
    masks, recons, confidences, normed_confidences, perc, correctly_classified, recon_without, correctly_classified_without = [], [], [], [], [], [], [], []
    rows = torch.arange(q_batch.size(0)).unsqueeze(1)

    for i in range(0, q_batch.shape[1] + 1, step_size):
        pos_to_mask = rnd_mask[:, :i]
        q_masked[rows, pos_to_mask]  = 0
        logits = transformer(inputs_embeds=q_masked, output_hidden_states=True).logits
        max_conf_per_pos, max_index_per_pos = torch.max(logits, dim=2)

        mask[rows, pos_to_mask[:, :i]] = 0
        index_masked[rows, pos_to_mask] = max_index_per_pos[rows, pos_to_mask]
        if i % plot_every == 0:
            recons.append(vqvae.decode_code(index_masked.reshape(-1, 20, 20)))
            class_logits = classifier(util.preprocess(util.denormalize(recons[-1])))
            correctly_classified.append(util.accuracy_no_reduction(class_logits, labels))
            masks.append(mask.clone())
            perc.append( i / 400)
            confidences.append(max_conf_per_pos)
            normed_confidences.append(torch.max(torch.softmax(logits, dim=2), dim=2)[0])
            recon_without.append(vqvae.decode(q_masked.permute(0, 2, 1).reshape(-1, 144, 20, 20)))
            class_logits = classifier(util.preprocess(util.denormalize(recon_without[-1])))
            correctly_classified_without.append(util.accuracy_no_reduction(class_logits, labels))
    save_pkl(file_name, recons, masks, perc, confidences, normed_confidences, correctly_classified, recon_without, correctly_classified_without)

@torch.no_grad()
def selective_direct_plot(img, labels, vqvae, transformer, classifier, step_size=1, img_ids=None, file_name='recons_sel_attn', plot_every=40):
    q_batch, _, id_b, _, _ = vqvae.encode(img)
    q_b, index_b = torch.flatten(q_batch, start_dim=2).permute(0, 2, 1), torch.flatten(id_b, start_dim=1)
    mask = torch.ones(size=(q_b.shape[0], q_b.shape[1]))
    rows = torch.arange(index_b.size(0)).unsqueeze(1)

    logits_from_unmasked_img = transformer(inputs_embeds=q_b, output_hidden_states=True).logits
    max_conf_unmasked_img, max_index_unmasked_img = torch.max(logits_from_unmasked_img, dim=2)
    arg_sort_max_conf_unmasked_img = torch.argsort(max_conf_unmasked_img, dim=1, descending=True)

    masks, recons, confidences, normed_confidences, perc, correctly_classified, recon_without, correctly_classified_without = [], [], [], [], [], [], [], []

    for i in range(0, q_b.shape[1] + 1, step_size):
        masked_pos = arg_sort_max_conf_unmasked_img[:, :i]
        mask[rows, masked_pos] = 0
        q_b[rows, masked_pos] = 0

        logits_from_masked_img = transformer(inputs_embeds=q_b, output_hidden_states=True).logits
        max_conf_masked_img, max_index_masked_img = torch.max(logits_from_masked_img, dim=2)
        index_b[rows, masked_pos] = max_index_masked_img[rows, masked_pos]
        if i % plot_every == 0:
            recons.append(vqvae.decode_code(index_b.reshape(-1, 20, 20).to(util.DEVICE)))
            class_logits = classifier(util.preprocess(util.denormalize(recons[-1])))
            correctly_classified.append(util.accuracy_no_reduction(class_logits, labels))
            masks.append(mask.clone())
            perc.append(i / 400)
            confidences.append(max_conf_masked_img)
            normed_confidences.append(torch.max(torch.softmax(logits_from_masked_img, dim=2), dim=2)[0])
            recon_without.append(vqvae.decode(q_b.permute(0, 2, 1).reshape(-1, 144, 20, 20)))
            class_logits = classifier(util.preprocess(util.denormalize(recon_without[-1])))
            correctly_classified_without.append(util.accuracy_no_reduction(class_logits, labels))


    save_pkl(file_name, recons, masks, perc, confidences, normed_confidences, correctly_classified, recon_without, correctly_classified_without)


@torch.no_grad()
def additive_plot(img, labels, vqvae, transformer, classifier, step_size=1, img_ids=None, file_name='recons_sel_attn', plot_every=40, init_unmask_pos=None):
    q_batch, _, id_b, _, _ = vqvae.encode(img)
    q_batch, index_repr_batch = torch.flatten(q_batch, start_dim=2).permute(0, 2, 1), torch.flatten(id_b, start_dim=1)
    q_masked, _, _ = util.full_mask(q_batch, index_repr_batch)
    pos_to_unmask = torch.empty((q_batch.shape[0], 0), dtype=torch.int64).to(util.DEVICE)
    rows = torch.arange(q_batch.size(0)).unsqueeze(1)
    mask = torch.zeros(size=(q_batch.shape[0], q_batch.shape[1]))
    masks, recons, confidences, normed_confidences, perc, correctly_classified, recon_without, correctly_classified_without = [], [], [], [], [], [], [], []

    for i in range(0, q_batch.shape[1] + 1, step_size):
        q_masked[rows, pos_to_unmask] = q_batch[rows, pos_to_unmask]
        logits = transformer(inputs_embeds=q_masked, output_hidden_states=True).logits
        max_conf_per_pos, max_index_per_pos = torch.max(logits, dim=2)
        max_index_per_pos[rows, pos_to_unmask] = index_repr_batch[rows, pos_to_unmask]
        recons_from_max_indices = vqvae.decode_code(max_index_per_pos.reshape(-1, 20, 20).to(util.DEVICE))
        sorted_max_conf_per_pos = torch.argsort(max_conf_per_pos, dim=1)
        pos_to_unmask = util.conc_unique_elements(pos_to_unmask, sorted_max_conf_per_pos, n_elements=1)
        if i % plot_every == 0:
            recons.append(recons_from_max_indices)
            mask[rows, pos_to_unmask[:, :i]] = 1
            class_logits = classifier(util.preprocess(util.denormalize(recons[-1])))
            correctly_classified.append(util.accuracy_no_reduction(class_logits, labels))
            masks.append(mask.clone())
            perc.append((400 - i) / 400)
            confidences.append(max_conf_per_pos)
            normed_confidences.append(torch.max(torch.softmax(logits, dim=2), dim=2)[0])
            recon_without.append(vqvae.decode(q_masked.permute(0, 2, 1).reshape(-1, 144, 20, 20)))
            class_logits = classifier(util.preprocess(util.denormalize(recon_without[-1])))
            correctly_classified_without.append(util.accuracy_no_reduction(class_logits, labels))
    save_pkl(file_name, recons, masks, perc, confidences, normed_confidences, correctly_classified, recon_without, correctly_classified_without)


if __name__ == '__main__':

    util.set_seed(1)

    dataloader = util.load_val_data(1024)
    img, labels = next(iter(dataloader))

    init_trans = 'selective_transformer'
    vqvae, transformer, classifier = util.model_setup(init_from=init_trans)
    random_plot(img=img, labels=labels, vqvae=vqvae, transformer=transformer, classifier=classifier, step_size=40,
                file_name=f'{init_trans}_rnd')  # transfomer - masking technique
    selective_direct_plot(img=img, labels=labels, vqvae=vqvae, transformer=transformer, classifier=classifier, step_size=40,
                          file_name=f'{init_trans}_sel')
    additive_plot(img=img, labels=labels, vqvae=vqvae, transformer=transformer, classifier=classifier, step_size=1,
                  file_name=f'{init_trans}_add')