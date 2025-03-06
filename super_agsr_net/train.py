import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from preprocessing import pad_HR_adj
from model import gaussian_noise_layer
import copy
from tqdm.notebook import tqdm
from utils import get_device

device = get_device()

# Loss functions
criterion = nn.SmoothL1Loss(beta=0.01)
criterion_L1 = nn.L1Loss()
bce_loss = nn.BCELoss()

def cal_error(model_outputs, hr, mask):
    """Calculate L1 error on masked (off-diagonal) entries."""
    return criterion_L1(model_outputs[mask], hr[mask])

def train_gan(
    netG,
    optimizerG,
    netD,
    optimizerD,
    subjects_adj,
    subjects_labels,
    args,
    test_adj=None,
    test_ground_truth=None,
    stop_gan_mae=None,
):
    """
    Train the GAN AGSR model with coupled encoder–decoder architecture and additional
    reconstruction and latent space losses.

    netG: Generator (updated GSRNet)
    netD: Discriminator
    subjects_adj: List/array of LR adjacency matrices
    subjects_labels: List/array of HR ground truth adjacency matrices
    args: Dictionary of hyperparameters
    """
    all_epochs_loss = []
    no_epochs = args.epochs
    best_mae = np.inf
    early_stop_patient = args.early_stop_patient
    early_stop_count = 0
    best_model = None

    # Move models to device.
    netG = netG.to(device)
    netD = netD.to(device)

    # Create a base mask for the HR graph (off-diagonal elements).
    base_mask = torch.triu(torch.ones(args.hr_dim, args.hr_dim), diagonal=1).bool().to(device)

    with tqdm(range(no_epochs), desc="Epoch Progress", unit="epoch") as tepoch:
        for epoch in tepoch:
            epoch_loss = []
            epoch_error = []

            netG.train()
            netD.train()

            for lr, hr in zip(subjects_adj, subjects_labels):
                optimizerG.zero_grad()
                optimizerD.zero_grad()

                # Convert LR and HR matrices from numpy arrays to tensors.
                lr = torch.from_numpy(lr).type(torch.FloatTensor).to(device)
                hr = torch.from_numpy(hr).type(torch.FloatTensor).to(device)

                # Forward pass through the generator.
                # Now netG receives both LR and HR graphs.
                # It returns: reconstructed_hr, latent_lr, latent_hr, decoded_features, decoded_adj.
                reconstructed_hr, latent_lr, latent_hr, decoded_features, decoded_adj = netG(lr, hr)

                # Prepare ground truth HR for eigen-decomposition and discriminator.
                padded_hr = pad_HR_adj(hr, args.padding).to(device)
                _, U_hr = torch.linalg.eigh(padded_hr, UPLO="U")
                U_hr = U_hr.to(device)

                # Create a mask for off-diagonal elements.
                mask = torch.ones_like(reconstructed_hr, dtype=torch.bool).to(device)
                mask.fill_diagonal_(0)

                # Reconstruction loss: compare reconstructed HR with ground truth HR.
                filtered_recon = torch.masked_select(reconstructed_hr, mask)
                filtered_hr = torch.masked_select(hr, mask)
                recon_loss = criterion(filtered_recon, filtered_hr)

                # Latent space loss: enforce consistency between LR and HR latent embeddings.
                if latent_hr is not None:
                    latent_loss = criterion_L1(latent_lr, latent_hr)
                else:
                    latent_loss = 0

                # print(latent_loss)

                # Additional eigen loss: align the learned mapping (from the GSR layer) with the HR eigen-space.
                eigen_loss = criterion(netG.layer.weights, U_hr)

                # print(eigen_loss)

                # Composite generator loss (you can adjust weighting factors as needed).
                composite_loss = args.lmbda * recon_loss + latent_loss + eigen_loss

                # Logging error (using L1 error on the reconstructed HR graph).
                error = cal_error(reconstructed_hr, hr, mask)

                # print(error)

                # print(reconstructed_hr)
                # print(hr)
                # print(mask)

                # Prepare data for discriminator training.
                real_data = reconstructed_hr.detach()
                # Crop padded_hr to match HR dimensions.
                total_length = padded_hr.shape[0]
                middle_length = args.hr_dim
                start_index = (total_length - middle_length) // 2
                end_index = start_index + middle_length
                cropped_hr = padded_hr[start_index:end_index, start_index:end_index]

                # Train discriminator.
                if stop_gan_mae is None or best_mae >= stop_gan_mae:
                    fake_data = gaussian_noise_layer(cropped_hr, args)
                    d_real = netD(real_data)
                    d_fake = netD(fake_data)

                    # print(d_real)
                    # print(d_fake)

                    dc_loss_real = bce_loss(d_real, torch.ones_like(d_real))
                    dc_loss_real = torch.clamp(dc_loss_real, 1e-7, 1 - 1e-7)
                    dc_loss_fake = bce_loss(d_fake, torch.zeros_like(d_real))
                    dc_loss_fake = torch.clamp(dc_loss_fake, 1e-7, 1 - 1e-7)
                    dc_loss = dc_loss_real + dc_loss_fake
                    dc_loss.backward()
                    optimizerD.step()

                # Update generator.
                if stop_gan_mae is None or best_mae >= stop_gan_mae:
                    d_fake = netD(gaussian_noise_layer(cropped_hr, args))
                    gen_adv_loss = bce_loss(d_fake, torch.ones_like(d_fake))
                    generator_loss = gen_adv_loss + composite_loss
                else:
                    generator_loss = composite_loss

                generator_loss.backward()
                optimizerG.step()

                epoch_loss.append(generator_loss.item())
                epoch_error.append(error.item())

            all_epochs_loss.append(np.mean(epoch_loss))

            # Evaluate on test data if provided.
            if test_adj is not None and test_ground_truth is not None:
                test_error = test_gan(netG, test_adj, test_ground_truth, args)
                if test_error < best_mae:
                    best_mae = test_error
                    early_stop_count = 0
                    best_model = copy.deepcopy(netG)
                elif early_stop_count >= early_stop_patient:
                    if test_adj is not None and test_ground_truth is not None:
                        test_error = test_gan(best_model, test_adj, test_ground_truth, args)
                        print(f"Val Error: {test_error:.6f}")
                    return best_model
                else:
                    early_stop_count += 1

                tepoch.set_postfix(
                    train_loss=np.mean(epoch_loss),
                    train_error=np.mean(epoch_error),
                    test_error=test_error,
                )
            else:
                tepoch.set_postfix(
                    train_loss=np.mean(epoch_loss),
                    train_error=np.mean(epoch_error)
                )

    if not best_model:
        best_model = copy.deepcopy(netG)

    if test_adj is not None and test_ground_truth is not None:
        test_error = test_gan(netG, test_adj, test_ground_truth, args)
        print(f"Val Error: {test_error:.6f}")

    return best_model

def test_gan(model, test_adj, test_labels, args):
    """
    Test the GAN model using the updated generator.
    """
    model.eval()
    test_error = []
    g_t = []
    base_mask = torch.triu(torch.ones(args.hr_dim, args.hr_dim), diagonal=1).bool().to(device)

    for lr, hr in zip(test_adj, test_labels):
        all_zeros_lr = not np.any(lr)
        all_zeros_hr = not np.any(hr)
        with torch.no_grad():
            if (not all_zeros_lr) and (not all_zeros_hr):
                lr = torch.from_numpy(lr).type(torch.FloatTensor).to(device)
                np.fill_diagonal(hr, 1)
                hr = torch.from_numpy(hr).type(torch.FloatTensor).to(device)
                # Forward pass now requires both LR and HR.
                preds, _, _, _, _ = model(lr, hr)
                preds = preds.to(device)
                mask = torch.ones_like(preds, dtype=torch.bool).to(device)
                mask.fill_diagonal_(0)
                error = cal_error(preds, hr, mask)
                g_t.append(hr.flatten())
                test_error.append(error.item())
    return np.mean(test_error)
