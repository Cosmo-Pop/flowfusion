# train_from_scratch.py

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader
import sys

from symplectic_flow import SymplecticFlowModel, create_symplectic_flow_model
from diffusion import PopulationModelDiffusionConditional

def main():
    # --- Configuration ---
    try:
        ORIGINAL_MODEL_PATH = "/Users/gurjeetjagwani/Desktop/old_pop-cosmos/pop-cosmos/uncertainty/uncertainty_1000_VPSDE_256_5_no_zero_points.pt"
        DATA_PATH = "/Users/gurjeetjagwani/Desktop/old_pop-cosmos/pop-cosmos/uncertainty/COSMOS_noise_model_training_data_no_zero_points.npy"
        SAVE_PATH = "./from_scratch_symplectic_model.pt"
    except FileNotFoundError: print("Please update file paths."); sys.exit(1)

    DEVICE, LR, NUM_EPOCHS, BATCH_SIZE = "cpu", 1e-4, 120, 1024
    print(f"--- Training from Scratch (Rectified Flow) on {DEVICE} ---")

    teacher_for_init = torch.load(ORIGINAL_MODEL_PATH, map_location=DEVICE, weights_only=False)
    student_model = create_symplectic_flow_model(teacher_for_init).to(DEVICE)
    
    asinh_mags, log_flux_errors = np.load(DATA_PATH)
    dataset = TensorDataset(torch.tensor(log_flux_errors, dtype=torch.float32), torch.tensor(asinh_mags, dtype=torch.float32))
    optimizer = torch.optim.AdamW(student_model.parameters(), lr=LR)
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    for epoch in range(NUM_EPOCHS):
        student_model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
        for x0_batch, c_batch in pbar:
            optimizer.zero_grad()
            q0 = (x0_batch.to(DEVICE) - student_model.shift) / student_model.scale
            c_norm = (c_batch.to(DEVICE) - student_model.conditional_shift) / student_model.conditional_scale
            p0 = torch.randn_like(q0)
            z_q, z_p = torch.randn_like(q0), torch.randn_like(q0)
            
            start_state = torch.cat([q0, p0], dim=1)
            end_state = torch.cat([z_q, z_p], dim=1)
            v_target = end_state - start_state
            
            t = torch.rand(x0_batch.shape[0], device=DEVICE).view(-1, 1)
            state_t = (1 - t) * start_state + t * end_state
            
            v_pred = student_model.model(t.squeeze(), state_t, c_norm)
            loss = F.mse_loss(v_pred, v_target)
            
            loss.backward()
            optimizer.step()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    torch.save(student_model.state_dict(), SAVE_PATH)
    print(f"Training complete. Model saved to {SAVE_PATH}")
    torch.save(student_model, '/Users/gurjeetjagwani/Desktop/old_pop-cosmos/pop-cosmos/uncertainty/trained_symplectic.pt')

if __name__ == "__main__":
    main()