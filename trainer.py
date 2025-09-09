import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch_ecg.models.unets.ecg_unet import ECG_UNet
import wfdb

# --- CONFIG ---
LEAD = "ii"  # Change to your preferred lead
DATA_DIR = "data/ludb"
BATCH_SIZE = 8
EPOCHS = 20
LR = 1e-3

# --- DATASET ---
class LUDBDataset(Dataset):
    def __init__(self, data_dir, lead):
        self.data_dir = data_dir
        self.lead = lead
        self.records = [f.split(".")[0] for f in os.listdir(data_dir) if f.endswith(".hea")]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        # Load signal
        rec = wfdb.rdrecord(os.path.join(self.data_dir, record))
        fs = int(rec.fs)
        lead_idx = rec.sig_name.index(self.lead.upper())
        sig = rec.p_signal[:, lead_idx].astype("float32")
        x = (sig - sig.mean()) / (sig.std() + 1e-6)
        x = torch.from_numpy(x).float().unsqueeze(0)  # [1, T]

        # Load annotation and create mask
        ann_path = os.path.join(self.data_dir, f"{record}.{self.lead}")
        mask = self.parse_annotation(ann_path, len(x[0]))
        mask = torch.from_numpy(mask).long()  # [T]

        return x, mask

    def parse_annotation(self, ann_path, length):
        # You must implement this function to parse .ii files and return a mask of shape [length]
        # Each sample: 0=BG, 1=P, 2=QRS, 3=T
        mask = np.zeros(length, dtype=np.int64)
        # Example: fill mask using annotation intervals
        # for each wave: mask[start:end] = class_id
        # You need to parse the .ii file format here
        return mask

# --- TRAINING ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ECG_UNet(in_channels=1, num_classes=4, base_filters=16).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = torch.nn.CrossEntropyLoss()

dataset = LUDBDataset(DATA_DIR, LEAD)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for x, mask in dataloader:
        x, mask = x.to(device), mask.to(device)
        optimizer.zero_grad()
        out = model(x)  # [B, 4, T]
        out = out.permute(0, 2, 1).reshape(-1, 4)  # [B*T, 4]
        mask = mask.view(-1)  # [B*T]
        loss = criterion(out, mask)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss/len(dataloader):.4f}")

torch.save(model.state_dict(), "ludb_unet_checkpoint.pth")
print("Training complete. Checkpoint saved as ludb_unet_checkpoint.pth")