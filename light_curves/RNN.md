---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.15.2
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

```python
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch 
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import random

try:
    import wandb
except:
    !python3 -m pip install wandb
    import wandb

import warnings
warnings.filterwarnings('ignore')
torch.cuda.device_count()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

```

### Read in Kauffmann Data:

```python

r=0
redshifts, o3lum,o3corr, bpt1,bpt2, rml50, rmu, con, d4n,hda, vdisp = [],[],[],[],[],[],[],[],[],[],[]
with open("data/agn.dat_dr4_release.v2", 'r') as file:
    for line in file:
        parts = line.split()  # Splits the line into parts
        redshifts.append(float(parts[5]))
        o3lum.append(float(parts[6]))
        o3corr.append(float(parts[7]))
        bpt1.append(float(parts[8]))
        bpt2.append(float(parts[9]))
        rml50.append(float(parts[10]))
        rmu.append(float(parts[11]))
        con.append(float(parts[12]))
        d4n.append(float(parts[13]))
        hda.append(float(parts[14]))
        vdisp.append(float(parts[15]))
        r+=1
redshifts, o3lum,o3corr, bpt1,bpt2, rml50, rmu, con, d4n,hda, vdisp = np.array(redshifts), np.array(o3lum),np.array(o3corr), np.array(bpt1),np.array(bpt2), np.array(rml50), np.array(rmu), np.array(con), np.array(d4n),np.array(hda), np.array(vdisp)

df_lc = pd.read_parquet('data/df_lc_kauffmann.parquet')
bands_inlc = ['zr','zi','zg','W1','W2']
numobjs = len(df_lc.index.get_level_values('objectid')[:].unique())

```

### Preprocessing on a smaller subset for testing:

```python
def find_max_length(df_lc, bands_inlc=['zr', 'zi', 'zg','W1','W2'], subset_size=100, padding_value=-1):
    """
    Determine the maximum length of time series in a subset of the data.
    
    Parameters:
    - df_lc: DataFrame with light curve data.
    - bands_inlc: List of bands to include in the analysis (default: ['zr', 'zi', 'zg']).
    - subset_size: Number of objects to use for determining the max length (default: 100).
    - padding_value: Value used for padding sequences (default: -1).

    Returns:
    - max_len: The maximum length of the time series in the subset.
    """
    objids = df_lc.index.get_level_values('objectid').unique()[:subset_size]
    max_len = 0
    for obj in objids:
        singleobj = df_lc.loc[obj]
        label = singleobj.index.unique('label')[0]
        for band in bands_inlc:
            try:
                band_lc = singleobj.xs((label, band), level=('label', 'band'))
                max_len = max(max_len, len(band_lc))
            except KeyError:
                continue
    return max_len

def unify_lc_for_rnn_multi_band(df_lc, redshifts, max_len, bands_inlc=['zr', 'zi', 'zg', 'W1', 'W2'], padding_value=-1):
    objids = df_lc.index.get_level_values('objectid').unique()
    if isinstance(redshifts, np.ndarray):
        redshifts = dict(zip(objids, redshifts))
    padded_times_all, padded_fluxes_all = [], []
    
    for obj in objids:
        redshift = redshifts.get(obj, None)
        if redshift is None:
            continue
        singleobj = df_lc.loc[obj]
        label = singleobj.index.unique('label')[0]
        bands = singleobj.index.get_level_values('band').unique()
        
        if len(np.intersect1d(bands, bands_inlc)) == len(bands_inlc):
            obj_times, obj_fluxes = [], []
            for band in bands_inlc:
                if (label, band) in singleobj.index:
                    band_lc = singleobj.xs((label, band), level=('label', 'band'))
                    band_lc_clean = band_lc[band_lc.index.get_level_values('time') < 65000]
                    x = np.array(band_lc_clean.index.get_level_values('time'))
                    y = np.array(band_lc_clean.flux)
                    
                    sorted_indices = np.argsort(x)
                    x = x[sorted_indices]
                    y = y[sorted_indices]
                    
                    if len(x) > max_len:
                        x = x[:max_len]
                        y = y[:max_len]
                    
                    if len(x) > 0:
                        padded_x = np.pad(x, (0, max_len - len(x)), 'constant', constant_values=(padding_value,))
                        padded_y = np.pad(y, (0, max_len - len(y)), 'constant', constant_values=(padding_value,))
                        obj_times.append(padded_x)
                        obj_fluxes.append(padded_y)
                    else:
                        break
            if len(obj_times) == len(bands_inlc):
                padded_times_all.append(obj_times)
                padded_fluxes_all.append(obj_fluxes)
    
    padded_times_all = np.array(padded_times_all)
    padded_fluxes_all = np.array(padded_fluxes_all)
    return padded_times_all, padded_fluxes_all

# Set the maximum length for padding
max_len = find_max_length(df_lc, bands_inlc=bands_inlc, subset_size=300)
print(max_len)

# Select a random subset of object IDs
padding_value=-1
subset_size = 10000
random_obj_ids = random.sample(list(df_lc.index.get_level_values('objectid').unique()), subset_size)

# Extract the data for the selected objects
df_lc_subset = df_lc.loc[random_obj_ids]
redshifts_subset = {obj_id: redshifts[obj_id] for obj_id in random_obj_ids}

# Use the modified function to prepare the data
padded_times, padded_fluxes = unify_lc_for_rnn_multi_band(df_lc_subset, redshifts_subset, max_len)

# Convert the data to PyTorch tensors
padded_times_tensor = torch.tensor(padded_times, dtype=torch.float32)
padded_fluxes_tensor = torch.tensor(padded_fluxes, dtype=torch.float32)

# Combine time and flux into a single input tensor
input_tensor = torch.stack((padded_times_tensor, padded_fluxes_tensor), dim=-1)  # (num_samples, num_bands, seq_len, 2)
target_tensor = padded_fluxes_tensor  # (num_samples, num_bands, seq_len)

# Print shapes for debugging
print(f"padded_times_tensor shape: {padded_times_tensor.shape}")
print(f"padded_fluxes_tensor shape: {padded_fluxes_tensor.shape}")
print(f"input_tensor shape: {input_tensor.shape}")
print(f"target_tensor shape: {target_tensor.shape}")

```

### define a RNN model for gap filling and train:

```python
class MultiBandTimeSeriesRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_bands, num_layers=2):
        super(MultiBandTimeSeriesRNN, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)  # Output is the flux value
    
    def forward(self, x):
        batch_size, num_bands, seq_len, _ = x.size()
        x = x.view(batch_size * num_bands, seq_len, -1)  # Combine batch and bands dimensions
        h_0 = torch.zeros(self.lstm.num_layers, batch_size * num_bands, self.lstm.hidden_size).to(x.device)
        c_0 = torch.zeros(self.lstm.num_layers, batch_size * num_bands, self.lstm.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h_0, c_0))
        out = self.fc(out)
        out = out.view(batch_size, num_bands, seq_len, -1)  # Reshape back to (batch_size, num_bands, seq_len, 1)
        return out
    

# Create a dataset and dataloader
dataset = TensorDataset(input_tensor, target_tensor)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Define the model parameters
input_size = 2  # Time and flux as input
hidden_size = 64
num_bands = len(bands_inlc)  # Number of bands
num_layers = 2

# Instantiate the model
model = MultiBandTimeSeriesRNN(input_size, hidden_size, num_bands, num_layers)
model.to(device)

# Print the model summary
print(model)

```

```python
wandb.finish()
wandb.init(project='lightcurve-RNN')

# Define the model parameters
input_size = 2  # Time and flux as input
hidden_size = 64
num_bands = len(bands_inlc)  # Number of bands
num_layers = 2

# Define the model
class MultiBandTimeSeriesRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_bands, num_layers=2):
        super(MultiBandTimeSeriesRNN, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)  # Output is the flux value
    
    def forward(self, x):
        batch_size, num_bands, seq_len, _ = x.size()
        x = x.view(batch_size * num_bands, seq_len, -1)  # Combine batch and bands dimensions
        h_0 = torch.zeros(self.lstm.num_layers, batch_size * num_bands, self.lstm.hidden_size).to(x.device)
        c_0 = torch.zeros(self.lstm.num_layers, batch_size * num_bands, self.lstm.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h_0, c_0))
        out = self.fc(out)
        out = out.view(batch_size, num_bands, seq_len, -1)  # Reshape back to (batch_size, num_bands, seq_len, 1)
        return out

# Instantiate the model
model = MultiBandTimeSeriesRNN(input_size, hidden_size, num_bands, num_layers)
model.to(device)

# Define loss and optimizer
criterion = nn.MSELoss(reduction='none')  # Use 'none' to apply mask later
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Create a dataset and dataloader
dataset = TensorDataset(input_tensor, target_tensor)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# Initialize variables for best model saving
best_loss = float('inf')
best_model_path = "best_model_checkpoint.pth"

# Training the model
num_epochs = 30
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

wandb.watch(model, log="all")

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for i, (inputs, targets) in enumerate(dataloader):
        # Move data to the device
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Mask the padded values
        mask = inputs[:, :, :, 0] != padding_value
        
        # Forward pass
        outputs = model(inputs)
        
        # Apply the mask to the outputs and targets
        mask_expanded = mask.unsqueeze(-1).expand_as(outputs)
        outputs_masked = outputs[mask_expanded].view(-1)
        targets_masked = targets[mask].view(-1)
        
        loss = criterion(outputs_masked, targets_masked).mean()
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    
    # Log the average loss for the epoch
    avg_loss = running_loss / len(dataloader)
    wandb.log({"epoch": epoch + 1, "loss": avg_loss})
    
    #print(f'Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}')
    
    # Save the best model checkpoint
    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(model.state_dict(), best_model_path)
        wandb.save(best_model_path)

# Save the final model checkpoint
final_model_path = "final_model_checkpoint.pth"
torch.save(model.state_dict(), final_model_path)
wandb.save(final_model_path)

print('Training complete.')

```

### Test on some random objects to see how the trained model works:

```python
valid_data_found = False

while not valid_data_found:
    try:
        random_obj_id = random.choice(df_lc.index.get_level_values('objectid').unique())
        
        # Extract the data for the selected object
        df_lc_single = df_lc.loc[[random_obj_id]]
        redshifts_single = {random_obj_id: redshifts[random_obj_id]}
        
        # Use the existing function to prepare the data for the selected object
        padded_times_single, padded_fluxes_single = unify_lc_for_rnn_multi_band(df_lc_single, redshifts_single, max_len, bands_inlc=bands_inlc)
        
        # Convert the selected object's data to PyTorch tensors
        padded_times_tensor_single = torch.tensor(padded_times_single, dtype=torch.float32)
        padded_fluxes_tensor_single = torch.tensor(padded_fluxes_single, dtype=torch.float32)
        
        # Combine time and flux into a single input tensor
        input_tensor_single = torch.stack((padded_times_tensor_single, padded_fluxes_tensor_single), dim=-1)  # (num_bands, seq_len, 2)
        target_tensor_single = padded_fluxes_tensor_single.unsqueeze(-1)  # (num_bands, seq_len, 1)
        
        input_tensor_single = input_tensor_single.to(device)
        
        # Use the model to make predictions
        model.eval()
        with torch.no_grad():
            predictions_single = model(input_tensor_single)
            predictions_single = predictions_single.squeeze(0).cpu().numpy()  # Remove batch dimension
        
        # Apply the mask to the predictions
        mask = padded_times_tensor_single != padding_value
        predicted_fluxes_masked_single = np.where(mask.cpu().numpy(), predictions_single.squeeze(-1), np.nan)
        
        valid_data_found = True
    except Exception as e:
        print(f"Error encountered: {e}. Retrying with a different object.")

# Colors for different bands
colors = ['#3182bd', '#6baed6', '#9ecae1', '#e6550d', '#fd8d3c']

# Create a figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# Iterate over each band to plot the results
for band in range(len(colors)):
    mask_band = (padded_times_tensor_single[0, band] != padding_value) & (padded_fluxes_tensor_single[0, band] != -1)
    
    # Original time series
    valid_times_original = padded_times_tensor_single[0, band][mask_band]
    valid_fluxes_original = padded_fluxes_tensor_single[0, band][mask_band]
    ax1.plot(valid_times_original, valid_fluxes_original, marker='o', linestyle='', label=bands_inlc[band], color=colors[band])
    ax1.plot(valid_times_original, valid_fluxes_original, linestyle='--', color=colors[band])

    # Predicted time series
    valid_fluxes_predicted = predicted_fluxes_masked_single[0, band][mask_band]
    ax2.plot(valid_times_original, valid_fluxes_predicted, marker='x', linestyle='', label=f'Predicted Data {bands_inlc[band]}', color=colors[band])
    ax2.plot(valid_times_original, valid_fluxes_predicted, linestyle='-', color=colors[band])

# Set labels and title for the original data subplot
ax1.set_xlabel('Time (MJD)', size=15)
ax1.set_ylabel('Flux (mJy)', size=15)
ax1.legend()

# Set labels and title for the predicted data subplot
ax2.set_xlabel('Time (MJD)', size=15)
ax2.set_ylabel('Flux (mJy)', size=15)
ax2.legend()

plt.show()

```

```python

```
