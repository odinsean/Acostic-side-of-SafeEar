import os
import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# 1. READ AUDIO + MFCC FEATURES EXTRACTION

def extract_mfcc(path, sr=16000, n_mfcc=40):
    y, sr = librosa.load(path, sr=sr) # Load audio
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc) # Extract MFCC
    return mfcc.astype(np.float32) # Return as float32

# 2. DATA AUGMENTATIONS to DIVERSIFY DATASET

def augment_audio(y):
    # Add white noise
    if np.random.rand() < 0.3:
        noise = np.random.randn(len(y)) * 0.005
        y = y + noise
    
    # Pitch shift
    if np.random.rand() < 0.3:
        y = librosa.effects.pitch_shift(y, sr=16000, n_steps=np.random.uniform(-2, 2))
    
    # Time stretch
    if np.random.rand() < 0.3:
        y = librosa.effects.time_stretch(y, rate=np.random.uniform(0.8, 1.2))

    return y

# PAD/TRIM FUNCTION

def pad_or_trim(mfcc, max_len=200): # Pad or trim MFCC to fixed length
    T = mfcc.shape[-1] # current time frames

    if T < max_len:
        # pad with zeros
        pad_width = max_len - T
        mfcc = np.pad(mfcc, ((0,0),(0,pad_width)), mode='constant')
    else:
        # trim
        mfcc = mfcc[:, :max_len]

    return mfcc

# DATASET CLASS

class AudioDataset(Dataset):
    def __init__(self, file_paths, labels):
        self.file_paths = file_paths # List of audio file paths
        self.labels = labels # Corresponding labels

    def __getitem__(self, idx): # Get item by index
        path = self.file_paths[idx] # Audio file path
        label = self.labels[idx] # Corresponding label
        

        # Load raw audio
        y, sr = librosa.load(path, sr=16000)

        # Augmentation
        y = augment_audio(y)

        
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40) # Extract MFCC
        mfcc = pad_or_trim(mfcc, max_len=200) # Pad/trim to fixed length
        mfcc = mfcc[np.newaxis, :, :]  # Add channel dimension
        mfcc = mfcc.astype(np.float32) # Convert to float32

        return torch.tensor(mfcc), torch.tensor(label) # Return MFCC and label as tensors

    def __len__(self):
        return len(self.file_paths) # Total number of samples
    

# 3. CNN CLASSIFIER

class CNNClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()

        self.net = nn.Sequential( # CNN architecture
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2,2)),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2,2)),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.fc = nn.Linear(64, num_classes) # Final classification layer

    def forward(self, x): # Forward pass
        x = self.net(x) # Pass through CNN
        x = x.view(x.size(0), -1) # Flatten
        return self.fc(x) # Final output

# 4. TRAIN THE MODEL

def train_model(model, loader, epochs=10, lr=1e-3): # Train the CNN model
    device = "cuda" if torch.cuda.is_available() else "cpu" # Use GPU if available
    model = model.to(device) # Move model to device

    criterion = nn.CrossEntropyLoss() # Loss function
    optimizer = optim.Adam(model.parameters(), lr=lr) # Adam Optimizer

    for ep in range(epochs): # Training loop
        model.train() # Set model to training mode
        total_loss = 0 # Initialize total loss

        for mfcc, labels in loader: # Iterate over batches
            mfcc, labels = mfcc.to(device), labels.to(device) # Move data to device

            preds = model(mfcc) # Forward pass
            loss = criterion(preds, labels) # Compute loss

            optimizer.zero_grad() # Zero gradients
            loss.backward() # Backpropagation
            optimizer.step() # Update weights

            total_loss += loss.item() # Accumulate loss

        print(f"Epoch {ep+1}/{epochs}  Loss: {total_loss/len(loader):.4f}") # Print average loss

    return model # Return trained model

# 5. PREDICTION FUNCTION

def predict_file(model, filepath): # Predict if audio file is real or fake
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    # Load audio
    y, sr = librosa.load(filepath, sr=16000)


    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40) # Extract MFCC
    mfcc = pad_or_trim(mfcc, max_len=200) # Pad/trim to fixed length
    mfcc = mfcc[np.newaxis, :, :] # Add channel dimension
    mfcc = mfcc.astype(np.float32) # Convert to float32

    # Convert to tensor
    x = torch.tensor(mfcc).unsqueeze(0).to(device)   # Add batch dimension and move to device

    # Model forward pass
    with torch.no_grad():
        outputs = model(x)
        pred = torch.argmax(outputs, dim=1).item()

    return pred

# IMPLMENTATION of TRAINING AND FINAL PREDICTIONS

if __name__ == "__main__":
    # dataset preparation
    train_files = ["ember.wav", "real.wav"]
    train_labels = [1,0]      # real=0, fake=1

    dataset = AudioDataset(train_files, train_labels) # Create dataset
    loader = DataLoader(dataset, batch_size=4, shuffle=True) # DataLoader

    model = CNNClassifier(num_classes=2) # Initialize model
    trained_model = train_model(model, loader, epochs=10) # Train model
    # Predict on the audios that are not in the dataset
    resultREAL = predict_file(trained_model, "test.wav") 
    if resultREAL == 0:
        print("Test file is REAL")
    else:
        print("Test file is FAKE")
