import os
import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


# 1. READ AUDIO + MFCC FEATURES


def extract_mfcc(path, sr=16000, n_mfcc=40): # Extract MFCC features from audio file
    y, sr = librosa.load(path, sr=sr) # Load audio
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc) # Compute MFCCs
    return mfcc.astype(np.float32) # Return MFCCs as float32 


# 2. DATA AUGMENTATIONS for dataset


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


# DATASET CLASS


def pad_or_trim(mfcc, max_len=200): # Pad or trim MFCC to fixed length
    T = mfcc.shape[-1] # Current time frames

    if T < max_len:
        # pad with zeros
        pad_width = max_len - T
        mfcc = np.pad(mfcc, ((0,0),(0,pad_width)), mode='constant') # pad on time axis
    else:
        # trim
        mfcc = mfcc[:, :max_len]

    return mfcc
class AudioDataset(Dataset):
    def __init__(self, file_paths, labels, samples_per_file=3): # Dataset with augmentations
        self.file_paths = file_paths # list of audio file paths
        self.labels = labels # corresponding labels
        self.samples_per_file = samples_per_file # augmentations per file
        self.n_files = len(self.file_paths) # number of files

    def __len__(self):
        return self.n_files * self.samples_per_file # total samples

    def __getitem__(self, idx):
        # Determine which file and augmentation
        file_idx = idx % self.n_files # file index
        path = self.file_paths[file_idx] # file path
        label = self.labels[file_idx] # label

        # Load raw audio
        y, sr = librosa.load(path, sr=16000) # Load audio

        # Augmentation
        y = augment_audio(y)

        # MFCC + pad/trim + dtype
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        mfcc = pad_or_trim(mfcc, max_len=200)
        mfcc = mfcc[np.newaxis, :, :].astype(np.float32)

        return torch.tensor(mfcc), torch.tensor(label)
    


# 3. CNN CLASSIFIER


class CNNClassifier(nn.Module): # CNN for audio classification
    def __init__(self, num_classes=2):
        super().__init__()

        self.net = nn.Sequential(
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
        x = self.net(x) # CNN layers
        x = x.view(x.size(0), -1) # Flatten
        return self.fc(x) # Classifier output


# 4. TRAIN THE MODEL


def train_model(model, loader, epochs=10, lr=1e-3): # Train CNN model
    device = "cuda" if torch.cuda.is_available() else "cpu" # Use GPU if available
    model = model.to(device) # Move model to device

    criterion = nn.CrossEntropyLoss() # Loss function
    optimizer = optim.Adam(model.parameters(), lr=lr) # Optimizer

    for ep in range(epochs): # Training loop
        model.train() # Set model to training mode
        total_loss = 0 # Initialize loss

        for mfcc, labels in loader: # Iterate over batches
            mfcc, labels = mfcc.to(device), labels.to(device) # Move data to device

            preds = model(mfcc) # Forward pass
            loss = criterion(preds, labels) # Compute loss

            optimizer.zero_grad() # Zero gradients
            loss.backward() # Backpropagation
            optimizer.step() # Update weights

            total_loss += loss.item() # Accumulate loss

        print(f"Epoch {ep+1}/{epochs}  Loss: {total_loss/len(loader):.4f}") # Print average loss

    return model


# 5. PREDICTION FUNCTION


def predict_file(model, filepath):
    device = "cuda" if torch.cuda.is_available() else "cpu" # Use GPU if available
    model = model.to(device) # Move model to device
    model.eval() # Set model to evaluation mode

    # Load audio
    y, sr = librosa.load(filepath, sr=16000)

    # SAME preprocessing as training
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfcc = pad_or_trim(mfcc, max_len=200)
    mfcc = mfcc[np.newaxis, :, :]
    mfcc = mfcc.astype(np.float32)

    # Convert to tensor
    x = torch.tensor(mfcc).unsqueeze(0).to(device)   # shape: (1, 1, 40, 200)

    # Model forward pass
    with torch.no_grad():
        outputs = model(x)
        pred = torch.argmax(outputs, dim=1).item()

    return pred


# 6. MAIN SCRIPT TO TRAIN AND MAKE FINAL PREDICTIONS


if __name__ == "__main__":
    # dataset
    train_files = ["sean.wav", "irishMan.wav","odin.wav","fake2.wav","real3.wav","fake3.wav","real4.wav","fake4.wav","real5.wav","fake5.wav"] # list of training audio files
    train_labels = [0,1,0,1,0,1,0,1,0,1]      # real=0, fake=1

    dataset = AudioDataset(train_files, train_labels) # create dataset
    loader = DataLoader(dataset, batch_size=4, shuffle=True) # create dataloader

    model = CNNClassifier(num_classes=2) # initialize model
    trained_model = train_model(model, loader, epochs=10) # train model

    # final predictions
    resultREAL = predict_file(trained_model, "testReal.wav")
    if resultREAL == 0:
        print("TestReal file is REAL")
    else:
        print("TestReal file is FAKE")
    resultFAKE = predict_file(trained_model, "testfake.wav")
    if resultFAKE == 0:
        print("TestFake file is REAL")
    else:
        print("TestFake file is FAKE")
