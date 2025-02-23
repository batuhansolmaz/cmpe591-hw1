import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os

class PushDataset(Dataset):
    def __init__(self, data_dir):
        self.images_before = []  
        self.images_after = []   
        self.positions = []      
        self.actions = []        
        
        for i in range(4):
            self.images_before.append(torch.load(f'{data_dir}/imgs_before_{i}.pt'))
            self.images_after.append(torch.load(f'{data_dir}/imgs_after_{i}.pt'))
            self.positions.append(torch.load(f'{data_dir}/positions_{i}.pt'))
            self.actions.append(torch.load(f'{data_dir}/actions_{i}.pt'))
        
        self.images_before = torch.cat(self.images_before)
        self.images_after = torch.cat(self.images_after)
        self.positions = torch.cat(self.positions)
        self.actions = torch.cat(self.actions)
    
    def __len__(self):
        return len(self.actions)
    
    def __getitem__(self, idx):
        if not hasattr(self, 'reconstruction_mode') or not self.reconstruction_mode:
            return (self.images_before[idx].float() / 255.0,
                    self.actions[idx],
                    self.positions[idx])
        else:
            return (self.images_before[idx].float() / 255.0,
                    self.actions[idx],
                    self.positions[idx],
                    self.images_after[idx].float() / 255.0)  # Also normalize target

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.net = nn.Sequential(
            nn.Linear(3 * 128 * 128 + 4, 512),  
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 2)  
        )
        
    def forward(self, img, action):
        x = self.flatten(img)
        action_onehot = torch.zeros(action.size(0), 4, device=action.device)
        action = action.long()
        action_onehot.scatter_(1, action.unsqueeze(1), 1)
        x = torch.cat([x, action_onehot], dim=1)
        return self.net(x)

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Linear(128 * 16 * 16 + 4, 512),
            nn.ReLU(),
            nn.Linear(512, 2)
        )
        
    def forward(self, img, action):
        x = self.conv(img)
        x = x.flatten(1)
        action_onehot = torch.zeros(action.size(0), 4, device=action.device)
        action = action.long()
        action_onehot.scatter_(1, action.unsqueeze(1), 1)
        x = torch.cat([x, action_onehot], dim=1)
        return self.fc(x)

class ImagePredictor(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.enc1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )  
        
        self.enc2 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )  
        
        self.action_embed = nn.Sequential(
            nn.Linear(4, 128),
            nn.ReLU(),
            nn.Linear(128, 64 * 64)
        )
        
        self.position_embed = nn.Sequential(
            nn.Linear(2, 128),
            nn.ReLU(),
            nn.Linear(128, 64 * 64)
        )
        
        self.dec1 = nn.Sequential(
            nn.Conv2d(130, 128, kernel_size=3, padding=1),  # 128 + 2 channels for action/pos
            nn.BatchNorm2d(128),
            nn.ReLU()
        )  
        
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1), 
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64 + 64, 64, kernel_size=3, padding=1),  
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 3, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, img, action, position):
        x1 = self.enc1(img)  
        x = self.enc2(x1)    
        
        action_onehot = torch.zeros(action.size(0), 4, device=action.device)
        action = action.long()
        action_onehot.scatter_(1, action.unsqueeze(1), 1)
        
        action_feat = self.action_embed(action_onehot)
        action_feat = action_feat.view(-1, 1, 64, 64)
        
        position_feat = self.position_embed(position)
        position_feat = position_feat.view(-1, 1, 64, 64)
        
        x = torch.cat([x, action_feat, position_feat], dim=1)  
        x = self.dec1(x)  
        
        x = self.dec2[0:3](x) 
        
        x = torch.cat([x, x1], dim=1)  
        
        x = self.dec2[3:](x) 
        
        return x

