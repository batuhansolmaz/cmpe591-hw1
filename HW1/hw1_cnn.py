import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from models import PushDataset, CNN

def train(data_dir='push_data', epochs=50):
   
    dataset = PushDataset(data_dir)
    splits = torch.load('data_splits.pt')
    
    train_dataset = torch.utils.data.Subset(dataset, splits['train_indices'])
    val_dataset = torch.utils.data.Subset(dataset, splits['val_indices'])
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CNN().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for imgs, actions, positions in train_loader:
            imgs = imgs.to(device)
            actions = actions.to(device)
            positions = positions.to(device)
            
            optimizer.zero_grad()
            pred_positions = model(imgs, actions)
            loss = criterion(pred_positions, positions)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        train_loss = train_loss / len(train_loader)
        train_losses.append(train_loss)
            
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for imgs, actions, positions in val_loader:
                imgs = imgs.to(device)
                actions = actions.to(device)
                positions = positions.to(device)
                pred_positions = model(imgs, actions)
                val_loss += criterion(pred_positions, positions).item()
        
        val_loss = val_loss / len(val_loader)
        val_losses.append(val_loss)
        
        print(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'cnn_model.pth')
    
    torch.save({
        'train': train_losses,
        'val': val_losses
    }, 'cnn_losses.pt')

def test(data_dir='push_data', model_path='cnn_model.pth'):
    
    dataset = PushDataset(data_dir)
    splits = torch.load('data_splits.pt')
    
    test_dataset = torch.utils.data.Subset(dataset, splits['test_indices'])
    test_loader = DataLoader(test_dataset, batch_size=32)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CNN().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    criterion = nn.MSELoss()
    total_loss = 0
    with torch.no_grad():
        for imgs, actions, positions in test_loader:
            imgs = imgs.to(device)
            actions = actions.to(device)
            positions = positions.to(device)
            
            pred_positions = model(imgs, actions)
            loss = criterion(pred_positions, positions)
            total_loss += loss.item()
    
    avg_loss = total_loss / len(test_loader)
    print(f'Test Loss: {avg_loss:.4f}')
    return avg_loss

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'test'], required=True)
    parser.add_argument('--data_dir', default='push_data')
    parser.add_argument('--model_path', default='cnn_model.pth')
    args = parser.parse_args()
    
    if args.mode == 'train':
        train(args.data_dir)
    else:
        test(args.data_dir, args.model_path) 