import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from models import PushDataset, ImagePredictor
import torch.nn.functional as F

def combined_loss(pred, target):
    l1_loss = nn.L1Loss()(pred, target)
    
    # MSE loss for color accuracy
    mse_loss = nn.MSELoss()(pred, target)
    
    # Extra loss term for red channel to focus on the object
    red_loss = nn.MSELoss()(pred[:, 0], target[:, 0]) * 2.0
    
    return l1_loss + mse_loss + red_loss

def train(data_dir='push_data', epochs=50):
    dataset = PushDataset(data_dir)
    dataset.reconstruction_mode = True  
    splits = torch.load('data_splits.pt')
    
    train_dataset = torch.utils.data.Subset(dataset, splits['train_indices'])
    val_dataset = torch.utils.data.Subset(dataset, splits['val_indices'])
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ImagePredictor().to(device)
    criterion = combined_loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)  # AdamW often works better
    
    history = {
        'train_loss': [],
        'val_loss': []
    }
    
    best_val_loss = float('inf')
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch in train_loader:
            imgs_before, actions, positions, imgs_after = [x.to(device) for x in batch]
            
            optimizer.zero_grad(set_to_none=True)  
            
            pred_imgs = model(imgs_before, actions, positions)
            loss = criterion(pred_imgs, imgs_after)
            loss.backward()
            optimizer.step()
            
            print(f'Loss: {loss.item():.4f}')  
            train_loss += loss.item()
        
        train_loss = train_loss / len(train_loader)
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                imgs_before, actions, positions, imgs_after = [x.to(device) for x in batch]
                
                pred_imgs = model(imgs_before, actions, positions)
                loss = criterion(pred_imgs, imgs_after)
                val_loss += loss.item()
        
        val_loss = val_loss / len(val_loader)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        print(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'reconstruction_model.pth')
    
    history_filename = f'reconstruction_losses.pt'
    torch.save(history, history_filename)

def test(data_dir='push_data', model_path='reconstruction_model.pth'):
    dataset = PushDataset(data_dir)
    dataset.reconstruction_mode = True
    splits = torch.load('data_splits.pt')
    
    test_dataset = torch.utils.data.Subset(dataset, splits['test_indices'])
    test_loader = DataLoader(test_dataset, batch_size=32)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ImagePredictor().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    criterion = combined_loss
    test_loss = 0
    
    with torch.no_grad():
        for imgs_before, actions, positions, imgs_after in test_loader:
            imgs_before = imgs_before.to(device)
            actions = actions.to(device)
            positions = positions.to(device)
            imgs_after = imgs_after.to(device)
            
            pred_imgs = model(imgs_before, actions, positions)
            test_loss += criterion(pred_imgs, imgs_after).item()
    
    test_loss = test_loss / len(test_loader)
    print(f'Test Loss: {test_loss:.4f}')
    return test_loss

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'test'], required=True)
    parser.add_argument('--data_dir', default='push_data')
    parser.add_argument('--model_path', default='reconstruction_model.pth')
    args = parser.parse_args()
    
    if args.mode == 'train':
        train(args.data_dir)
    else:
        test(args.data_dir, args.model_path)