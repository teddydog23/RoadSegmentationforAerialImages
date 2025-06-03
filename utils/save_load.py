import torch
import os

def save_model(model, path, epoch, score):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    state = {
        'model_state_dict': model.state_dict(),
        'epoch': epoch,
        'score': score
    }
    torch.save(state, path)

def load_model(model, path, device='cpu'):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model, checkpoint.get('epoch', 0), checkpoint.get('score', None)
