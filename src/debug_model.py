import torch
import torchvision.models as models
from pathlib import Path
from collections import OrderedDict

def debug_state_dict(model_path: str) -> None:
    """Debug state dict loading by printing key mappings."""
    
    # Load state dict
    print(f"\nLoading state dict from: {model_path}")
    state_dict = torch.load(model_path)
    
    # Initialize a new DenseNet121
    print("\nInitializing new DenseNet121")
    model = models.densenet121(pretrained=False)
    
    # Print original keys
    print("\nOriginal state dict keys:")
    for k in list(state_dict.keys())[:5]:
        print(f"  {k}")
    print("  ...")
    
    # Create mapping
    print("\nCreating key mapping:")
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith('backbone.0.'):
            new_key = k.replace('backbone.0.', '')
            new_state_dict[new_key] = v
            print(f"  {k} -> {new_key}")
    
    # Print model's expected keys
    print("\nModel's expected keys:")
    for k in list(model.state_dict().keys())[:5]:
        print(f"  {k}")
    print("  ...")
    
    # Check shapes
    print("\nComparing tensor shapes:")
    for k in list(new_state_dict.keys())[:5]:
        if k in model.state_dict():
            print(f"  {k}: {new_state_dict[k].shape} -> {model.state_dict()[k].shape}")
    
    # Try loading
    print("\nAttempting to load state dict...")
    try:
        model.load_state_dict(new_state_dict)
        print("Successfully loaded state dict!")
    except Exception as e:
        print(f"Error loading state dict: {str(e)}")
        
        # Print missing and unexpected keys
        model_keys = set(model.state_dict().keys())
        dict_keys = set(new_state_dict.keys())
        
        missing = model_keys - dict_keys
        unexpected = dict_keys - model_keys
        
        if missing:
            print("\nMissing keys:")
            for k in missing:
                print(f"  {k}")
        
        if unexpected:
            print("\nUnexpected keys:")
            for k in unexpected:
                print(f"  {k}")

if __name__ == '__main__':
    debug_state_dict('Models/DenseNet121.pt')