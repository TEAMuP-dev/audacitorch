from pathlib import Path

import torch
import json

def save_model(model: torch.jit.ScriptModule, metadata: dict, root_dir: Path):
    root_dir.mkdir(exist_ok=True, parents=True)
    
    # save model and metadata!
    torch.jit.save(model, root_dir / 'model.pt')

    with open(root_dir / 'metadata.json', 'w') as f:
      json.dump(metadata, f)

