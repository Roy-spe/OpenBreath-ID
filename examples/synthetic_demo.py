"""Data-free shape/parameter demonstration; no learned biometric performance."""
from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

import numpy as np
import torch
from openbreath_id.neural_data import preprocess_window
from openbreath_id.neural_models import (
    V3StackedChannelEncoder, V3SharedTwoTowerEncoder, V3BilateralInteractionEncoder,
)


def main():
    torch.manual_seed(2027)
    torch.set_num_threads(1)
    rng = np.random.default_rng(2027)
    time = np.arange(180) / 6.0
    windows = []
    for phase in np.linspace(0, 1, 4):
        raw = np.column_stack((np.sin(2 * np.pi * 0.23 * time + phase),
                              -0.8 * np.sin(2 * np.pi * 0.23 * time + phase + 0.12)))
        raw += rng.normal(0, 0.02, raw.shape)
        windows.append(preprocess_window(raw, normalization='shared_robust', canonicalize_polarity=True))
    batch = torch.from_numpy(np.stack(windows)).float()
    systems = [
        ('stacked', V3StackedChannelEncoder(256, 72), 2156160),
        ('shared_two_tower', V3SharedTwoTowerEncoder(256, 128, 72), 2182560),
        ('bie', V3BilateralInteractionEncoder(256, 128, 40), 2187760),
    ]
    report = {'synthetic_only': True, 'trained_weights': False, 'biometric_performance_evaluated': False, 'models': []}
    with torch.inference_mode():
        for name, model, expected in systems:
            model.eval()
            count = sum(p.numel() for p in model.parameters())
            embedding = model(batch)
            norms = torch.linalg.vector_norm(embedding, dim=1)
            assert count == expected, (name, count)
            assert list(embedding.shape) == [4, 256]
            assert torch.isfinite(embedding).all()
            assert torch.allclose(norms, torch.ones_like(norms), atol=1e-6)
            report['models'].append({'name': name, 'parameters': count, 'output_shape': list(embedding.shape), 'unit_norm': True})
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
