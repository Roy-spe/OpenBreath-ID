"""Standard-library-only release integrity and aggregate-figure checks."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    manifest = json.loads((ROOT / 'RELEASE_MANIFEST.json').read_text(encoding='utf-8'))
    for name, expected in manifest['files'].items():
        path = (ROOT / name).resolve()
        assert path.is_relative_to(ROOT.resolve()) and path.is_file(), name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, name
        if path.suffix == '.py':
            tree = ast.parse(path.read_text(encoding='utf-8'), filename=name)
            if name.startswith('src/openbreath_id/'):
                for node in ast.walk(tree):
                    if isinstance(node, ast.ImportFrom) and node.level == 1:
                        assert (path.parent / (node.module.split('.')[0] + '.py')).is_file(), (name, node.module)
    spec = importlib.util.spec_from_file_location('public_figure', ROOT / 'scripts/figure_duration.py')
    figure = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(figure)
    checks = figure.audit(figure.TEX.read_text(encoding='utf-8'))
    print(json.dumps({'status': 'PASS', 'files_verified': len(manifest['files']),
                      'figure_conditions': checks['saved_conditions'],
                      'figure_cells': checks['heatmap_cells'], 'new_experiments': False}, indent=2))


if __name__ == '__main__':
    main()
