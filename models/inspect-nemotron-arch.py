#!/usr/bin/env python3
"""
Inspect a Nemotron .nemo checkpoint and compare its tensor names
against src/nemotron-arch.h to find:
  - tensors in the checkpoint that have no enum entry (missing)
  - enum entries with no matching checkpoint tensor (orphaned)
  - per-layer tensors: verify layer count and name patterns
  - shape / dtype summary for every tensor

Usage:
    python models/inspect-nemotron-arch.py models/nemotron/nemotron-3.5-asr-streaming-0.6b.nemo
"""

import argparse
import re
import sys
from pathlib import Path

import torch


# ---------------------------------------------------------------------------
# Parse the C++ header to extract enum entries and tensor name map entries
# ---------------------------------------------------------------------------

def parse_arch_header(header_path: str):
    """Return (enum_names: list[str], name_map: dict[str, str]) from the .h file."""
    with open(header_path) as f:
        text = f.read()

    # Extract enum values
    enum_block = re.search(
        r'enum\s+nemotron_tensor\s*\{(.*?)\};', text, re.DOTALL
    )
    if not enum_block:
        raise SystemExit("Could not find enum nemotron_tensor in header")

    enum_names = [
        line.strip().rstrip(',')
        for line in enum_block.group(1).splitlines()
        if line.strip() and not line.strip().startswith('//')
    ]

    # Extract NEMOTRON_TENSOR_NAMES map entries
    # Pattern: {ENUM_VALUE, "tensor.name"},
    name_map = {}
    map_block = re.search(
        r'NEMOTRON_TENSOR_NAMES\s*=\s*\{(.*?)\};', text, re.DOTALL
    )
    if not map_block:
        raise SystemExit("Could not find NEMOTRON_TENSOR_NAMES in header")

    for m in re.finditer(
        r'\{(\w+),\s*"([^"]+)"\}', map_block.group(1)
    ):
        enum_key, tensor_name = m.group(1), m.group(2)
        name_map[enum_key] = tensor_name

    # Extract NEMOTRON_TENSOR_INFO map entries
    op_map = {}
    info_block = re.search(
        r'NEMOTRON_TENSOR_INFO\s*=\s*\{(.*?)\};', text, re.DOTALL
    )
    if info_block:
        for m in re.finditer(
            r'\{(\w+),\s*(GGML_OP_\w+)\}', info_block.group(1)
        ):
            op_map[m.group(1)] = m.group(2)

    return enum_names, name_map, op_map


# ---------------------------------------------------------------------------
# Load the .nemo checkpoint
# ---------------------------------------------------------------------------

def load_nemo_checkpoint(nemo_path: str):
    """Load a .nemo tar and return the PyTorch state dict."""
    import tarfile

    nemo_path = Path(nemo_path)
    if not nemo_path.exists():
        raise SystemExit(f"File not found: {nemo_path}")

    print(f"Opening .nemo archive: {nemo_path}")

    with tarfile.open(nemo_path, 'r:*') as tar:
        # Find the checkpoint file inside
        ckpt_members = [
            m for m in tar.getmembers()
            if m.name.endswith('.ckpt') or m.name.endswith('.pt')
        ]
        if not ckpt_members:
            # List what we actually found
            print("Available members in .nemo:")
            for m in tar.getmembers():
                print(f"  {m.name}")
            raise SystemExit("No .ckpt or .pt file found in .nemo archive")

        ckpt_name = ckpt_members[0].name
        print(f"Found checkpoint: {ckpt_name}")

        f = tar.extractfile(ckpt_members[0])
        if f is None:
            raise SystemExit(f"Could not extract {ckpt_name}")

        # Load the checkpoint
        print("Loading checkpoint (this may take a moment)...")
        checkpoint = torch.load(f, map_location='cpu', weights_only=False)

    # The state dict may be nested under a key like 'state_dict' or 'model_ema'
    if isinstance(checkpoint, dict):
        for key in ('state_dict', 'model', 'model_state_dict'):
            if key in checkpoint:
                print(f"State dict found under key: '{key}'")
                return checkpoint[key]
        # If no known key, maybe it IS the state dict
        print("No known wrapper key — treating entire dict as state dict")
        return checkpoint
    return checkpoint


# ---------------------------------------------------------------------------
# Match & compare
# ---------------------------------------------------------------------------

def expand_template(template: str, layer_count: int):
    """Expand a %d template into concrete names for each layer."""
    if '%d' not in template:
        return [template]
    return [template.replace('%d', str(i)) for i in range(layer_count)]


def detect_layer_count(checkpoint_names: set, template: str) -> int:
    """Detect how many layers exist by finding the max index in checkpoint."""
    if '%d' not in template:
        return 0
    # Build a regex from the template: "encoder.layers.%d.foo" -> r"encoder\.layers\.(\d+)\.foo"
    pattern = template.replace('%d', r'(\d+)').replace('.', r'\.')
    pattern = re.escape(template.replace('%d', '(%d)'))
    # Simpler: just match the prefix before %d
    prefix = template.split('%d')[0]
    suffix = template.split('%d')[1]
    regex = re.escape(prefix) + r'(\d+)' + re.escape(suffix)

    max_idx = -1
    for name in checkpoint_names:
        m = re.match(regex, name)
        if m:
            idx = int(m.group(1))
            max_idx = max(max_idx, idx)
    return max_idx + 1 if max_idx >= 0 else 0


def main():
    parser = argparse.ArgumentParser(
        description='Inspect Nemotron .nemo checkpoint vs. nemotron-arch.h'
    )
    parser.add_argument('nemo', help='Path to .nemo file')
    parser.add_argument(
        '--header',
        default='src/nemotron-arch.h',
        help='Path to the arch header (default: src/nemotron-arch.h)',
    )
    args = parser.parse_args()

    # Parse header
    print("=" * 72)
    print("Parsing arch header...")
    enum_names, name_map, op_map = parse_arch_header(args.header)
    print(f"  Enum entries:      {len(enum_names)}")
    print(f"  Name map entries:  {len(name_map)}")
    print(f"  Op map entries:    {len(op_map)}")

    # Check enum ↔ name map consistency
    mismatches = []
    for en in enum_names:
        if en not in name_map:
            mismatches.append(f"  ENUM '{en}' has no entry in NEMOTRON_TENSOR_NAMES")
    for en in name_map:
        if en not in enum_names:
            mismatches.append(f"  NAME_MAP key '{en}' is not in enum")
    for en in op_map:
        if en not in enum_names:
            mismatches.append(f"  OP_MAP key '{en}' is not in enum")

    if mismatches:
        print("\n⚠ Internal header inconsistencies:")
        for m in mismatches:
            print(m)
    else:
        print("  Header internal consistency: OK")

    # Load checkpoint
    print("\n" + "=" * 72)
    print("Loading .nemo checkpoint...")
    state_dict = load_nemo_checkpoint(args.nemo)
    checkpoint_names = set(state_dict.keys())
    print(f"  Total tensors in checkpoint: {len(checkpoint_names)}")

    # Separate templates from non-templates
    template_entries = {k: v for k, v in name_map.items() if '%d' in v}
    static_entries = {k: v for k, v in name_map.items() if '%d' not in v}

    print(f"\n  Static (non-layer) name map entries: {len(static_entries)}")
    print(f"  Template (per-layer) name map entries: {len(template_entries)}")

    # Detect layer count from first template
    layer_count = 0
    if template_entries:
        first_template = list(template_entries.values())[0]
        layer_count = detect_layer_count(checkpoint_names, first_template)
        print(f"  Detected encoder layer count: {layer_count}")

    # Expand all templates and build expected set
    expected_names = set()
    for en, name in static_entries.items():
        expected_names.add(name)
    for en, template in template_entries.items():
        for expanded in expand_template(template, layer_count):
            expected_names.add(expanded)

    # Compare
    print("\n" + "=" * 72)
    print("Comparison results")
    print("=" * 72)

    missing_in_header = checkpoint_names - expected_names
    orphaned_in_header = expected_names - checkpoint_names

    if missing_in_header:
        print(f"\n⚠ Tensors in checkpoint but NOT in header ({len(missing_in_header)}):")
        for name in sorted(missing_in_header):
            shape = tuple(int(s) for s in state_dict[name].shape)
            dtype = str(state_dict[name].dtype).replace('torch.', '')
            print(f"  {name}")
            print(f"    shape: {shape}, dtype: {dtype}")
    else:
        print("\n✓ All checkpoint tensors are covered by the header")

    if orphaned_in_header:
        print(f"\n⚠ Entries in header but NOT in checkpoint ({len(orphaned_in_header)}):")
        for name in sorted(orphaned_in_header):
            # Find which enum key maps to this
            for en, tmpl in name_map.items():
                if tmpl == name:
                    print(f"  {name}  (enum: {en})")
                    break
    else:
        print("✓ All header entries have matching checkpoint tensors")

    # Summary table
    print("\n" + "=" * 72)
    print("Full tensor listing (checkpoint)")
    print("=" * 72)
    print(f"{'Name':<70} {'Shape':<30} {'Dtype':<8} {'In Header'}")
    print("-" * 140)
    for name in sorted(checkpoint_names):
        shape = tuple(int(s) for s in state_dict[name].shape)
        dtype = str(state_dict[name].dtype).replace('torch.', '')
        in_header = '✓' if name in expected_names else '✗ MISSING'
        print(f"{name:<70} {str(shape):<30} {dtype:<8} {in_header}")

    # Print layer tensor breakdown
    if layer_count > 0:
        print("\n" + "=" * 72)
        print(f"Per-layer tensor count: {len(template_entries)} templates × {layer_count} layers")
        print(f"  Total per-layer tensors: {len(template_entries) * layer_count}")
        print(f"  Static tensors: {len(static_entries)}")
        print(f"  Grand expected total: {len(expected_names)}")

    print("\nDone.")


if __name__ == '__main__':
    main()
