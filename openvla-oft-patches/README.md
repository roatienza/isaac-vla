# OpenVLA-OFT Patches

This directory contains patches and utilities for the [openvla-oft](https://github.com/moojink/openvla-oft) repository, specifically for fine-tuning on LIBERO datasets.

## Changes

### `prismatic/vla/datasets/rlds/libero_reader.py`
- Fixed `from_generator` returning `_FlatMapDataset` instead of `DLataset` (missing `traj_map` method)
- Added `_wrap_as_dlaset()` monkey-patch to fix dataset type
- Fixed scalar `language_instruction` (rank-0 tensor) causing `tf.gather` failure by tiling across trajectory length
- Rewrote to parse flat `tf.train.Example` format (one episode per TFRecord)

### `prismatic/vla/datasets/rlds/dataset.py`
- Updated dataset loading logic

### `prismatic/vla/datasets/rlds/oxe/configs.py`
- LIBERO dataset configurations

### `prismatic/vla/datasets/rlds/oxe/transforms.py`
- Data transformation utilities

### `scripts/convert_libero_to_rlds.py`
- Original LIBERO HDF5 to RLDS TFRecord conversion script

### `scripts/convert_libero_to_rlds_v2.py`
- Improved conversion script with better error handling

### `check_tfrecord.py`
- Utility for inspecting TFRecord files

## How to Apply

1. Clone openvla-oft:
```bash
git clone https://github.com/moojink/openvla-oft.git
cd openvla-oft
```

2. Copy patched files over:
```bash
cp /path/to/isaac-vla/openvla-oft-patches/prismatic/vla/datasets/rlds/libero_reader.py prismatic/vla/datasets/rlds/
cp /path/to/isaac-vla/openvla-oft-patches/prismatic/vla/datasets/rlds/dataset.py prismatic/vla/datasets/rlds/
cp /path/to/isaac-vla/openvla-oft-patches/prismatic/vla/datasets/rlds/oxe/*.py prismatic/vla/datasets/rlds/oxe/
cp /path/to/isaac-vla/openvla-oft-patches/scripts/*.py scripts/
cp /path/to/isaac-vla/openvla-oft-patches/check_tfrecord.py .
```

## Reference

- Original repo: https://github.com/moojink/openvla-oft
- Fork with patches: https://github.com/roatienza/openvla-oft
