"""
Custom TFRecord reader for LIBERO datasets.
Reads flat tf.train.Example format where each example is one episode
with all steps concatenated into flat arrays.

TFRecord format (one Example per episode):
- num_steps: int64 (scalar)
- observation/image_primary: bytes (T * H * W * 3 uint8)
- observation/image_wrist: bytes (T * H * W * 3 uint8)
- observation/proprio: float32 (T * 8)
- observation/timestep: int64 (T)
- action: float32 (T * 7)
- task/language_instruction: bytes (utf-8 string)
- dataset_name: bytes (utf-8 string)
"""
import numpy as np
import tensorflow as tf
import dlimp as dl
from pathlib import Path


def get_libero_dataset_path(data_dir: str, name: str) -> str:
    """Get the path to the LIBERO dataset directory."""
    base_dir = Path(data_dir) / name
    version_dir = base_dir / "1.0.0"

    if version_dir.exists():
        return str(version_dir)
    return str(base_dir)


def _parse_example_dict(raw_record_bytes):
    """Parse a single tf.train.Example into a trajectory dict (Python/numpy)."""
    example = tf.train.Example()
    example.ParseFromString(raw_record_bytes)

    num_steps = int(example.features.feature["num_steps"].int64_list.value[0])
    img_h, img_w, img_c = 256, 256, 3

    # Parse images
    img_primary_data = np.frombuffer(
        example.features.feature["observation/image_primary"].bytes_list.value[0],
        dtype=np.uint8
    ).reshape(-1, img_h, img_w, img_c)[:num_steps]

    img_wrist_data = np.frombuffer(
        example.features.feature["observation/image_wrist"].bytes_list.value[0],
        dtype=np.uint8
    ).reshape(-1, img_h, img_w, img_c)[:num_steps]

    # Parse action: (T, 7)
    action = np.array(
        example.features.feature["action"].float_list.value,
        dtype=np.float32
    ).reshape(-1, 7)[:num_steps]

    # Parse proprio: (T, 8)
    proprio = np.array(
        example.features.feature["observation/proprio"].float_list.value,
        dtype=np.float32
    ).reshape(-1, 8)[:num_steps]

    # Parse timestep: (T,)
    timestep = np.array(
        example.features.feature["observation/timestep"].int64_list.value,
        dtype=np.int64
    )[:num_steps]

    # Language instruction - tile across trajectory length so it's a sequence
    language_instruction = example.features.feature[
        "task/language_instruction"
    ].bytes_list.value[0]
    language_instruction_tiled = np.array([language_instruction] * num_steps, dtype=object)

    # Generate episode flags
    indices = np.arange(num_steps)
    is_first = (indices == 0).astype(np.bool_)
    is_last = (indices == num_steps - 1).astype(np.bool_)
    is_terminal = is_last.copy()
    discount = np.ones(num_steps, dtype=np.float32)
    reward = np.zeros(num_steps, dtype=np.float32)

    return {
        "action": action,
        "observation": {
            "image_primary": img_primary_data,
            "image_wrist": img_wrist_data,
            "proprio": proprio,
            "timestep": timestep,
        },
        "language_instruction": language_instruction_tiled,
        "is_first": is_first,
        "is_last": is_last,
        "is_terminal": is_terminal,
        "discount": discount,
        "reward": reward,
    }


def _libero_generator(tfrecord_path):
    """Generator that yields parsed trajectory dicts from TFRecords."""
    dataset = tf.data.TFRecordDataset(str(tfrecord_path), compression_type=None)
    for raw_record in dataset:
        yield _parse_example_dict(raw_record.numpy())


def _get_output_signature():
    """Return the output signature for the generator."""
    return {
        "action": tf.TensorSpec(shape=(None, 7), dtype=tf.float32),
        "observation": {
            "image_primary": tf.TensorSpec(shape=(None, 256, 256, 3), dtype=tf.uint8),
            "image_wrist": tf.TensorSpec(shape=(None, 256, 256, 3), dtype=tf.uint8),
            "proprio": tf.TensorSpec(shape=(None, 8), dtype=tf.float32),
            "timestep": tf.TensorSpec(shape=(None,), dtype=tf.int64),
        },
        "language_instruction": tf.TensorSpec(shape=(None,), dtype=tf.string),
        "is_first": tf.TensorSpec(shape=(None,), dtype=tf.bool),
        "is_last": tf.TensorSpec(shape=(None,), dtype=tf.bool),
        "is_terminal": tf.TensorSpec(shape=(None,), dtype=tf.bool),
        "discount": tf.TensorSpec(shape=(None,), dtype=tf.float32),
        "reward": tf.TensorSpec(shape=(None,), dtype=tf.float32),
    }


def _wrap_as_dlaset(dataset):
    """Wrap a tf.data.Dataset as a DLataset to enable traj_map and other dlimp methods."""
    if isinstance(dataset, dl.DLataset):
        return dataset
    # Monkey-patch: make the result a DLataset subclass
    result_type = type("DLataset", (dl.DLataset, type(dataset)), dl.DLataset.__dict__.copy())
    dataset.__class__ = result_type
    dataset.is_flattened = False
    return dataset


def load_libero_dataset(data_dir: str, name: str, shuffle: bool = True, num_parallel_reads: int = -1):
    """
    Load a LIBERO dataset from TFRecords using dlimp.

    Reads flat tf.train.Example format and parses into trajectory dicts
    matching the OXE standardization pipeline expected format.
    """
    dataset_path = get_libero_dataset_path(data_dir, name)
    tfrecord_path = Path(dataset_path) / "train.tfrecord"

    if not tfrecord_path.exists():
        raise FileNotFoundError(f"TFRecord not found: {tfrecord_path}")

    # Use from_generator for full control over parsing
    output_signature = _get_output_signature()
    dataset = tf.data.Dataset.from_generator(
        lambda: _libero_generator(tfrecord_path),
        output_signature=output_signature,
    )

    # Wrap as DLataset to enable traj_map and other dlimp methods
    dataset = _wrap_as_dlaset(dataset)

    # Shuffle if requested
    if shuffle:
        dataset = dataset.shuffle(buffer_size=1000)

    return dataset
