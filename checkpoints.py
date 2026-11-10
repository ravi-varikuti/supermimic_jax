"""Atomic, non-pickle checkpoints for array pytrees, restored against a template."""

import io
import json
import os
from pathlib import Path
import tempfile
import zipfile

import jax
import numpy as np


def save_checkpoint(path, components, metadata):
    """Include optimizer states, replay, environment carry and RNGs in components."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    paths_and_leaves, _ = jax.tree_util.tree_flatten_with_path(components)
    leaves = [leaf for _, leaf in paths_and_leaves]
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    os.close(descriptor)
    try:
        specs = []
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
            for index, leaf in enumerate(leaves):
                typed_key = jax.dtypes.issubdtype(getattr(leaf, "dtype", np.dtype(float)), jax.dtypes.prng_key)
                value = jax.random.key_data(leaf) if typed_key else leaf
                if isinstance(value, (int, float, bool)):
                    value = jax.device_put(value)
                value = np.asarray(jax.device_get(value))
                if value.dtype.hasobject:
                    raise TypeError("Checkpoints only support numeric array leaves")
                spec = {"shape": list(value.shape), "dtype": str(value.dtype), "key": bool(typed_key)}
                spec["path"] = jax.tree_util.keystr(paths_and_leaves[index][0])
                if typed_key:
                    spec["key_impl"] = str(jax.random.key_impl(leaf))
                specs.append(spec)
                with archive.open(f"arrays/{index}.npy", "w", force_zip64=True) as stream:
                    np.lib.format.write_array(stream, value, allow_pickle=False)
            archive.writestr("metadata.json", json.dumps({"version": 2, "metadata": metadata, "arrays": specs}))
        with open(temporary, "rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def checkpoint_metadata(path):
    with zipfile.ZipFile(path) as archive:
        info = json.loads(archive.read("metadata.json"))
    if info["version"] not in (1, 2):
        raise ValueError("Unsupported checkpoint version")
    return info["metadata"]


def load_checkpoint(path, template, expected_metadata=None):
    """Recreate static model/optimizer structure from matching config before loading."""
    paths_and_leaves, structure = jax.tree_util.tree_flatten_with_path(template)
    leaves = [leaf for _, leaf in paths_and_leaves]
    restored = []
    with zipfile.ZipFile(path) as archive:
        info = json.loads(archive.read("metadata.json"))
        if info["version"] not in (1, 2):
            raise ValueError("Unsupported checkpoint version")
        if expected_metadata is not None and info["metadata"] != expected_metadata:
            raise ValueError("Checkpoint config, source or motion assets do not match this run")
        if len(leaves) != len(info["arrays"]):
            raise ValueError("Checkpoint tree does not match template")
        for index, (leaf, spec) in enumerate(zip(leaves, info["arrays"])):
            if info["version"] == 2 and spec["path"] != jax.tree_util.keystr(paths_and_leaves[index][0]):
                raise ValueError(f"Checkpoint array {index} has a different tree path")
            typed_key = jax.dtypes.issubdtype(getattr(leaf, "dtype", np.dtype(float)), jax.dtypes.prng_key)
            expected_leaf = jax.random.key_data(leaf) if typed_key else leaf
            if isinstance(expected_leaf, (int, float, bool)):
                expected_leaf = jax.device_put(expected_leaf)
            expected = np.asarray(jax.device_get(expected_leaf))
            if list(expected.shape) != spec["shape"] or str(expected.dtype) != spec["dtype"] or bool(typed_key) != spec["key"]:
                raise ValueError(f"Checkpoint array {index} does not match template")
            value = np.load(io.BytesIO(archive.read(f"arrays/{index}.npy")), allow_pickle=False)
            if list(value.shape) != spec["shape"] or str(value.dtype) != spec["dtype"]:
                raise ValueError(f"Checkpoint array {index} is inconsistent with manifest")
            # Host bookkeeping (including float64 running weights) stays on host.
            value = value if isinstance(leaf, (np.ndarray, np.generic)) else jax.device_put(value)
            if typed_key:
                value = jax.random.wrap_key_data(value, impl=spec["key_impl"])
            restored.append(value)
    return jax.tree.unflatten(structure, restored)
