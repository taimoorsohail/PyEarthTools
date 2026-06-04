# Copyright Commonwealth of Australia, Bureau of Meteorology 2024.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from __future__ import annotations

from typing import Optional, Any, Literal
import os
import shutil
import re

from pathlib import Path
import yaml
import numpy as np
import xarray as xr

import pyearthtools.pipeline
from pyearthtools.utils.initialisation import Dumper, Loader, update_contents
from pyearthtools.utils.initialisation.imports import dynamic_import
from pyearthtools.utils.initialisation.mixin import InitialisationRecordingMixin
from pyearthtools.training.data.datamodule import PipelineDataModule

CONFIG_KEY = "--CONFIG--"
SUFFIX = ".edm"
RESOURCE_DIR_SUFFIX = ".resources"
MANIFEST_KIND = "pyearthtools.training.datamodule_manifest"
MANIFEST_VERSION = 1
INLINE_NDARRAY_LIMIT = 1024
NODE_KIND_KEY = "pet_kind"

SaveMode = Literal["yaml", "manifest", "snapshot", "none"]


def _full_class_name(obj: Any) -> str:
    return f"{type(obj).__module__}.{type(obj).__name__}"


def _clean_hint(hint: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", hint).strip("._")
    return cleaned or "resource"


def _next_resource_path(context: dict[str, Any], hint: str, suffix: str) -> Path:
    context["resource_index"] += 1
    file_name = f"{context['resource_index']:04d}_{_clean_hint(hint)}{suffix}"
    return context["resource_dir"] / file_name


def _serialise_path_reference(value: str | Path, context: dict[str, Any], hint: str) -> dict[str, Any]:
    value_type = "path" if isinstance(value, Path) else "str"
    source = Path(value).expanduser()

    node: dict[str, Any] = {
        NODE_KIND_KEY: "path_reference",
        "value_type": value_type,
        "original": str(value),
    }

    if not source.exists():
        node["storage"] = "literal"
        return node

    node["source"] = {
        "path": str(source),
        "exists": True,
        "is_dir": source.is_dir(),
        "size": None if source.is_dir() else source.stat().st_size,
        "mtime": source.stat().st_mtime,
    }

    if context["mode"] == "snapshot":
        suffix = source.suffix if source.is_file() else ""
        destination = _next_resource_path(context, hint, suffix)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if source.is_dir():
            shutil.copytree(source, destination, dirs_exist_ok=True)
        else:
            shutil.copy2(source, destination)
        node["storage"] = "snapshot"
        node["path"] = destination.relative_to(context["manifest_dir"]).as_posix()
    else:
        node["storage"] = "reference"
        node["path"] = str(source)
    return node


def _serialise_xarray(data: xr.Dataset | xr.DataArray, context: dict[str, Any], hint: str) -> dict[str, Any]:
    if context["manifest_dir"] is None:
        raise ValueError("Manifest and snapshot datamodule saves require a filesystem path.")

    destination = _next_resource_path(context, hint, ".nc")
    destination.parent.mkdir(parents=True, exist_ok=True)
    data.to_netcdf(destination, engine="scipy")

    return {
        NODE_KIND_KEY: "xarray",
        "object_type": "dataarray" if isinstance(data, xr.DataArray) else "dataset",
        "path": destination.relative_to(context["manifest_dir"]).as_posix(),
        "dims": dict(data.sizes),
        "variables": list(getattr(data, "data_vars", [])),
        "name": data.name if isinstance(data, xr.DataArray) else None,
    }


def _serialise_ndarray(array: np.ndarray, context: dict[str, Any], hint: str) -> dict[str, Any]:
    if array.size <= INLINE_NDARRAY_LIMIT:
        return {
            NODE_KIND_KEY: "ndarray_inline",
            "dtype": str(array.dtype),
            "shape": list(array.shape),
            "data": array.tolist(),
        }

    if context["manifest_dir"] is None:
        raise ValueError("Manifest and snapshot datamodule saves require a filesystem path.")

    destination = _next_resource_path(context, hint, ".npy")
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.save(destination, array)
    return {
        NODE_KIND_KEY: "ndarray",
        "dtype": str(array.dtype),
        "shape": list(array.shape),
        "path": destination.relative_to(context["manifest_dir"]).as_posix(),
    }


def _serialise_value(value: Any, context: dict[str, Any], hint: str) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        if isinstance(value, str) and os.path.exists(value):
            return _serialise_path_reference(value, context, hint)
        return value

    if isinstance(value, Path):
        return _serialise_path_reference(value, context, hint)

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, np.ndarray):
        return _serialise_ndarray(value, context, hint)

    if isinstance(value, (xr.Dataset, xr.DataArray)):
        return _serialise_xarray(value, context, hint)

    if isinstance(value, tuple):
        return {NODE_KIND_KEY: "tuple", "items": [_serialise_value(item, context, hint) for item in value]}

    if isinstance(value, list):
        return [_serialise_value(item, context, hint) for item in value]

    if isinstance(value, dict):
        return {str(key): _serialise_value(val, context, f"{hint}_{key}") for key, val in value.items()}

    if isinstance(value, InitialisationRecordingMixin):
        node = {
            NODE_KIND_KEY: "initialised",
            "class": _full_class_name(value),
            "initialisation": _serialise_value(value.initialisation, context, hint),
        }
        if getattr(value, "_property", None) is not None:
            node["property"] = value._property
        return node

    return {
        NODE_KIND_KEY: "yaml_fragment",
        "yaml": yaml.dump(value, Dumper=Dumper, sort_keys=False),
    }


def _deserialise_value(value: Any, base_path: Path | None = None) -> Any:
    if isinstance(value, list):
        return [_deserialise_value(item, base_path=base_path) for item in value]

    if not isinstance(value, dict) or NODE_KIND_KEY not in value:
        if isinstance(value, dict):
            return {key: _deserialise_value(val, base_path=base_path) for key, val in value.items()}
        return value

    kind = value[NODE_KIND_KEY]

    if kind == "tuple":
        return tuple(_deserialise_value(item, base_path=base_path) for item in value["items"])

    if kind == "ndarray_inline":
        return np.asarray(value["data"], dtype=value["dtype"]).reshape(value["shape"])

    if kind == "ndarray":
        if base_path is None:
            raise FileNotFoundError("Cannot resolve ndarray snapshot without a base path.")
        return np.load(base_path / value["path"], allow_pickle=False)

    if kind == "xarray":
        if base_path is None:
            raise FileNotFoundError("Cannot resolve xarray snapshot without a base path.")
        resource_path = base_path / value["path"]
        if value["object_type"] == "dataarray":
            return xr.load_dataarray(resource_path, engine="scipy")
        return xr.load_dataset(resource_path, engine="scipy")

    if kind == "path_reference":
        resolved = value["original"]
        if value["storage"] == "snapshot":
            if base_path is None:
                raise FileNotFoundError("Cannot resolve snapshot path without a base path.")
            resolved = str(base_path / value["path"])
        elif value["storage"] == "reference":
            resolved = value["path"]
        return Path(resolved) if value["value_type"] == "path" else resolved

    if kind == "initialised":
        init_data = _deserialise_value(value["initialisation"], base_path=base_path)
        if not isinstance(init_data, dict):
            raise TypeError(f"Expected a dictionary of init arguments, received {type(init_data)}.")
        import_target = dynamic_import(value["class"])
        obj = import_target(*init_data.pop("__args", []), **init_data)  # type: ignore[misc]
        if "property" in value:
            return getattr(obj, value["property"])
        return obj

    if kind == "yaml_fragment":
        return yaml.load(value["yaml"], Loader)

    raise TypeError(f"Unknown manifest node kind {kind!r}.")


def _manifest_for(datamodule: PipelineDataModule, path: Path, mode: SaveMode) -> dict[str, Any]:
    manifest_path = path.with_suffix(SUFFIX)
    resource_dir = manifest_path.with_name(f"{manifest_path.stem}{RESOURCE_DIR_SUFFIX}")
    resource_dir.mkdir(parents=True, exist_ok=True)

    context = {
        "mode": mode,
        "manifest_dir": manifest_path.parent,
        "resource_dir": resource_dir,
        "resource_index": 0,
    }

    return {
        NODE_KIND_KEY: MANIFEST_KIND,
        "version": MANIFEST_VERSION,
        "mode": mode,
        "resources": resource_dir.name,
        "metadata": {"VERSION": pyearthtools.pipeline.__version__},
        "datamodule": _serialise_value(datamodule, context, "datamodule"),
    }


def _load_manifest(manifest: dict[str, Any], base_path: Path | None = None) -> PipelineDataModule:
    datamodule = _deserialise_value(manifest["datamodule"], base_path=base_path)
    if not isinstance(datamodule, PipelineDataModule):
        raise FileNotFoundError("Loaded manifest did not produce a valid PipelineDataModule.")
    return datamodule


def save(datamodule: PipelineDataModule, path: Optional[str | Path] = None, *, mode: SaveMode = "yaml") -> None | str:
    """
    Save `PipelineDataModule`

    Args:
        datamodule (PipelineDataModule):
            Datamodule to save.
        path (Optional[FILE], optional):
            File to save to. If not given return save str for `mode="yaml"`.
        mode (SaveMode, optional):
            Save mode to use. `yaml` preserves legacy eager full-object yaml dumping,
            `manifest` writes a lightweight manifest plus sidecar payloads for heavy
            in-memory objects, `snapshot` additionally snapshots referenced files into
            the run directory, and `none` skips saving.

    Returns:
        (Union[None, str]):
            If `path` is None, the datamodule in save form, else None.
    """
    if mode == "none":
        return None

    if mode == "yaml":
        datamodule_yaml = yaml.dump(datamodule, Dumper=Dumper, sort_keys=False)

        extra_info: dict[str, Any] = {"VERSION": pyearthtools.pipeline.__version__}

        full_yaml = datamodule_yaml + f"\n{CONFIG_KEY}\n" + yaml.dump(extra_info)

        if path is None:
            return full_yaml

        path = Path(path)

        path.parent.mkdir(parents=True, exist_ok=True)
        path = path.with_suffix(SUFFIX)

        with open(str(path), "w") as file:
            file.write(full_yaml)
        return None

    if path is None:
        raise ValueError(f"Save mode {mode!r} requires a filesystem path.")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = path.with_suffix(SUFFIX)
    manifest = _manifest_for(datamodule, path, mode)
    manifest_contents = yaml.safe_dump(manifest, sort_keys=False)
    with open(manifest_path, "w") as file:
        file.write(manifest_contents)
    return None


def load(stream: str | Path, **kwargs: Any) -> PipelineDataModule:
    """
    Load `Datamodule` config

    Args:
        stream (Union[str, Path]):
            File or dump to load
        kwargs (Any):
            Updates to default values include in the config.

    Returns:
        (pyearthtools.pipeline.Pipeline):
            Loaded Pipeline
    """
    contents = None
    stream_path: Path | None = None

    if os.path.sep in str(stream) or os.path.exists(stream):
        try:
            stream_path = Path(stream)
            if stream_path.is_dir():
                raise FileNotFoundError(f"{stream!r} is directory and cannot be opened.")
            contents = "".join(open(str(stream_path), "r").readlines())
        except OSError:
            pass

    if contents is None:
        contents = str(stream)

    if not isinstance(contents, str):
        raise TypeError(f"Cannot parse contents of type {type(contents)} -{contents}.")

    contents = update_contents(contents, **kwargs)

    try:
        manifest = yaml.safe_load(contents)
    except yaml.YAMLError:
        manifest = None

    if isinstance(manifest, dict) and manifest.get(NODE_KIND_KEY) == MANIFEST_KIND:
        return _load_manifest(manifest, base_path=stream_path.parent if stream_path is not None else None)

    if CONFIG_KEY in contents:
        config_str = contents[contents.index(CONFIG_KEY) :].replace(CONFIG_KEY, "")
        contents = contents[: contents.index(CONFIG_KEY)].replace(CONFIG_KEY, "")
        config = yaml.load(config_str, yaml.Loader)
    else:
        config = {}  # noqa: F841

    loaded_obj = yaml.load(contents, Loader)
    if not isinstance(loaded_obj, PipelineDataModule):
        raise FileNotFoundError(f"Cannot load {stream!r}, is it a valid datamodule?")
    return loaded_obj
