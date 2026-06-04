from __future__ import annotations

import os
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import xarray as xr


PROJECT_ROOT = Path(__file__).resolve().parents[3]
for package in ("utils", "data", "pipeline", "training"):
    package_src = str(PROJECT_ROOT / "packages" / package / "src")
    if package_src not in sys.path:
        sys.path.insert(0, package_src)


def _install_common_stubs():
    os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/pet_training_tests_mpl")

    if "graphviz" not in sys.modules:
        sys.modules["graphviz"] = types.SimpleNamespace(Digraph=object)

    if "numcodecs" not in sys.modules:
        numcodecs = types.ModuleType("numcodecs")
        blosc = types.ModuleType("numcodecs.blosc")

        class Blosc: ...

        blosc.Blosc = Blosc
        numcodecs.blosc = blosc
        sys.modules["numcodecs"] = numcodecs
        sys.modules["numcodecs.blosc"] = blosc

    if "einops" not in sys.modules:
        einops = types.ModuleType("einops")
        einops.rearrange = lambda value, *args, **kwargs: value
        sys.modules["einops"] = einops


def _install_lightning_stub():
    if "lightning" in sys.modules:
        return

    lightning = types.ModuleType("lightning")
    pytorch = types.ModuleType("lightning.pytorch")
    callbacks = types.ModuleType("lightning.pytorch.callbacks")
    loggers = types.ModuleType("lightning.pytorch.loggers")
    utilities = types.ModuleType("lightning.pytorch.utilities")

    class LightningModule: ...

    class LightningDataModule: ...

    class Trainer:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def save_checkpoint(self, path):
            self.last_checkpoint = path

        def predict(self, *args, **kwargs):
            return []

        def fit(self, *args, **kwargs):
            self.fit_args = (args, kwargs)

    class ModelCheckpoint:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class TensorBoardLogger:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    class CSVLogger(TensorBoardLogger):
        pass

    class CombinedLoader:
        def __init__(self, loaders, mode):
            self.loaders = loaders
            self.mode = mode

    lightning.LightningModule = LightningModule
    lightning.LightningDataModule = LightningDataModule
    lightning.Trainer = Trainer
    pytorch.callbacks = callbacks
    pytorch.loggers = loggers
    callbacks.ModelCheckpoint = ModelCheckpoint
    loggers.TensorBoardLogger = TensorBoardLogger
    loggers.CSVLogger = CSVLogger
    utilities.CombinedLoader = CombinedLoader

    sys.modules["lightning"] = lightning
    sys.modules["lightning.pytorch"] = pytorch
    sys.modules["lightning.pytorch.callbacks"] = callbacks
    sys.modules["lightning.pytorch.loggers"] = loggers
    sys.modules["lightning.pytorch.utilities"] = utilities


_install_common_stubs()

from pyearthtools.data.indexes.fake import FakeIndex
from pyearthtools.pipeline.controller import Pipeline
from pyearthtools.pipeline.iterators import DateRange
from pyearthtools.pipeline.operations.xarray.conversion import ToNumpy
from pyearthtools.pipeline.operations.xarray.normalisation import Deviation
from pyearthtools.training.data.datamodule import PipelineDataModule
from pyearthtools.training.data.fileio import MANIFEST_KIND, NODE_KIND_KEY, load, save


def _normalisation_datasets():
    time = np.array(
        [
            np.datetime64("2000-01-01T00"),
            np.datetime64("2000-01-01T06"),
            np.datetime64("2000-01-01T12"),
            np.datetime64("2000-01-01T18"),
        ],
        dtype="datetime64[ns]",
    )
    latitude = np.linspace(90, -90, 8)
    longitude = np.linspace(0, 360, 10)
    shape = (len(time), len(latitude), len(longitude))

    mean = xr.Dataset(
        {
            "ocean_heat_content_2d": (("time", "latitude", "longitude"), np.ones(shape, dtype=np.float32)),
            "total_surface_heat_flx": (("time", "latitude", "longitude"), np.full(shape, 2.0, dtype=np.float32)),
        },
        coords={"time": time, "latitude": latitude, "longitude": longitude},
    )
    deviation = xr.Dataset(
        {
            "ocean_heat_content_2d": (("time", "latitude", "longitude"), np.full(shape, 3.0, dtype=np.float32)),
            "total_surface_heat_flx": (("time", "latitude", "longitude"), np.full(shape, 4.0, dtype=np.float32)),
        },
        coords={"time": time, "latitude": latitude, "longitude": longitude},
    )
    return mean, deviation


def _pipeline_from_normalisation(mean, deviation):
    return Pipeline(
        FakeIndex(
            variable=["ocean_heat_content_2d", "total_surface_heat_flx"],
            interval=(6, "hour"),
            data_size=(8, 10),
            random=False,
        ),
        Deviation(mean=mean, deviation=deviation),
        ToNumpy(),
        iterator=DateRange("2000-01-01", "2000-01-02", "6 hours"),
    )


def test_manifest_mode_round_trips_with_sidecar_xarray(tmp_path):
    mean, deviation = _normalisation_datasets()
    pipeline = _pipeline_from_normalisation(mean, deviation)
    datamodule = PipelineDataModule(pipeline)

    manifest_base = tmp_path / "DataModule"
    save(datamodule, manifest_base, mode="manifest")

    manifest_path = manifest_base.with_suffix(".edm")
    legacy_yaml = save(datamodule, mode="yaml")

    assert manifest_path.exists()
    assert f"{NODE_KIND_KEY}: {MANIFEST_KIND}" in manifest_path.read_text()
    assert manifest_path.stat().st_size < len(legacy_yaml)
    assert (tmp_path / "DataModule.resources").is_dir()

    loaded_datamodule = load(manifest_path)
    loaded_pipeline = loaded_datamodule.pipelines
    assert isinstance(loaded_pipeline, Pipeline)
    assert np.allclose(next(iter(pipeline)), next(iter(loaded_pipeline)))


def test_snapshot_mode_keeps_path_backed_normalisation_reproducible(tmp_path, monkeypatch):
    mean, deviation = _normalisation_datasets()
    mean_path = tmp_path / "mean.nc"
    deviation_path = tmp_path / "deviation.nc"
    mean.to_netcdf(mean_path, engine="scipy")
    deviation.to_netcdf(deviation_path, engine="scipy")

    monkeypatch.setattr(Deviation, "open_file", classmethod(lambda cls, file: xr.load_dataset(file, engine="scipy")))

    pipeline = _pipeline_from_normalisation(str(mean_path), str(deviation_path))
    datamodule = PipelineDataModule(pipeline)

    expected = next(iter(pipeline))
    manifest_path = (tmp_path / "SnapshotDataModule").with_suffix(".edm")
    save(datamodule, manifest_path.with_suffix(""), mode="snapshot")

    mean_path.unlink()
    deviation_path.unlink()

    loaded_datamodule = load(manifest_path)
    loaded_pipeline = loaded_datamodule.pipelines
    assert isinstance(loaded_pipeline, Pipeline)
    assert np.allclose(expected, next(iter(loaded_pipeline)))


def test_lightning_wrapper_respects_datamodule_save_mode(tmp_path):
    _install_lightning_stub()

    from lightning import LightningModule
    from pyearthtools.training.wrapper.lightning.wrapper import LightningWrapper

    mean, deviation = _normalisation_datasets()
    pipeline = _pipeline_from_normalisation(mean, deviation)

    run_dir = tmp_path / "run_none"
    LightningWrapper(LightningModule(), pipeline, run_dir, datamodule_save_mode="none")
    assert not (run_dir / "DataModule.edm").exists()

    manifest_dir = tmp_path / "run_manifest"
    LightningWrapper(LightningModule(), pipeline, manifest_dir, datamodule_save_mode="manifest")
    assert (manifest_dir / "DataModule.edm").exists()
