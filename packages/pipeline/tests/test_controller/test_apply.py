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

"""`Pipeline.apply(sample)` skips PipelineIndex steps that are safe to skip and raises for the rest."""

from __future__ import annotations

import pytest

import pyearthtools.utils

from pyearthtools.pipeline import Pipeline, PipelineIndex
from pyearthtools.pipeline.modifications.cache import Cache, MemCache
from pyearthtools.pipeline.warnings import PipelineWarning

from tests.fake_pipeline_steps import FakeIndex, MultiplicationOperation

pyearthtools.utils.config.set({"pipeline.run_parallel": False})


class PassThroughIndex(PipelineIndex):
    """Stands in for a cache: retrieves from the parent pipeline without changing the data."""

    skip_in_apply = True

    def __getitem__(self, idx):
        return self.parent_pipeline()[idx]


class FetchesMoreIndex(PipelineIndex):
    """Stands in for a temporal window: needs to fetch more data from upstream."""

    def __getitem__(self, idx):
        return [self.parent_pipeline()[idx], self.parent_pipeline()[idx + 1]]


def test_caches_are_skippable():
    assert Cache.skip_in_apply
    assert MemCache.skip_in_apply


def test_index_steps_not_skippable_by_default():
    assert not FetchesMoreIndex.skip_in_apply


def test_apply_skips_skippable_index_with_warning():
    pipe = Pipeline(FakeIndex(), PassThroughIndex(), MultiplicationOperation(10))
    with pytest.warns(PipelineWarning, match="PassThroughIndex"):
        assert pipe.apply(2) == 20
    # retrieving by index still goes through the PipelineIndex
    assert pipe[2] == 20


def test_apply_raises_for_index_that_fetches_more_data():
    pipe = Pipeline(FakeIndex(), FetchesMoreIndex(), MultiplicationOperation(10))
    with pytest.raises(TypeError, match="FetchesMoreIndex"):
        pipe.apply(2)
