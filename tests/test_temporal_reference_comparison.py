import importlib.util
from pathlib import Path
import pytest

spec=importlib.util.spec_from_file_location('temporal_reference_compare',Path(__file__).resolve().parents[1]/'tools/compare_temporal_reference.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def test_shortening_can_remove_reference_events():
    result=module.compare({0,1},{1,2,3},5)
    assert result['ora_only_frames']==1
    assert result['reference_only_frames']==2
    assert result['reference_coverage']==pytest.approx(1/3)


def test_original_frame_contract_rejects_duplicates_and_wrong_base():
    for row in ({'highlight_frames':[1,1]},{'highlight_frames':[5]},{'highlight_frames':[1],'frame_index_base':1}):
        with pytest.raises(ValueError):module.mask(row,5)


def test_pts_intervals_are_half_open_and_overlap_is_not_double_counted():
    assert module.interval_frames([0,.03,.09,.15,.24],[[.03,.15],[.09,.25]])==4
