import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))
sys.path.insert(0, str(REPO / 'src/video_highlight/ora'))

from video_highlight.config import load_config
from video_highlight import cli
from video_highlight.runtime.resource_tuning import ResourceSnapshot, choose_plan, concurrency_limit
from video_highlight.runtime.stage2_prefetch import prefetch
from video_highlight.ora.export_submission import inclusive_ranges, selected_frames
from video_highlight.ora.validate_export import valid_event_interval

spec = importlib.util.spec_from_file_location('build_index', REPO / 'tools/build_index.py')
index_tool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(index_tool)


class ConfigurationTests(unittest.TestCase):
    def test_custom_model_port_is_used_by_inference_and_repair(self):
        config = {'temporal_run_dir': 'unused', 'temporal_python': sys.executable,
                  'test_index': 'index.json', 'video_dir': 'videos',
                  'model_revision': 'test', 'model_port': 8123}
        with patch.object(cli, 'run') as run:
            cli.temporal(config, dry_run=True)
        for call in run.call_args_list[:2]:
            command = call.args[0]
            self.assertEqual(command[command.index('--base-url') + 1], 'http://127.0.0.1:8123/v1')

    def test_relative_paths_and_venv_executable_are_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'configs').mkdir()
            path = root / 'configs/config.json'
            path.write_text(json.dumps({'run_root': '../runs/demo', 'video_dir': '../videos',
                                        'test_index': '../index.json', 'python_executable': '../venv/bin/python'}))
            config = load_config(path)
            self.assertEqual(config['run_root'], str((root / 'runs/demo').resolve()))
            self.assertEqual(config['python_executable'], os.path.abspath(root / 'venv/bin/python'))

    def test_configuration_without_required_input_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'config.json'
            path.write_text('{}')
            with self.assertRaises(ValueError):
                load_config(path)


class InputIndexTests(unittest.TestCase):
    def test_original_configuration_without_video_id_uses_filename(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            videos, configs = root / 'videos', root / 'configs'
            videos.mkdir(); configs.mkdir()
            (videos / '0.mp4').touch()
            (configs / '0.jsonl').write_text('{"targetRatioWH":[16,9]}\n')
            self.assertEqual(index_tool.build_index(videos, configs), [{'video_id': '0', 'targetRatioWH': [16, 9]}])

    def test_mismatched_configuration_id_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            videos, configs = root / 'videos', root / 'configs'
            videos.mkdir(); configs.mkdir()
            (videos / '0.mp4').touch()
            (configs / '0.jsonl').write_text('{"video_id":"1","targetRatioWH":[16,9]}\n')
            with self.assertRaises(ValueError):
                index_tool.build_index(videos, configs)


class TemporalContractTests(unittest.TestCase):
    def test_variable_rate_timestamps_map_to_original_frame_indices(self):
        self.assertEqual(selected_frames([0, .03, .07, .14], [[.02, .14]]), [1, 2])

    def test_inclusive_ranges_keep_gaps_and_both_endpoints(self):
        self.assertEqual(inclusive_ranges([1, 2, 3, 7, 8]), [[1, 3], [7, 8]])
        self.assertEqual(inclusive_ranges([]), [])

    def test_only_exact_rounded_duration_is_allowed_above_raw_duration(self):
        self.assertTrue(valid_event_interval(0, 5.86667, 5.866667))
        self.assertFalse(valid_event_interval(0, 5.86668, 5.866667))
        self.assertFalse(valid_event_interval(0, float('nan'), 6))
        self.assertFalse(valid_event_interval(4, 3, 6))


class OrderedPrefetchTests(unittest.TestCase):
    def test_prefetch_keeps_sequence_under_backpressure(self):
        self.assertEqual(list(prefetch(iter(range(100)), capacity=1)), list(range(100)))

    def test_decoder_failure_reaches_the_inference_consumer(self):
        def source():
            yield 'frame'
            raise ValueError('decoder failed')
        stream = prefetch(source())
        self.assertEqual(next(stream), 'frame')
        with self.assertRaisesRegex(ValueError, 'decoder failed'):
            next(stream)


class AdaptiveSchedulingTests(unittest.TestCase):
    def snapshot(self, **changes):
        values = dict(cpu_count=15, cpu_load_percent=30, ram_available_gib=80,
                      gpu_name='RTX 4090', gpu_total_gib=24, gpu_free_gib=23, gpu_utilization_percent=30)
        values.update(changes)
        return ResourceSnapshot(**values)

    def test_24gb_machine_gets_validated_parallel_plan(self):
        plan = choose_plan(self.snapshot())
        self.assertEqual((plan.yolo_batch_size, plan.pose_batch_size, plan.tracking_workers), (64, 32, 3))

    def test_busy_gpu_uses_free_memory_instead_of_total_memory(self):
        plan = choose_plan(self.snapshot(gpu_free_gib=2))
        self.assertEqual(plan.yolo_batch_size, 8)
        self.assertEqual(plan.tracking_workers, 1)

    def test_cpu_pressure_and_low_memory_reduce_new_work(self):
        self.assertEqual(concurrency_limit(self.snapshot(cpu_load_percent=96), 3), 1)
        self.assertEqual(concurrency_limit(self.snapshot(gpu_free_gib=.5), 3), 1)
        self.assertEqual(concurrency_limit(self.snapshot(), 3), 3)

    def test_compatibility_profile_keeps_original_batches_and_serial_tracking(self):
        plan = choose_plan(self.snapshot(), 'compat')
        self.assertEqual((plan.yolo_batch_size, plan.pose_batch_size, plan.tracking_workers), (8, 4, 1))


if __name__ == '__main__':
    unittest.main()
