import json
from pathlib import Path
import subprocess

import pytest

from video_highlight import experiments as exp
from video_highlight import runner
from video_highlight.prompts import bundle_path


@pytest.fixture
def repo(tmp_path, monkeypatch):
    subprocess.run(['git', 'init', '-q', str(tmp_path)], check=True)
    (tmp_path / 'source.py').write_text('version = 1\n')
    subprocess.run(['git', '-C', str(tmp_path), 'add', '.'], check=True)
    subprocess.run(['git', '-C', str(tmp_path), '-c', 'user.name=Test', '-c', 'user.email=test@example.invalid',
                    'commit', '-qm', 'source'], check=True)
    monkeypatch.setattr(exp, 'REPO', tmp_path)
    return tmp_path


def configuration():
    return {'pipeline': 'baseline_center', 'dataset': {'id': 'synthetic'}, 'model': {'id':'none'}, 'prompts': {}}


def test_source_changes_cannot_be_registered_as_clean(repo):
    (repo / 'source.py').write_text('version = 2\n')
    with pytest.raises(ValueError, match='Commit source'):
        exp.create('test-run', 'tester', configuration())


def test_metadata_commit_allows_resume_without_changing_code_identity(repo):
    record = exp.create('test-run', 'tester', configuration())
    subprocess.run(['git', '-C', str(repo), 'add', 'experiments'], check=True)
    subprocess.run(['git', '-C', str(repo), '-c', 'user.name=Test', '-c', 'user.email=test@example.invalid',
                    'commit', '-qm', 'metadata'], check=True)
    assert exp.verify('test-run')[0]['git_commit'] == record['git_commit']


def test_configuration_tampering_blocks_resume(repo):
    exp.create('test-run', 'tester', configuration())
    path = exp.experiment_path('test-run') / 'config.json'
    changed = json.loads(path.read_text())
    changed['pipeline'] = 'qwen_window'
    exp.write_json(path, changed)
    with pytest.raises(ValueError, match='configuration or prompt'):
        exp.verify('test-run')


def test_duplicate_ids_do_not_overwrite_experiments(repo):
    exp.create('test-run', 'tester', configuration())
    with pytest.raises(FileExistsError):
        exp.create('test-run', 'other', configuration())


def test_metrics_distinguish_proxy_and_leaderboard(repo):
    exp.create('test-run', 'tester', configuration())
    with pytest.raises(ValueError, match='metric_type'):
        exp.record_metrics('test-run', {'score':42})
    exp.record_metrics('test-run', {'score': .5, 'metric_type':'temporal_proxy'})
    assert exp.registry()[0]['metric_type'] == 'temporal_proxy'


def test_prompt_paths_cannot_escape_bundle_root():
    with pytest.raises(ValueError):
        bundle_path('../../')


def test_dry_run_does_not_create_outputs(tmp_path, monkeypatch):
    config = runner.load_config(runner.REPO / 'configs/pipelines/baseline_center.json')
    root = tmp_path / 'runs'
    runner.execute(config, 'test-demo', local={'run_root':str(root)}, dry_run=True)
    assert not root.exists()


def test_reference_route_cannot_claim_to_run():
    with pytest.raises(ValueError, match='reference assets only'):
        runner.commands({'pipeline':'ora_reranker'}, 'test-reference')


def test_ora_9b_name_and_revision_are_forwarded():
    commands, generated, _ = runner.commands({'pipeline':'ora_direct', 'model': {'served_name':'Video-ORA-9B',
        'revision':'revision9'}, 'parameters': {'fps':3}}, 'test-9b')
    resolved = next(iter(generated.values()))
    assert resolved['served_model_name'] == 'Video-ORA-9B'
    assert resolved['model_revision'] == 'revision9'
    assert resolved['temporal_parameters']['fps'] == 3


def test_local_paths_cannot_override_algorithm_parameters():
    with pytest.raises(ValueError, match='Only machine paths'):
        runner.commands(configuration(), 'test-run', {'threshold':.4})


def test_missing_model_revision_cannot_be_formal_experiment(repo):
    config = configuration()
    config['model'] = {'id':'Qwen/Qwen2.5-VL-3B-Instruct', 'revision':'unknown'}
    with pytest.raises(ValueError, match='exact model revision'):
        exp.create('test-run', 'tester', config)


def test_spatial_threshold_overrides_keep_io_contract():
    from video_highlight.spatial_parameters import apply_parameters
    command = ['python', 'run.py', '--conf', '.15', '--output', 'unchanged.jsonl']
    result = apply_parameters([('YOLO 主体定位', command)], {'yolo':{'conf':.2}})
    assert result[0][1][-1] == 'unchanged.jsonl'
    assert result[0][1][3] == '.2' or result[0][1][3] == '0.2'
    with pytest.raises(ValueError, match='Unsupported'):
        apply_parameters([('YOLO 主体定位', command)], {'yolo':{'output':'other.jsonl'}})


def test_duplicate_predictions_are_penalized_by_reference_metric():
    from video_highlight.evaluation import weighted_f1
    truth = [{'frame':0,'bboxes':[0,0,100]}]
    perfect = weighted_f1(truth, truth, 100, 100, [1,1])['f1']
    duplicate = weighted_f1(truth + truth, truth, 100, 100, [1,1])['f1']
    assert perfect == 1
    assert duplicate == pytest.approx(2/3)
