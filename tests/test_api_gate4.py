"""Integration gate: real manifest, checkpoint predictions and held-out metrics."""
import json
from pathlib import Path
import sys

import numpy as np
import torch
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "api"))
sys.path.insert(0, str(ROOT / "src"))
from main import app, state


def main():
    torch.set_num_threads(1)
    expected = json.loads((ROOT / "notes/results/gate4_evaluation.json").read_text())
    checks = []
    with TestClient(app) as client:
        entries = client.get('/designs').json()
        assert len(entries) == 12
        assert {entry['model_source'] for entry in entries} == {'checkpoint'}
        for entry in entries:
            sample_id = entry['id']
            detail = client.get('/designs/' + sample_id).json()
            assert detail['die_w_dbu'] > 0
            assert detail['die_h_dbu'] > 0
            response = client.get('/designs/' + sample_id + '/prediction')
            assert response.status_code == 200
            prediction = response.json()
            assert prediction['model_source'] == 'checkpoint'
            assert prediction['inference_seconds'] > 0
            assert prediction['reference_router_seconds'] == 122
            assert prediction['sample_router_seconds'] > 0
            graph = state['store'].get_graph(sample_id)
            with torch.no_grad():
                actual = state['store'].model(graph).numpy()
            assert np.array_equal(np.asarray(prediction['horizontal']), actual[:, 0])
            assert np.array_equal(np.asarray(prediction['vertical']), actual[:, 1])
            truth = client.get('/designs/' + sample_id + '/truth').json()
            assert np.array_equal(np.asarray(truth['horizontal']), graph['gcell'].y_ratio[:, 0].numpy())
            hotspots = client.get('/designs/' + sample_id + '/hotspots').json()
            assert len(hotspots['hotspots']) == 20
            if entry['split'] == 'test':
                response = client.get('/designs/' + sample_id + '/metrics')
                assert response.status_code == 200
                metrics = response.json()['channels']
                reference = next(row for row in expected['samples'] if sample_id == row['design'] + '__' + row['variant'] and row['method'] == 'CongestionGNN')
                assert abs(metrics['horizontal']['spearman'] - reference['channels']['H']['spearman']) < 1e-10
                assert abs(metrics['vertical']['spearman'] - reference['channels']['V']['spearman']) < 1e-10
            checks.append(dict(id=sample_id, inference_seconds=prediction['inference_seconds'], reference_router_seconds=prediction['reference_router_seconds']))
            print('PASS', sample_id, flush=True)
        assert client.get('/designs/missing').status_code == 404
        assert client.get('/').status_code == 200
    (ROOT / 'notes/results/gate4_api.json').write_text(json.dumps(checks, indent=2) + '\n')
    print('PASS: all real designs; checkpoint output equality; held-out metric agreement; static frontend')


if __name__ == '__main__':
    main()
