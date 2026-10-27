"""Live Chromium gate and pixel comparison with the archived OpenROAD dump.

Requires the API on localhost:8000 and installed Google Chrome.
Run scripts/prepare_real_api.py first; the two reference files are extracted
from the verified archive if absent. This checks the actual rendered canvas.
"""
import csv
import json
from pathlib import Path
import tarfile

import numpy as np
import torch
from chrome_gate import ChromePage

ROOT = Path(__file__).resolve().parents[1]


def main():
    raw_path = ROOT / "data/raw/sweep_dynamic_node__k5b_congestion.npz"
    csv_path = ROOT / "data/raw/dynamic_node__k5b_gui_heatmap.csv"
    if not raw_path.exists() or not csv_path.exists():
        with tarfile.open('/Users/sidhant/Downloads/phase14_final_20261027.tar.gz') as archive:
            for path in [raw_path, csv_path]:
                path.write_bytes(archive.extractfile('Congestion-detection-gnn/' + str(path.relative_to(ROOT))).read())
    raw = np.load(raw_path)
    graph = torch.load(ROOT / 'data/processed/corpus/dynamic_node__u40_s1_graph.pt', weights_only=False)
    assert np.array_equal(raw['ratio'].reshape(-1, 2).astype(np.float32), graph['gcell'].y_ratio.numpy())
    ny = int(raw['ny'])
    nx = int(raw['nx'])
    pitch_x = float(raw['pitch_x']) / float(raw['dbu_per_micron'])
    pitch_y = float(raw['pitch_y']) / float(raw['dbu_per_micron'])
    reference = np.full((ny, nx), np.nan)
    with csv_path.open() as stream:
        for row in csv.DictReader(stream):
            x = round(float(row['x0']) / pitch_x)
            y = round(float(row['y0']) / pitch_y)
            reference[y, x] = float(row['value (%)']) / 100
    assert np.isfinite(reference).all()
    difference = np.abs(reference - raw['ratio'].max(axis=2))
    result = dict(reference='dynamic_node__k5b_gui_heatmap.csv', corpus_sample='dynamic_node__u40_s1',
                  reference_labels_equal=True, reference_cells=int(reference.size),
                  max_ratio_difference=float(difference.max()))
    with ChromePage() as page:
        page.call('Page.navigate', dict(url='http://127.0.0.1:8000'))
        page.wait("document.querySelector('#metrics-panel')?.textContent.includes('Spearman H')")
        selected = page.evaluate("document.querySelector('#design-select').value")
        assert selected.startswith('aes__')
        assert page.evaluate("document.querySelector('#fixture-banner').hidden")
        split_label = page.evaluate("document.querySelector('#design-meta').innerText")
        assert 'TEST (HELD OUT)' in split_label.upper(), split_label
        assert '122' in page.evaluate("document.querySelector('#router-value').innerText")
        result['held_out_design'] = selected
        result['held_out_metrics_text'] = page.evaluate("document.querySelector('#metrics-panel').innerText")
        evaluation = json.loads((ROOT / 'notes/results/gate4_evaluation.json').read_text())
        expected = next(row for row in evaluation['samples'] if row['method'] == 'CongestionGNN' and selected == row['design'] + '__' + row['variant'])
        for channel in 'HV':
            for metric in ['spearman', 'kendall', 'mae', 'rmse', 'precision', 'recall', 'roc_auc']:
                assert f"{expected['channels'][channel][metric]:.3f}" in result['held_out_metrics_text']
        result['banner_hidden'] = True
        result['split_label'] = split_label
        page.screenshot(ROOT / 'notes/figures/gate4_aes_ui.png')

        # Test the renderer with deliberately asymmetric rows, independently
        # of the real-data colour comparison below.
        result['orientation_probe'] = page.evaluate("""async () => {
            const {GridRenderer} = await import('/js/renderer.js');
            const canvas = document.createElement('canvas');
            canvas.width = 20; canvas.height = 20;
            const renderer = new GridRenderer(canvas);
            renderer.setGridShape(2, 2);
            renderer.buildBuffer('probe', [0, 0, 1, 1], v => v ? [255, 0, 0] : [0, 0, 255]);
            renderer.drawInto('probe', {x: 0, y: 0, width: 20, height: 20});
            return {top: Array.from(renderer.ctx.getImageData(5, 5, 1, 1).data),
                    bottom: Array.from(renderer.ctx.getImageData(5, 15, 1, 1).data)};
        }""")
        page.evaluate("document.querySelector('#design-select').value = 'dynamic_node__u40_s1'; document.querySelector('#design-select').dispatchEvent(new Event('change'))")
        page.wait("document.querySelector('#design-meta').textContent.includes('12,977')")
        page.evaluate("document.querySelector('[data-mode=truth]').click()")
        page.screenshot(ROOT / 'notes/figures/gate4_dynamic_truth_ui.png')
        result['pixels'] = page.evaluate("""async ({values, ny, nx}) => {
            const {congestionColor} = await import('/js/colormap.js');
            const {GridRenderer} = await import('/js/renderer.js');
            const detail = await (await fetch('/designs/dynamic_node__u40_s1')).json();
            const canvas = document.querySelector('#main-canvas');
            const renderer = new GridRenderer(canvas);
            const rect = renderer.computeDestRect(canvas.width, canvas.height, detail.die_w_dbu/detail.die_h_dbu);
            const ctx = canvas.getContext('2d');
            let matching = 0, flippedMatching = 0, maxError = 0;
            for (let y=0; y<ny; y++) for (let x=0; x<nx; x++) {
                const px = Math.floor(rect.x + (x+0.5)*rect.width/nx);
                const py = Math.floor(rect.y + (ny-y-0.5)*rect.height/ny);
                const actual = Array.from(ctx.getImageData(px, py, 1, 1).data).slice(0,3);
                const expected = Array.from(new Uint8ClampedArray(congestionColor(values[y*nx+x], 1.001)));
                const flipped = Array.from(new Uint8ClampedArray(congestionColor(values[(ny-y-1)*nx+x], 1.001)));
                const error = Math.max(...actual.map((v,i)=>Math.abs(v-expected[i])));
                maxError = Math.max(maxError,error);
                if(error<=1) matching++;
                if(Math.max(...actual.map((v,i)=>Math.abs(v-flipped[i])))<=1) flippedMatching++;
            }
            return {matching, flippedMatching, total:ny*nx, maxError};
        }""", dict(values=reference.reshape(-1).tolist(), ny=ny, nx=nx))
        result['browser_errors'] = page.errors
    output = ROOT / 'notes/results/gate4_browser.json'
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)
    assert not result['browser_errors']
    assert result['orientation_probe']['top'] == [255, 0, 0, 255]
    assert result['orientation_probe']['bottom'] == [0, 0, 255, 255]
    assert result['pixels']['matching'] == result['pixels']['total']
    assert difference.max() < 1e-6
    print('PASS: held-out UI, banner, split, metrics and every OpenROAD heatmap cell at the correct orientation')


if __name__ == '__main__':
    main()
