"""Create a corrected report copy from the known handoff template and measured JSON."""
import copy
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import zipfile

ROOT = Path(__file__).resolve().parents[1]
W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


def replace_text(paragraph, text):
    nodes = paragraph.findall(".//" + W + "t")
    if not nodes:
        run = ET.SubElement(paragraph, W + "r")
        nodes = [ET.SubElement(run, W + "t")]
    nodes[0].text = text
    for node in nodes[1:]:
        node.text = ""


def replace_table(table, rows):
    templates = table.findall(W + "tr")
    for row in templates:
        table.remove(row)
    for index in range(len(rows)):
        row = copy.deepcopy(templates[0] if index == 0 else templates[1])
        cells = row.findall(W + "tc")
        while len(cells) < len(rows[index]):
            row.append(copy.deepcopy(cells[-1]))
            cells = row.findall(W + "tc")
        for extra in cells[len(rows[index]):]:
            row.remove(extra)
        for column in range(len(rows[index])):
            cell = cells[column]
            paragraphs = cell.findall(W + "p")
            replace_text(paragraphs[0], str(rows[index][column]))
            for paragraph in paragraphs[1:]:
                cell.remove(paragraph)
        table.append(row)


def main():
    source = Path('/Users/sidhant/Downloads/BTech_Project_Report_HGNN_Congestion.docx')
    assert hashlib.sha256(source.read_bytes()).hexdigest() == '7b5d5d65e41908266211adf614aee373ce5c58f7cb18a3680e912da77d40b95d'
    results = json.loads((ROOT / 'notes/results/gate4_evaluation.json').read_text())
    with zipfile.ZipFile(source) as archive:
        raw = archive.read('word/document.xml')
    for namespace in ET.iterparse(__import__('io').BytesIO(raw), events=['start-ns']):
        prefix = namespace[1][0]
        uri = namespace[1][1]
        if not prefix.startswith('ns'):
            ET.register_namespace(prefix, uri)
    document = ET.fromstring(raw)
    paragraphs = document.findall('.//' + W + 'p')
    summary = results['summary']
    gnn = summary['CongestionGNN']
    measured = f"Held-out aes Spearman is {gnn['H']['spearman']:.4f} horizontally and {gnn['V']['spearman']:.4f} vertically, averaged over its three variants."
    limits = ('The corpus has four designs in one technology, with two training designs, one validation design and one test design. '
              'The headline result therefore rests on aes alone. Seed and utilisation vary together, so their effects cannot be separated. '
              'tinyRocket spans only 60/63/66% utilisation and its final variant has IQR 0.1667, just below the 0.17 target after both allowed step-downs. '
              'Quantised routing ratios make IQR move in discrete steps. dynamic_node responds non-monotonically to utilisation. '
              'No final sample overflows; development runs at higher tinyRocket utilisation did overflow and were replaced for label quality, not prediction scores.')
    changes = {
        62: 'The final experiment uses twelve real OpenROAD samples across four nangate45 designs, split by design. ' + measured + ' Density-only, capacity-only and RUDY use the identical test split. Earlier synthetic training results validate infrastructure only. CircuitNet, ISPD2015, CNN comparison, ablation and cross-technology transfer were not performed.',
        142: 'Real nangate45 evaluation, per channel',
        145: 'Held-out aes variants, per channel',
        211: 'The final corpus contains dynamic_node, tinyRocket, ibex and aes in nangate45, with three placement variants each. Graphs use placed cells and net connectivity; labels are extracted after global routing. CircuitNet and ISPD2015 remain future evaluation datasets.',
        212: 'The project contributes a validated extraction and graph-construction pipeline, measured memory engineering, and a ranking-first evaluation protocol. The heterogeneous architecture follows published work. Cross-technology transfer is not claimed.',
        225: 'Evaluation: Compare the trained model with RUDY, density-only and capacity-only on genuine router labels using a disjoint design-level split and per-channel ranking and hotspot metrics.',
        226: 'Generalisation: Evaluate transfer to a held-out design within nangate45. Cross-technology generalisation remains future work.',
        233: 'Chapter 5 presents pipeline gates, synthetic scale measurements, partition fidelity, training infrastructure and the final real-corpus comparison. Synthetic and real-label findings are distinguished explicitly.',
        316: 'This implementation combines logical and spatial relations in a heterogeneous graph and measures memory use through tiled graph construction and spatial partitioning. Hierarchical grids and cross-technology transfer are future work. Ranking metrics assess how well predictions prioritise regions.',
        362: 'Held-out-design evaluation; cross-technology transfer deferred',
        415: 'The final real corpus comprises twelve samples: dynamic_node at 40/55/65% utilisation, ibex at 50/65/75%, tinyRocket at 60/63/66%, and aes at 35/40/45%. The split is dynamic_node plus tinyRocket for training, ibex for validation, and aes for test. All variants of a design stay in one split.',
        416: 'Earlier single-design sweeps were development experiments and are excluded from the final corpus. An incremental reroute overwrote congestion grids; measurements taken before its correction are invalid for the final pressure comparison. The verified final corpus has twelve distinct congestion maps.',
        417: 'No CircuitNet or ISPD2015 accuracy experiment was completed. Large synthetic graphs support the scale-engineering experiments only; the final predictive evaluation uses the four-design OpenROAD corpus.',
        420: 'Ground truth is extracted after global routing, separately for horizontal and vertical demand/capacity ratios. The corpus-generation image lacks OpenROAD Python support, so Tcl extraction plus odb_text_to_npz.py produces the same NPZ schema as the Python reference. Inputs come from placement before CTS; labels come from the subsequent global route.',
        422: 'The final recipe applies a uniform metal2–5 routing-layer cap, SKIP_INCREMENTAL_REPAIR=1 and OPT_POST_GRT_WNS=1. Utilisation steps were selected before model training to preserve label spread, targeting roughly 0.17–0.23 combined IQR with the documented step-down rule. Spatial blockage bands were abandoned because OpenROAD records them as usage rather than reduced capacity, leaving the intervention invisible in input features. The earlier uncapped-utilisation comparison is invalidated by the reroute bug and cannot quantitatively justify the cap.',
        509: 'Primary metric: Spearman with midrank tie handling, reported per channel. Kendall tau-b is exact below the project threshold and flagged when sampled. Hotspots are true ratios >= 0.9; predictions select the top-|S| regions with fractional credit for cutoff ties. Precision equals recall by construction; ROC-AUC uses the full map. Top-20 hit rate is excluded because saturated targets have large tie blocks.',
        660: 'F. Real-Corpus Evaluation on nangate45',
        661: 'One fixed CPU run used up to 200 epochs, learning rate 0.001, accumulation of two graphs, Huber plus ranking loss with weight 0.5, and early-stopping patience 20. The best checkpoint was selected only by ibex validation Spearman. Density and capacity regressions were fitted only on the six training samples; RUDY is unfitted. ' + measured,
        662: 'Table 5.5 Held-out aes: unweighted mean over three variants, per channel',
        699: 'All entries are measured from the validation-selected checkpoint and the identical held-out samples. Training stopped after 100 epochs, selecting zero-based epoch 79. No CNN or net-node ablation was run. Model selection did not use test scores. GNN CPU inference medians (25 calls after one warm-up, one thread; excluding graph construction and loading) were ' + ', '.join(f"{row['inference_seconds'] * 1000:.1f} ms on {row['variant']}" for row in results['samples'] if row['method'] == 'CongestionGNN') + '. The historical router reference is 122 seconds; it is not the measured routing duration of every test variant.',
        700: 'Table 5.6 Held-out aes variants: Spearman H/V for every evaluated method',
        721: 'H and V are reported separately. This is within-technology generalisation to one unseen netlist, not a cross-technology transfer study.',
        723: 'Density-only uses cell and pin density at local neighbourhood scales; capacity-only uses capacity and inverse-capacity features without connectivity. RUDY uses net bounding boxes and capacity normalisation. These comparisons separate useful analytical structure from what the learned model achieves on this split.',
        724: 'RUDY outperforms CongestionGNN on both channels across the reported aggregate ranking and hotspot metrics. Density-only exceeds the model horizontally and is close vertically; capacity-only is substantially weaker. Thus the model learns beyond capacity alone, but this experiment does not demonstrate an advantage over analytical routing demand or density. The contribution is reframed around the validated extraction pipeline and separately measured memory engineering. No further corpus sweeps or test-driven model changes were made, and without an ablation the isolated benefit of connectivity cannot be claimed.',
        725: limits,
        732: 'The interface now serves the real corpus and checkpoint. Macro outlines remain unavailable. Prediction on user-modified placements has not been validated and must not be treated as demonstrated generalisation.',
        741: 'The contribution is the validated router-label pipeline, memory-conscious graph construction and partitioning measurements, and honest per-channel comparison on a small real corpus. CPU synthetic memory measurements do not establish GPU training requirements, and cross-technology transfer remains future work.',
    }
    for index in changes:
        replace_text(paragraphs[index], changes[index])
    tables = document.findall('.//' + W + 'tbl')
    benchmark = next(t for t in tables if 'CNN baseline' in ''.join(t.itertext()))
    transfer = next(t for t in tables if 'N28 → N14' in ''.join(t.itertext()))
    rows = [['Method / channel', 'Spearman', 'Kendall', 'Precision', 'Recall', 'ROC-AUC']]
    for method in summary:
        for channel in 'HV':
            metrics = summary[method][channel]
            rows.append([method + ' / ' + channel] + [f'{metrics[k]:.4f}' for k in ('spearman', 'kendall', 'precision', 'recall', 'roc_auc')])
    replace_table(benchmark, rows)
    rows = [['Variant / method', 'Spearman H', 'Spearman V', 'Kendall sampled']]
    for sample in results['samples']:
        rows.append([sample['variant'] + ' / ' + sample['method'], f"{sample['channels']['H']['spearman']:.4f}", f"{sample['channels']['V']['spearman']:.4f}", str(sample['channels']['H']['kendall_sampled'] or sample['channels']['V']['kendall_sampled'])])
    replace_table(transfer, rows)
    target = ROOT / 'deliverables/BTech_Project_Report_HGNN_Congestion_gate4.docx'
    target.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(source) as archive:
        with zipfile.ZipFile(target, 'w', zipfile.ZIP_DEFLATED) as output:
            for member in archive.infolist():
                output.writestr(member, ET.tostring(document, encoding='utf-8', xml_declaration=True) if member.filename == 'word/document.xml' else archive.read(member.filename))
    assert '[TBD]' not in ''.join(document.itertext())
    print(target)


if __name__ == '__main__':
    main()
