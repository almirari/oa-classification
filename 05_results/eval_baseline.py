import sys, csv
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import cohen_kappa_score, classification_report

try:
    SCRIPT_DIR = Path(__file__).resolve().parent
except NameError:
    SCRIPT_DIR = Path(r'D:\UNI\TUGAS-AKHIR\code\oa-classification\05_results')

PROJECT_ROOT = Path(r'D:\UNI\TUGAS-AKHIR\code\oa-classification')
RESULTS_DIR  = SCRIPT_DIR
EXTRACT_DIR  = PROJECT_ROOT / '_dataset_extracted'
DATASET_ZIP  = PROJECT_ROOT / 'dataset.zip'
SEEDS        = [33, 81, 5]

SCENARIOS = [
    {
        'name':      'Baseline — MobileNetV3-Small (Cross-Entropy)',
        'short':     'baseline_ce',
        'model_dir': PROJECT_ROOT / '01_baseline' / 'models' / 'ce',
        'prefix':    's1',      # s1_seed{seed}_best.pth
        'head':      'ce',
    },
    {
        'name':      'Baseline — MobileNetV3-Small (CORAL Ordinal)',
        'short':     'baseline_coral',
        'model_dir': PROJECT_ROOT / '01_baseline' / 'models' / 'coral',
        'prefix':    'coral',   # coral_seed{seed}_best.pth
        'head':      'coral',
    },
    {
        'name':      'Baseline — MobileNetV3-Small (CORN Ordinal)',
        'short':     'baseline_corn',
        'model_dir': PROJECT_ROOT / '01_baseline' / 'models' / 'corn',
        'prefix':    'corn',    # corn_seed{seed}_best.pth
        'head':      'corn',
    },
    {
        'name':      'Baseline — MobileNetV3-Small (Niu et al. Ordinal)',
        'short':     'baseline_niu',
        'model_dir': PROJECT_ROOT / '01_baseline' / 'models' / 'niu',
        'prefix':    'niu',     # niu_seed{seed}_best.pth
        'head':      'niu',
    },
]

CLASS_NAMES = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative DR']
NUM_CLASSES = len(CLASS_NAMES)
BATCH_SIZE  = 32
device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
_MEAN       = [0.485, 0.456, 0.406]
_STD        = [0.229, 0.224, 0.225]


# ── Dataset (ImageFolder-based, matching eval_singlekd's file handling) ───────

def _find_test_dir(root):
    """Find a 'test' folder under root that itself contains class subfolders."""
    for p in sorted(root.rglob('test')):
        if p.is_dir() and any(c.is_dir() for c in p.iterdir()):
            return p
    return None


def get_dataset_dir():
    """Returns the path to the 'test' folder, which must contain one
    subfolder per class (e.g. test/0, test/1, ... test/4) — the standard
    torchvision.datasets.ImageFolder layout. Extracts dataset.zip if needed."""
    if EXTRACT_DIR.exists():
        test_dir = _find_test_dir(EXTRACT_DIR)
        if test_dir:
            print(f"  Dataset already extracted → {test_dir}")
            return test_dir

    if not DATASET_ZIP.exists():
        raise FileNotFoundError(
            f"dataset.zip not found at {DATASET_ZIP}\n"
            f"Please place dataset.zip in {PROJECT_ROOT} and re-run."
        )
    print(f"  Extracting {DATASET_ZIP.name} ...")
    import zipfile
    EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(DATASET_ZIP, 'r') as zf:
        zf.extractall(EXTRACT_DIR)
    test_dir = _find_test_dir(EXTRACT_DIR)
    if not test_dir:
        raise FileNotFoundError(
            "No 'test/<class>/...' folder structure found inside "
            f"{EXTRACT_DIR}"
        )
    print(f"  Extracted → {test_dir}")
    return test_dir


# ── Models ────────────────────────────────────────────────────────────────────

class CoralHead(nn.Module):
    def __init__(self, in_f, nc):
        super().__init__()
        self.fc   = nn.Linear(in_f, 1, bias=False)
        self.bias = nn.Parameter(torch.zeros(nc - 1))
    def forward(self, x):
        return self.fc(x) + self.bias


class CornHead(nn.Module):
    def __init__(self, in_f, nc):
        super().__init__()
        self.classifiers = nn.ModuleList([nn.Linear(in_f, 1) for _ in range(nc - 1)])
    def forward(self, x):
        return [c(x) for c in self.classifiers]


class NiuHead(nn.Module):
    def __init__(self, in_f, nc):
        super().__init__()
        self.heads = nn.ModuleList([nn.Linear(in_f, 1) for _ in range(nc - 1)])
    def forward(self, x):
        return torch.cat([h(x) for h in self.heads], dim=1)


def _build_model(head):
    m    = models.mobilenet_v3_small(weights=None)
    in_f = m.classifier[-1].in_features
    if head == 'ce':
        m.classifier[-1] = nn.Linear(in_f, NUM_CLASSES)
    else:
        m.classifier[-1] = nn.Identity()
        if   head == 'coral': m.coral_head = CoralHead(in_f, NUM_CLASSES)
        elif head == 'corn':  m.corn_head  = CornHead(in_f, NUM_CLASSES)
        elif head == 'niu':   m.niu_head   = NiuHead(in_f, NUM_CLASSES)
    return m


def load_model(ckpt_path, head):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd   = ckpt.get('model_state_dict', ckpt)
    m    = _build_model(head).to(device)
    m.load_state_dict(sd, strict=True)
    return m, ckpt


# ── Prediction ────────────────────────────────────────────────────────────────

def predict_batch(model, imgs, head):
    if head == 'ce':
        return model(imgs).argmax(dim=1)

    feats = model(imgs)   # classifier[-1] = Identity → (B, in_features)

    if head == 'coral':
        return (torch.sigmoid(model.coral_head(feats)) > 0.5).sum(dim=1).long()

    if head == 'corn':
        logit_list = model.corn_head(feats)
        cp = torch.cat([torch.sigmoid(l) for l in logit_list], dim=1)
        return (torch.cumprod(cp, dim=1) > 0.5).sum(dim=1).long()

    if head == 'niu':
        return (torch.sigmoid(model.niu_head(feats)) > 0.5).sum(dim=1).long()

    raise ValueError(f"Unknown head: {head}")


def evaluate(model, head, loader):
    model.eval()
    all_preds, all_labels = [], []
    n = len(loader)
    with torch.no_grad():
        for i, (imgs, labels) in enumerate(loader):
            all_preds.extend(predict_batch(model, imgs.to(device), head).cpu().numpy())
            all_labels.extend(labels.numpy())
            print(f"\r    batch {i+1}/{n}", end='', flush=True)
    print()
    preds  = np.array(all_preds)
    labels = np.array(all_labels)
    report = classification_report(labels, preds, target_names=CLASS_NAMES,
                                   output_dict=True, zero_division=0)
    return {
        'qwk':      float(cohen_kappa_score(labels, preds, weights='quadratic')),
        'acc':      float(np.mean(labels == preds)),
        'mae':      float(np.mean(np.abs(labels - preds))),
        'rmse':     float(np.sqrt(np.mean((labels - preds) ** 2))),
        'macro_p':  report['macro avg']['precision'],
        'macro_r':  report['macro avg']['recall'],
        'macro_f1': report['macro avg']['f1-score'],
        'wtd_p':    report['weighted avg']['precision'],
        'wtd_r':    report['weighted avg']['recall'],
        'wtd_f1':   report['weighted avg']['f1-score'],
        'per_class': {cls: {
            'p':       report[cls]['precision'],
            'r':       report[cls]['recall'],
            'f1':      report[cls]['f1-score'],
            'support': int(report[cls]['support']),
        } for cls in CLASS_NAMES},
    }


# ── Formatting ────────────────────────────────────────────────────────────────

def mu_sd(v):
    return np.mean(v), np.std(v)

W = 76


def section(title):
    return [f"\n  ── {title} " + "─" * max(0, W - 6 - len(title))]


def format_results(scenario, seed_results, best_qwks):
    L = ["=" * W, f"  {scenario['name']}", "=" * W]

    for r in seed_results:
        bq = best_qwks.get(r['seed'], 'N/A')
        L += section(f"Seed {r['seed']}  (training best val QWK: {bq})")
        L += [
            f"  {'Metric':<14} {'Value':>10}",
            "  " + "-" * 26,
            f"  {'QWK':<14} {r['qwk']:>10.4f}",
            f"  {'Accuracy':<14} {r['acc']*100:>9.2f}%",
            f"  {'MAE':<14} {r['mae']:>10.4f}",
            f"  {'RMSE':<14} {r['rmse']:>10.4f}",
            f"  {'Macro P':<14} {r['macro_p']:>10.4f}",
            f"  {'Macro R':<14} {r['macro_r']:>10.4f}",
            f"  {'Macro F1':<14} {r['macro_f1']:>10.4f}",
            f"  {'Wtd P':<14} {r['wtd_p']:>10.4f}",
            f"  {'Wtd R':<14} {r['wtd_r']:>10.4f}",
            f"  {'Wtd F1':<14} {r['wtd_f1']:>10.4f}",
            "",
            f"  {'Class':<22} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>8}",
            "  " + "-" * 62,
        ]
        for cls in CLASS_NAMES:
            pc = r['per_class'][cls]
            L.append(f"  {cls:<22} {pc['p']:>10.4f} {pc['r']:>10.4f} "
                     f"{pc['f1']:>10.4f} {pc['support']:>8d}")

    if len(seed_results) > 1:
        L += section(f"AVERAGED ACROSS {len(seed_results)} SEEDS  "
                     f"(seeds: {[r['seed'] for r in seed_results]})")
        L += ["",
              f"  {'Metric':<14} {'Mean':>10} {'± Std':>10} {'Min':>10} {'Max':>10}",
              "  " + "-" * 54]
        for label, key in [
            ('QWK',      'qwk'),   ('Acc (%)', None),  ('MAE',      'mae'),
            ('RMSE',     'rmse'),  ('Macro P', 'macro_p'), ('Macro R', 'macro_r'),
            ('Macro F1', 'macro_f1'), ('Wtd P', 'wtd_p'), ('Wtd R', 'wtd_r'),
            ('Wtd F1',   'wtd_f1'),
        ]:
            vals = ([r['acc']*100 for r in seed_results] if key is None
                    else [r[key] for r in seed_results])
            m, s = mu_sd(vals)
            L.append(f"  {label:<14} {m:>10.4f} {s:>10.4f} "
                     f"{min(vals):>10.4f} {max(vals):>10.4f}")
        L += ["",
              f"  {'Class':<22} {'Precision':>16} {'Recall':>16} {'F1-Score':>16}",
              "  " + "-" * 72]
        for cls in CLASS_NAMES:
            pm, ps = mu_sd([r['per_class'][cls]['p']  for r in seed_results])
            rm, rs = mu_sd([r['per_class'][cls]['r']  for r in seed_results])
            fm, fs = mu_sd([r['per_class'][cls]['f1'] for r in seed_results])
            L.append(f"  {cls:<22} {pm:.4f} ± {ps:.4f}   "
                     f"{rm:.4f} ± {rs:.4f}   {fm:.4f} ± {fs:.4f}")

    L.append("")
    return "\n".join(L)


def _cls_keys():
    return [(cls,
             f'p_{cls.lower().replace(" ","_")}',
             f'r_{cls.lower().replace(" ","_")}',
             f'f1_{cls.lower().replace(" ","_")}') for cls in CLASS_NAMES]


def _csv_fieldnames():
    return (['scenario', 'seed', 'head', 'qwk', 'acc_pct', 'mae', 'rmse',
             'macro_p', 'macro_r', 'macro_f1', 'wtd_p', 'wtd_r', 'wtd_f1']
            + [k for _, p, r, f in _cls_keys() for k in (p, r, f)])


def write_csv(csv_path, scenario_short, head, seed_results):
    exists     = csv_path.exists()
    cls_keys   = _cls_keys()
    fieldnames = _csv_fieldnames()
    with open(csv_path, 'a', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            w.writeheader()
        for r in seed_results:
            row = {
                'scenario': scenario_short, 'seed': r['seed'], 'head': head,
                'qwk':      round(r['qwk'],      4),
                'acc_pct':  round(r['acc']*100,  4),
                'mae':      round(r['mae'],       4),
                'rmse':     round(r['rmse'],      4),
                'macro_p':  round(r['macro_p'],   4),
                'macro_r':  round(r['macro_r'],   4),
                'macro_f1': round(r['macro_f1'],  4),
                'wtd_p':    round(r['wtd_p'],     4),
                'wtd_r':    round(r['wtd_r'],     4),
                'wtd_f1':   round(r['wtd_f1'],    4),
            }
            for cls, pk, rk, fk in cls_keys:
                row[pk] = round(r['per_class'][cls]['p'],  4)
                row[rk] = round(r['per_class'][cls]['r'],  4)
                row[fk] = round(r['per_class'][cls]['f1'], 4)
            w.writerow(row)


def format_summary(summary_rows, title):
    W2 = 90
    lines = ["=" * W2, f"  {title}  (mean ± std across seeds)", "=" * W2]

    for row in summary_rows:
        lines += [f"\n  Scenario : {row['scenario']}", "",
                  f"  {'Metric':<14} {'Mean':>10} {'± Std':>10} {'Min':>10} {'Max':>10}",
                  "  " + "-" * 54]
        for label, key in [
            ('QWK',      'qwk'),   ('Acc (%)',  'acc'), ('MAE',      'mae'),
            ('RMSE',     'rmse'),  ('Macro P',  'macro_p'), ('Macro R',  'macro_r'),
            ('Macro F1', 'macro_f1'), ('Wtd P', 'wtd_p'), ('Wtd R',    'wtd_r'),
            ('Wtd F1',   'wtd_f1'),
        ]:
            vals = row[key]
            m, s = mu_sd(vals)
            lines.append(f"  {label:<14} {m:>10.4f} {s:>10.4f} "
                         f"{min(vals):>10.4f} {max(vals):>10.4f}")
        lines += ["",
                  f"  {'Class':<22} {'Precision':>16} {'Recall':>16} {'F1-Score':>16}",
                  "  " + "-" * 72]
        for cls in CLASS_NAMES:
            pm, ps = mu_sd(row['per_class'][cls]['p'])
            rm, rs = mu_sd(row['per_class'][cls]['r'])
            fm, fs = mu_sd(row['per_class'][cls]['f1'])
            lines.append(f"  {cls:<22} {pm:.4f} ± {ps:.4f}   "
                         f"{rm:.4f} ± {rs:.4f}   {fm:.4f} ± {fs:.4f}")

    # Cross-scenario ranking table
    lines += ["", "=" * W2,
              f"  RANKING  (by mean test QWK, descending)", "=" * W2,
              f"  {'#':<3} {'Scenario':<18} {'QWK mean':>10} {'± Std':>8} "
              f"{'Acc (%)':>10} {'MAE':>8} {'RMSE':>8}",
              "  " + "-" * 68]
    for i, row in enumerate(sorted(summary_rows,
                                   key=lambda r: np.mean(r['qwk']), reverse=True), 1):
        qm, qs = mu_sd(row['qwk'])
        lines.append(f"  {i:<3} {row['scenario']:<18} {qm:>10.4f} {qs:>8.4f} "
                     f"{np.mean(row['acc']):>10.2f} {np.mean(row['mae']):>8.4f} "
                     f"{np.mean(row['rmse']):>8.4f}")

    lines += ["=" * W2, ""]
    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS_DIR / 'baseline_results.csv'
    if csv_path.exists():
        csv_path.unlink()

    print("\n" + "=" * W)
    print("  01_baseline — Standalone Model Evaluation (CE / CORAL / CORN / Niu)")
    print("=" * W)
    print(f"  Project root : {PROJECT_ROOT}")
    print(f"  Results dir  : {RESULTS_DIR}")
    print(f"  Device       : {device}")

    print("\n[Dataset]")
    test_dir     = get_dataset_dir()
    val_tfm      = transforms.Compose([
        transforms.ToTensor(), transforms.Normalize(_MEAN, _STD)])
    test_dataset = datasets.ImageFolder(test_dir, transform=val_tfm)
    test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=0, pin_memory=False)
    print(f"  Test dir     : {test_dir}")
    print(f"  Test samples : {len(test_dataset)}")
    print(f"  Classes (folder → idx): {test_dataset.class_to_idx}")
    if test_dataset.classes != [str(i) for i in range(NUM_CLASSES)]:
        print(f"  ⚠ Class folder names {test_dataset.classes} don't look like "
              f"'0'..'{NUM_CLASSES-1}' — double-check CLASS_NAMES ordering above.")

    summary_rows = []

    for scenario in SCENARIOS:
        print(f"\n\n{'='*W}")
        print(f"  {scenario['name']}")
        print(f"{'='*W}")

        seed_results = []
        best_qwks    = {}

        for seed in SEEDS:
            fname     = f"{scenario['prefix']}_seed{seed}_best.pth"
            ckpt_path = scenario['model_dir'] / fname
            print(f"\n  Seed {seed} — {fname}")

            if not ckpt_path.exists():
                print(f"  ⚠ Not found: {ckpt_path}"); continue
            try:
                model, ckpt = load_model(ckpt_path, scenario['head'])
            except Exception as e:
                print(f"  ⚠ Load failed: {e}"); continue

            raw_bq = ckpt.get('best_qwk') or ckpt.get('qwk')
            best_qwks[seed] = f"{raw_bq:.4f}" if isinstance(raw_bq, float) else str(raw_bq)
            print(f"  Training best val QWK: {best_qwks[seed]}")
            print(f"  Evaluating ...")

            res         = evaluate(model, scenario['head'], test_loader)
            res['seed'] = seed
            seed_results.append(res)
            del model

            print(f"  QWK={res['qwk']:.4f} | Acc={res['acc']*100:.2f}% | "
                  f"MAE={res['mae']:.4f} | RMSE={res['rmse']:.4f}")
            sys.stdout.flush()

        if not seed_results:
            print("  ⚠ No seeds loaded."); continue

        block    = format_results(scenario, seed_results, best_qwks)
        out_path = RESULTS_DIR / f"{scenario['short']}_test_results.txt"
        out_path.write_text(block, encoding='utf-8')
        print(f"\n  Saved → {out_path.name}")
        write_csv(csv_path, scenario['short'], scenario['head'], seed_results)

        summary_rows.append({
            'scenario': scenario['short'],
            'qwk':      [r['qwk']       for r in seed_results],
            'acc':      [r['acc']*100    for r in seed_results],
            'mae':      [r['mae']        for r in seed_results],
            'rmse':     [r['rmse']       for r in seed_results],
            'macro_p':  [r['macro_p']    for r in seed_results],
            'macro_r':  [r['macro_r']    for r in seed_results],
            'macro_f1': [r['macro_f1']   for r in seed_results],
            'wtd_p':    [r['wtd_p']      for r in seed_results],
            'wtd_r':    [r['wtd_r']      for r in seed_results],
            'wtd_f1':   [r['wtd_f1']     for r in seed_results],
            'per_class': {
                cls: {
                    'p':  [r['per_class'][cls]['p']  for r in seed_results],
                    'r':  [r['per_class'][cls]['r']  for r in seed_results],
                    'f1': [r['per_class'][cls]['f1'] for r in seed_results],
                } for cls in CLASS_NAMES
            },
        })

    if summary_rows:
        summary = format_summary(summary_rows,
                                 'SUMMARY — 01_baseline (CE / CORAL / CORN / Niu)')
        print("\n\n" + summary)
        (RESULTS_DIR / 'baseline_summary.txt').write_text(summary, encoding='utf-8')
        print(f"  Summary → baseline_summary.txt")

    print(f"  CSV     → {csv_path.name}\n\nDone.\n")


main()