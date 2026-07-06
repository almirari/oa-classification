import sys, zipfile, csv
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

PROJECT_ROOT = SCRIPT_DIR.parent
DATASET_ZIP  = PROJECT_ROOT / 'dataset.zip'
RESULTS_DIR  = SCRIPT_DIR
EXTRACT_DIR  = PROJECT_ROOT / '_dataset_extracted'
SEEDS        = [33, 81, 5]
_MULTI       = PROJECT_ROOT / '04_kd_multi' / 'models'

STUDENT_SCENARIOS = [
    {
        'name':        'Multi-Teacher KD — CE  (ResNet-50-CE + EffNetB3-CE → MobileNetV3-Small)',
        'short':       'multi_ce_resnet50_effnetb3',
        'notebook':    'oa-multiteacher-ce.ipynb',
        'model_dir':   _MULTI / 'ce_resnet50_effnetb3',
        'ckpt_prefix': 'kd',                   # kd_seed{seed}_best.pth
    },
    {
        'name':        'Multi-Teacher KD — CORAL  (ResNet-50-CORAL + EffNetB3-CORAL → MobileNetV3-Small)',
        'short':       'multi_coral_resnet50_effnetb3',
        'notebook':    'oa-multiteacher-coral.ipynb',
        'model_dir':   _MULTI / 'coral_resnet50_effnetb3',
        'ckpt_prefix': 'coral_kd',             # coral_kd_seed{seed}_best.pth
    },
    {
        'name':        'Multi-Teacher KD — Hybrid CA-MKD  (ResNet-50-CE + EffNetB3-CORAL → MobileNetV3-Small)',
        'short':       'multi_hybrid_resnet50_effnetb3',
        'notebook':    'oa-multiteacher-hybrid-resnet50-effnetb3.ipynb',
        'model_dir':   _MULTI / 'hybrid_resnet50_effnetb3',
        'ckpt_prefix': 'hybrid_camkd',         # hybrid_camkd_seed{seed}_best.pth
    },
    {
        'name':        'Multi-Teacher KD — Hybrid CA-MKD  (ResNet-50-CE + ResNet-50-CORAL → MobileNetV3-Small)',
        'short':       'multi_hybrid_resnet50_resnet50',
        'notebook':    'oa-multiteacher-hybrid-resnet50-resnet50.ipynb',
        'model_dir':   _MULTI / 'hybrid_resnet50_resnet50',
        'ckpt_prefix': 'hybrid_camkd',         # hybrid_camkd_seed{seed}_best.pth
    },
]

TEACHER_SCENARIOS = [
    {
        'name':     'Teacher — ResNet-50-CE  (CE)',
        'short':    'teacher_ce_resnet50',
        'notebook': 'mkd_ce_resnet50_efficientnetb3.ipynb',
        'ckpt':     _MULTI / 'ce_resnet50_effnetb3' / 'kd_teacher_resnet50.pth',
    },
    {
        'name':     'Teacher — EfficientNetB3-CE  (CE)',
        'short':    'teacher_ce_effnetb3',
        'notebook': 'mkd_ce_resnet50_efficientnetb3.ipynb',
        'ckpt':     _MULTI / 'ce_resnet50_effnetb3' / 'kd_teacher_efficientnet_b3.pth',
    },
    {
        'name':     'Teacher — ResNet-50-CORAL  (CORAL)',
        'short':    'teacher_coral_resnet50',
        'notebook': 'mkd_coral_resnet50_efficientnetb3.ipynb',
        'ckpt':     _MULTI / 'coral_resnet50_effnetb3' / 'coral_kd_teacher_resnet50.pth',
    },
    {
        'name':     'Teacher — EfficientNetB3-CORAL  (CORAL)',
        'short':    'teacher_coral_effnetb3',
        'notebook': 'mkd_coral_resnet50_efficientnetb3.ipynb',
        'ckpt':     _MULTI / 'coral_resnet50_effnetb3' / 'coral_kd_teacher_efficientnet_b3.pth',
    },
    {
        'name':     'Teacher — ResNet-50-CE  (hybrid resnet50+effnetb3)',
        'short':    'teacher_hybrid_resnet50effnetb3_resnet50_ce',
        'notebook': 'mkd_hybrid_resnet50_efficientnetb3.ipynb',
        'ckpt':     _MULTI / 'hybrid_resnet50_effnetb3' / 'hybrid_teacher_resnet50_ce.pth',
    },
    {
        'name':     'Teacher — EfficientNetB3-CORAL  (hybrid resnet50+effnetb3)',
        'short':    'teacher_hybrid_resnet50effnetb3_effnetb3_coral',
        'notebook': 'mkd_hybrid_resnet50_efficientnetb3.ipynb',
        'ckpt':     _MULTI / 'hybrid_resnet50_effnetb3' / 'hybrid_teacher_efficientnet_coral.pth',
    },
    {
        'name':     'Teacher — ResNet-50-CE  (hybrid resnet50+resnet50)',
        'short':    'teacher_hybrid_resnet50resnet50_ce',
        'notebook': 'mkd_hybrid_resnet50_resnet50.ipynb',
        'ckpt':     _MULTI / 'hybrid_resnet50_resnet50' / 'hybrid_teacher_resnet50_ce.pth',
    },
    {
        'name':     'Teacher — ResNet-50-CORAL  (hybrid resnet50+resnet50)',
        'short':    'teacher_hybrid_resnet50resnet50_coral',
        'notebook': 'mkd_hybrid_resnet50_resnet50.ipynb',
        'ckpt':     _MULTI / 'hybrid_resnet50_resnet50' / 'hybrid_teacher_resnet50_coral.pth',
    },
]

# NOTE: assumed Kellgren-Lawrence (KL) knee-OA grading, 5 classes (0-4).
# Verify this against your actual dataset/CSV label scheme and edit if needed.
CLASS_NAMES = ['Grade 0 (Normal)', 'Grade 1 (Doubtful)', 'Grade 2 (Minimal)',
               'Grade 3 (Moderate)', 'Grade 4 (Severe)']
NUM_CLASSES = len(CLASS_NAMES)
BATCH_SIZE  = 32
device      = torch.device('cpu')
_MEAN       = [0.485, 0.456, 0.406]
_STD        = [0.229, 0.224, 0.225]


def _find_test_dir(root):
    """Find a 'test' folder under root that itself contains class subfolders."""
    for p in sorted(root.rglob('test')):
        if p.is_dir() and any(c.is_dir() for c in p.iterdir()):
            return p
    return None


def get_dataset_dir():
    """Returns the path to the 'test' folder, which must contain one
    subfolder per class (e.g. test/0, test/1, ... test/4) — the standard
    torchvision.datasets.ImageFolder layout. There is no CSV in this dataset."""
    if EXTRACT_DIR.exists():
        test_dir = _find_test_dir(EXTRACT_DIR)
        if test_dir:
            print(f"  Dataset already extracted → {test_dir}")
            return test_dir
    if not DATASET_ZIP.exists():
        raise FileNotFoundError(f"dataset.zip not found at {DATASET_ZIP}")
    print(f"  Extracting {DATASET_ZIP.name} ...")
    EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(DATASET_ZIP, 'r') as zf:
        zf.extractall(EXTRACT_DIR)
    test_dir = _find_test_dir(EXTRACT_DIR)
    if not test_dir:
        raise FileNotFoundError("No 'test/<class>/...' folder structure found inside dataset.zip")
    print(f"  Extracted → {test_dir}")
    return test_dir


class CoralHead(nn.Module):
    def __init__(self, in_features, num_classes):
        super().__init__()
        self.fc   = nn.Linear(in_features, 1, bias=False)
        self.bias = nn.Parameter(torch.zeros(num_classes - 1))
    def forward(self, x):
        return self.fc(x) + self.bias


def get_mobilenet_ce():
    m = models.mobilenet_v3_small(weights=None)
    m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, NUM_CLASSES)
    return m


def get_mobilenet_coral():
    m = models.mobilenet_v3_small(weights=None)
    in_f = m.classifier[-1].in_features
    m.classifier[-1] = nn.Identity()
    m.coral_head = CoralHead(in_f, NUM_CLASSES)
    return m


def get_resnet50_ce():
    m = models.resnet50(weights=None)
    m.fc = nn.Linear(m.fc.in_features, NUM_CLASSES)
    return m


def get_resnet50_coral():
    m = models.resnet50(weights=None)
    in_f = m.fc.in_features
    m.fc = nn.Identity()
    m.coral_head = CoralHead(in_f, NUM_CLASSES)
    return m


def get_efficientnetb3_ce():
    m = models.efficientnet_b3(weights=None)
    m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, NUM_CLASSES)
    return m


def get_efficientnetb3_coral():
    m = models.efficientnet_b3(weights=None)
    in_f = m.classifier[-1].in_features
    m.classifier[-1] = nn.Identity()
    m.coral_head = CoralHead(in_f, NUM_CLASSES)
    return m


def load_student(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd   = ckpt['model_state_dict']
    for builder, label in [(get_mobilenet_ce, 'CE'), (get_mobilenet_coral, 'CORAL')]:
        m = builder().to(device)
        try:
            m.load_state_dict(sd, strict=True)
            return m, label, ckpt
        except RuntimeError:
            continue
    raise RuntimeError(
        f"Neither CE nor CORAL student head matched.\n"
        f"First 10 keys: {list(sd.keys())[:10]}\nPath: {ckpt_path}"
    )


TEACHER_BUILDERS = [
    (get_resnet50_ce,          'CE',    'ResNet-50'),
    (get_resnet50_coral,       'CORAL', 'ResNet-50'),
    (get_efficientnetb3_ce,    'CE',    'EfficientNetB3'),
    (get_efficientnetb3_coral, 'CORAL', 'EfficientNetB3'),
]


def load_teacher(ckpt_path):
    raw = torch.load(ckpt_path, map_location=device, weights_only=False)
    # Support checkpoints wrapped under a key, or bare state dicts
    sd = (raw.get('model_state_dict')
          or raw.get('state_dict')
          or (raw if isinstance(raw, dict) else None))
    if sd is None:
        raise RuntimeError(f"Unexpected checkpoint format at {ckpt_path}")
    for builder, head_label, arch_label in TEACHER_BUILDERS:
        m = builder().to(device)
        try:
            m.load_state_dict(sd, strict=True)
            return m, head_label, arch_label, raw
        except RuntimeError:
            continue
    raise RuntimeError(
        f"No teacher architecture matched.\n"
        f"First 10 keys: {list(sd.keys())[:10]}\nPath: {ckpt_path}"
    )


def predict_batch(model, imgs, head):
    """Works for MobileNetV3-Small, ResNet-50, and EfficientNetB3 (CE and CORAL).

    For CORAL models the last classification layer is nn.Identity(), so model()
    returns the feature vector directly — no need to call internal sub-modules.
    """
    if head == 'CE':
        return model(imgs).argmax(dim=1)
    feats = model(imgs)   # Identity tail → feature vector (B, in_features)
    return (torch.sigmoid(model.coral_head(feats)) > 0.5).sum(dim=1).long()


def evaluate(model, head, loader):
    model.eval()
    preds_all, labels_all = [], []
    n = len(loader)
    with torch.no_grad():
        for i, (imgs, labels) in enumerate(loader):
            preds_all.extend(predict_batch(model, imgs.to(device), head).cpu().numpy())
            labels_all.extend(labels.numpy())
            print(f"\r    batch {i+1}/{n}", end='', flush=True)
    print()
    preds  = np.array(preds_all)
    labels = np.array(labels_all)
    report = classification_report(
        labels, preds, target_names=CLASS_NAMES, output_dict=True, zero_division=0
    )
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


def mu_sd(v): return np.mean(v), np.std(v)
W = 76


def section(title):
    return [f"\n  ── {title} " + "─" * max(0, W - 6 - len(title))]


def format_results(scenario, seed_results, best_qwks):
    L = []
    L += ["=" * W,
          f"  {scenario['name']}",
          f"  Notebook : {scenario['notebook']}",
          "=" * W]

    for r in seed_results:
        bq = best_qwks.get(r['seed'], 'N/A')
        L += section(f"Seed {r['seed']}  (training best val QWK: {bq})")
        L += [
            f"  Head : {r['head']}",
            "",
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
            ('QWK',      'qwk'),
            ('Acc (%)',   None),
            ('MAE',       'mae'),
            ('RMSE',      'rmse'),
            ('Macro P',   'macro_p'),
            ('Macro R',   'macro_r'),
            ('Macro F1',  'macro_f1'),
            ('Wtd P',     'wtd_p'),
            ('Wtd R',     'wtd_r'),
            ('Wtd F1',    'wtd_f1'),
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


def format_teacher_results(scenario, res):
    """Format results for a single-checkpoint teacher (no seed loop)."""
    L = []
    L += ["=" * W,
          f"  {scenario['name']}",
          f"  Notebook   : {scenario['notebook']}",
          f"  Checkpoint : {Path(scenario['ckpt']).name}",
          "=" * W]
    L += [
        f"  Architecture : {res['arch']}  |  Head : {res['head']}",
        "",
        f"  {'Metric':<14} {'Value':>10}",
        "  " + "-" * 26,
        f"  {'QWK':<14} {res['qwk']:>10.4f}",
        f"  {'Accuracy':<14} {res['acc']*100:>9.2f}%",
        f"  {'MAE':<14} {res['mae']:>10.4f}",
        f"  {'RMSE':<14} {res['rmse']:>10.4f}",
        f"  {'Macro P':<14} {res['macro_p']:>10.4f}",
        f"  {'Macro R':<14} {res['macro_r']:>10.4f}",
        f"  {'Macro F1':<14} {res['macro_f1']:>10.4f}",
        f"  {'Wtd P':<14} {res['wtd_p']:>10.4f}",
        f"  {'Wtd R':<14} {res['wtd_r']:>10.4f}",
        f"  {'Wtd F1':<14} {res['wtd_f1']:>10.4f}",
        "",
        f"  {'Class':<22} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>8}",
        "  " + "-" * 62,
    ]
    for cls in CLASS_NAMES:
        pc = res['per_class'][cls]
        L.append(f"  {cls:<22} {pc['p']:>10.4f} {pc['r']:>10.4f} "
                 f"{pc['f1']:>10.4f} {pc['support']:>8d}")
    L.append("")
    return "\n".join(L)


def format_teacher_summary(teacher_results, title):
    """Aggregated view of all teacher results: full metrics + per-class P/R/F1."""
    W2 = 90
    lines = ["=" * W2, f"  {title}", "=" * W2]

    for r in teacher_results:
        arch = r.get('arch', '')
        tag  = f"{arch}-{r['head']}" if arch else r['head']
        sep  = "─" * max(0, W2 - 8 - len(r['short']) - len(tag))
        lines += [
            f"\n  ── {r['short']}  ({tag}) {sep}",
            "",
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
            lines.append(f"  {cls:<22} {pc['p']:>10.4f} {pc['r']:>10.4f} "
                         f"{pc['f1']:>10.4f} {pc['support']:>8d}")

    lines += ["", "=" * W2, ""]
    return "\n".join(lines)


def write_csv(csv_path, scenario_short, seed_results):
    exists   = csv_path.exists()
    cls_keys = [(cls,
                 f'p_{cls.lower().replace(" ","_")}',
                 f'r_{cls.lower().replace(" ","_")}',
                 f'f1_{cls.lower().replace(" ","_")}') for cls in CLASS_NAMES]
    fieldnames = (['scenario', 'seed', 'head', 'qwk', 'acc_pct', 'mae', 'rmse',
                   'macro_p', 'macro_r', 'macro_f1', 'wtd_p', 'wtd_r', 'wtd_f1']
                  + [k for _, p, r, f in cls_keys for k in (p, r, f)])
    with open(csv_path, 'a', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            w.writeheader()
        for r in seed_results:
            row = {
                'scenario': scenario_short,
                'seed':     r['seed'],
                'head':     r['head'],
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


def write_csv_teacher(csv_path, scenario_short, res):
    write_csv(csv_path, scenario_short, [{**res, 'seed': 'shared'}])


def format_summary(summary_rows, title):
    W2 = 90
    lines = [
        "=" * W2,
        f"  {title}  (mean ± std across seeds)",
        "=" * W2,
    ]

    for row in summary_rows:
        lines += [
            f"\n  Scenario : {row['scenario']}",
            "",
            f"  {'Metric':<14} {'Mean':>10} {'± Std':>10} {'Min':>10} {'Max':>10}",
            "  " + "-" * 54,
        ]
        for label, key in [
            ('QWK',      'qwk'),
            ('Acc (%)',  'acc'),
            ('MAE',      'mae'),
            ('RMSE',     'rmse'),
            ('Macro P',  'macro_p'),
            ('Macro R',  'macro_r'),
            ('Macro F1', 'macro_f1'),
            ('Wtd P',    'wtd_p'),
            ('Wtd R',    'wtd_r'),
            ('Wtd F1',   'wtd_f1'),
        ]:
            vals = row[key]
            lines.append(f"  {label:<14} {np.mean(vals):>10.4f} {np.std(vals):>10.4f} "
                         f"{min(vals):>10.4f} {max(vals):>10.4f}")

        lines += [
            "",
            f"  {'Class':<22} {'Precision':>16} {'Recall':>16} {'F1-Score':>16}",
            "  " + "-" * 72,
        ]
        for cls in CLASS_NAMES:
            pv = row['per_class'][cls]['p']
            rv = row['per_class'][cls]['r']
            fv = row['per_class'][cls]['f1']
            pm, ps = np.mean(pv), np.std(pv)
            rm, rs = np.mean(rv), np.std(rv)
            fm, fs = np.mean(fv), np.std(fv)
            lines.append(f"  {cls:<22} {pm:.4f} ± {ps:.4f}   "
                         f"{rm:.4f} ± {rs:.4f}   {fm:.4f} ± {fs:.4f}")

    lines += ["=" * W2, ""]
    return "\n".join(lines)

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS_DIR / 'multi_kd_results.csv'
    if csv_path.exists():
        csv_path.unlink()

    print("\n" + "=" * W)
    print("  04_kd_multi — Student + Teacher Evaluation")
    print("=" * W)
    print(f"  Project root : {PROJECT_ROOT}")
    print(f"  Results dir  : {RESULTS_DIR}")

    print("\n[Dataset]")
    test_dir     = get_dataset_dir()
    val_tfm      = transforms.Compose([
        transforms.ToTensor(), transforms.Normalize(_MEAN, _STD)])
    test_dataset = datasets.ImageFolder(test_dir, transform=val_tfm)
    test_loader  = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=False)
    print(f"  Test dir     : {test_dir}")
    print(f"  Test samples : {len(test_dataset)}")
    print(f"  Classes (folder → idx): {test_dataset.class_to_idx}")
    if test_dataset.classes != [str(i) for i in range(NUM_CLASSES)]:
        print(f"  ⚠ Class folder names {test_dataset.classes} don't look like "
              f"'0'..'{NUM_CLASSES-1}' — double-check CLASS_NAMES ordering above.")

    summary_rows = []

    print(f"\n\n{'#'*W}")
    print("  STUDENT MODELS")
    print(f"{'#'*W}")

    for scenario in STUDENT_SCENARIOS:
        print(f"\n\n{'='*W}")
        print(f"  {scenario['name']}")
        print(f"{'='*W}")

        seed_results = []
        best_qwks    = {}

        for seed in SEEDS:
            fname     = f"{scenario['ckpt_prefix']}_seed{seed}_best.pth"
            ckpt_path = scenario['model_dir'] / fname
            print(f"\n  Seed {seed} — {fname}")

            if not ckpt_path.exists():
                print(f"  ⚠ Not found: {ckpt_path}"); continue
            try:
                model, head, ckpt = load_student(ckpt_path)
            except Exception as e:
                print(f"  ⚠ Load failed: {e}"); continue

            raw_bq = ckpt.get('best_qwk') or ckpt.get('qwk')
            best_qwks[seed] = f"{raw_bq:.4f}" if isinstance(raw_bq, float) else str(raw_bq)
            print(f"  Head : {head}  |  Training best val QWK: {best_qwks[seed]}")
            print(f"  Evaluating ...")

            model.eval()
            res         = evaluate(model, head, test_loader)
            res['seed'] = seed
            res['head'] = head
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
        write_csv(csv_path, scenario['short'], seed_results)

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
        summary = format_summary(summary_rows, 'SUMMARY — 04_kd_multi (students)')
        print("\n\n" + summary)
        (RESULTS_DIR / 'multi_kd_summary.txt').write_text(summary, encoding='utf-8')
        print(f"  Summary → multi_kd_summary.txt")

    print(f"\n\n{'#'*W}")
    print("  TEACHER MODELS")
    print(f"{'#'*W}")

    teacher_results = []

    for scenario in TEACHER_SCENARIOS:
        print(f"\n\n{'='*W}")
        print(f"  {scenario['name']}")
        print(f"{'='*W}")

        ckpt_path = Path(scenario['ckpt'])
        print(f"\n  Checkpoint — {ckpt_path.name}")

        if not ckpt_path.exists():
            print(f"  ⚠ Not found: {ckpt_path}"); continue
        try:
            model, head, arch, ckpt = load_teacher(ckpt_path)
        except Exception as e:
            print(f"  ⚠ Load failed: {e}"); continue

        print(f"  Architecture : {arch}  |  Head : {head}")
        print(f"  Evaluating ...")

        model.eval()
        res          = evaluate(model, head, test_loader)
        res['head']  = head
        res['arch']  = arch
        res['short'] = scenario['short']
        del model

        print(f"  QWK={res['qwk']:.4f} | Acc={res['acc']*100:.2f}% | "
              f"MAE={res['mae']:.4f} | RMSE={res['rmse']:.4f}")
        sys.stdout.flush()

        block    = format_teacher_results(scenario, res)
        out_path = RESULTS_DIR / f"{scenario['short']}_test_results.txt"
        out_path.write_text(block, encoding='utf-8')
        print(f"\n  Saved → {out_path.name}")
        write_csv_teacher(csv_path, scenario['short'], res)

        teacher_results.append(res)

    if teacher_results:
        t_summary = format_teacher_summary(
            teacher_results, 'TEACHER SUMMARY — 04_kd_multi')
        print("\n\n" + t_summary)
        (RESULTS_DIR / 'multi_kd_teacher_summary.txt').write_text(
            t_summary, encoding='utf-8')
        print(f"  Teacher summary → multi_kd_teacher_summary.txt")

    print(f"\n  CSV → {csv_path.name}\n\nDone.\n")


main()