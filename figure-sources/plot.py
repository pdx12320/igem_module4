#!/usr/bin/env python3
"""Regenerate five ORCA wiki figures from preserved results, without model execution."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from orca_plot import Theme, export


ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
SLUGS = (
    "pretraining-curves", "stage1-training-loss", "stage1-validation-ap",
    "stage1-test-pr", "stage2-ranking-reported",
)
VARIANTS = (
    ("Adapted, tuned", "1024_orca_puf12", "final_model/log_history.json"),
    ("Original, tuned", "1024_lamar_puf12", "final_model/checkpoint-1380/trainer_state.json"),
    ("Adapted, frozen", "1024_orca_puf12_frozen_encoder", "final_model/checkpoint-1260/trainer_state.json"),
    ("Original, frozen", "1024_lamar_puf12_frozen_encoder", "final_model/checkpoint-1440/trainer_state.json"),
    ("Random, tuned", "1024_orca_puf12_random", "final_model/checkpoint-1080/trainer_state.json"),
)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def source(path, read_range):
    return {"path": path.relative_to(ROOT).as_posix(),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "read_range": read_range}


def write_csv(path, rows):
    if not rows:
        raise ValueError("Refusing an empty plotted-data table")
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def check_series(rows, x_key, y_key):
    x = np.asarray([r[x_key] for r in rows], float)
    y = np.asarray([r[y_key] for r in rows], float)
    if not len(x) or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError(f"Invalid numeric observations: {x_key}/{y_key}")
    if np.any(np.diff(x) <= 0):
        raise ValueError(f"Nonchronological or duplicate trajectory positions: {x_key}")
    return x, y


def finish(fig, slug, theme, destination, rows, caption, alt, provenance, summary=None):
    out = destination / slug
    provenance = {
        "audience": "iGEM wiki, 960 CSS px display width",
        "uncertainty": "No error intervals or significance tests; no independent replicate estimates supplied",
        "exclusions": "No observations removed for outlier status or aesthetics",
        "smoothing": "None", "random_seed": None,
        "plot_script": source(HERE / "plot.py", "complete script"),
        **provenance,
    }
    export(fig, out, theme, caption=caption, alt_text=alt, provenance=provenance)
    write_csv(out / "plotted-data.csv", rows)
    if summary:
        write_csv(out / "summary.csv", summary)
    plt.close(fig)
    print(f"Exported {slug}: {len(rows)} data records")


def pretraining(theme, destination):
    path = ROOT / "results/model_continual_pretrain/transcriptome_pretrain/checkpoint-100000/trainer_state.json"
    history = read_json(path)["log_history"]
    train = [r for r in history if "loss" in r]
    valid = [r for r in history if "eval_loss" in r]
    tx, ty = check_series(train, "step", "loss")
    vx, vy = check_series(valid, "step", "eval_loss")
    _, accuracy = check_series(valid, "step", "eval_masked_accuracy")
    if len(train) != 1000 or len(valid) != 100 or tx[-1] != 100000:
        raise ValueError("Pretraining log invariants changed; review before publication")
    rows = []
    for i, r in enumerate(history):
        for key in ("loss", "eval_loss", "eval_masked_accuracy"):
            if key in r:
                rows.append({"source_log_index": i, "step": r["step"], "epoch": r["epoch"],
                             "metric": key, "value": r[key]})
    fig, axes = plt.subplots(2, 1, figsize=(10, 8.4), sharex=True)
    fig.subplots_adjust(left=.15, right=.96, top=.88, bottom=.12, hspace=.29)
    axes[0].plot(tx / 1000, ty, color=theme.c["indigo700"], linewidth=1.6, label="Training loss")
    axes[0].plot(vx / 1000, vy, color=theme.c["orange500"], marker="o", markersize=3.5,
                 markeredgecolor=theme.c["ink"], markeredgewidth=.55, label="Validation loss")
    axes[0].set(ylabel="Loss", ylim=(.9, 1.11))
    axes[1].plot(vx / 1000, accuracy, color=theme.c["teal600"], marker="s", markersize=3.5,
                 markeredgecolor=theme.c["ink"], markeredgewidth=.55)
    axes[1].set(xlabel="Optimizer step (×1,000)", ylabel="Masked accuracy", ylim=(.50, .60), xlim=(0, 101))
    axes[1].set_xticks([0, 20, 40, 60, 80, 100])
    axes[0].legend(loc="lower center", bbox_to_anchor=(.5, 1.05), ncol=2)
    for letter, ax in zip("ab", axes):
        ax.text(-.13, 1.01, letter, transform=ax.transAxes, fontsize=24)
        ax.tick_params(length=6)
    finish(fig, SLUGS[0], theme, destination, rows,
        "Continued-pretraining history. (a) All 1,000 logged training-loss values and all 100 validation-loss values are shown against optimizer step. (b) Validation masked-token accuracy at the same 100 evaluations. The final logged accuracy is 0.575468 at step 100,000; the earliest logged validation accuracy is 0.532084 at step 1,000. The supplied log contains no step-0 evaluation, so no before-adaptation baseline is plotted. Training-loss values are trainer logging-window summaries, not individual RNA observations; accuracy is the proportion correct at masked token positions. The evaluation script defaults to at most 10,000 validation sequences, but the precise masked-token denominator is not preserved in the log. Curves connect logged values without smoothing or added uncertainty. Loss and accuracy use separate, explicitly nonzero y-axis ranges. One training run is shown; these are computational logs, not biological replicate measurements.",
        "Two vertically aligned plots show continued-pretraining loss and validation masked-token accuracy through 100,000 optimizer steps. Accuracy rises from the earliest logged value of about 0.532 to 0.575; loss generally decreases after an early rise. No step-zero baseline is available.",
        {"sources": [source(path, "log_history[*]: loss, eval_loss, eval_masked_accuracy, step, epoch")],
         "chart_type": "two aligned line panels; distinct metrics use separate axes",
         "analysis_unit": "logged optimizer interval or validation evaluation; one run",
         "n": {"training_loss_records": 1000, "validation_evaluations": 100},
         "transforms": ["step divided by 1000 for display only"],
         "axis_ranges": {"loss": [.9, 1.11], "masked_accuracy": [.5, .6]},
         "missing": "No step-0 validation result in archive; not imputed"},
        [{"metric": "final_masked_accuracy", "step": int(vx[-1]), "value": float(accuracy[-1])}])


def stage1_histories():
    histories, sources, rows, summary = {}, [], [], []
    for name, directory, suffix in VARIANTS:
        path = ROOT / "results/model_1024_classifier" / directory / suffix
        blob = read_json(path)
        history = blob if isinstance(blob, list) else blob["log_history"]
        sources.append(source(path, "log_history, or complete list: loss and eval_pr_auc with epoch and step"))
        post_train = False
        trains, validations = [], []
        for index, record in enumerate(history):
            if "train_runtime" in record:
                post_train = True
            for key in ("loss", "eval_pr_auc"):
                if key not in record:
                    continue
                restored = post_train and key == "eval_pr_auc"
                rows.append({"group": name, "source_path": path.relative_to(ROOT).as_posix(),
                             "source_log_index": index, "epoch": record["epoch"], "step": record["step"],
                             "metric": key, "value": record[key],
                             "record_role": "restored_best_checkpoint_evaluation" if restored else "training_trajectory",
                             "plotted_as_trajectory": not restored})
                if key == "loss":
                    trains.append(record)
                elif not restored:
                    validations.append(record)
        check_series(trains, "epoch", "loss")
        check_series(validations, "epoch", "eval_pr_auc")
        histories[name] = (trains, validations)
        best = max(validations, key=lambda r: r["eval_pr_auc"])
        summary.append({"group": name, "training_loss_records": len(trains),
                        "validation_records": len(validations), "last_logged_epoch": trains[-1]["epoch"],
                        "best_logged_validation_ap": best["eval_pr_auc"], "best_epoch": best["epoch"],
                        "best_step": best["step"]})
    return histories, sources, rows, summary


def stage1_trajectories(theme, destination):
    histories, sources, rows, summary = stage1_histories()
    for slug, metric, ylabel, ylim in (
        (SLUGS[1], "loss", "Training loss", (0, .72)),
        (SLUGS[2], "eval_pr_auc", "Validation average precision", (.65, .85)),
    ):
        fig, ax = plt.subplots(figsize=(10, 7.5))
        fig.subplots_adjust(left=.15, right=.965, top=.96, bottom=.32)
        for name, (train, validation) in histories.items():
            observations = train if metric == "loss" else validation
            x, y = check_series(observations, "epoch", metric)
            style = theme.style(name)
            ax.plot(x, y, label=name, color=style["color"], marker=style["marker"],
                    linestyle=style["linestyle"], markersize=5.5, markeredgecolor=theme.c["ink"],
                    markeredgewidth=.5, linewidth=2)
        ax.set(xlabel="Epoch", ylabel=ylabel, xlim=(0, 12.3), ylim=ylim)
        ax.set_xticks([0, 2, 4, 6, 8, 10, 12])
        if metric == "eval_pr_auc":
            ax.axhline(150 / 213, color=theme.c["line"], linewidth=1.5, linestyle=(0, (2, 2)), zorder=0)
            ax.set_yticks([.65, .70, .75, .80, .85])
        ax.legend(loc="upper center", bbox_to_anchor=(.5, -.19), ncol=2, columnspacing=1.2, handlelength=2)
        if metric == "loss":
            caption = "Stage I training trajectories. All available logged training-loss summaries are shown for five model variants: transcriptome-adapted LAMAR with all parameters tuned, original LAMAR with all parameters tuned, their frozen-encoder counterparts, and random initialization with all parameters tuned. Each line represents one run; the training split contains 3,835 region records. Logs end at different epochs, and no values are extrapolated beyond them. The adapted run uses the full final-run log through epoch 7; the other variants use their supplied checkpoint histories. Values are logging-window training summaries, not independent replicates. Lines connect observations without smoothing, uncertainty intervals, or significance tests. The configurations and run lengths differ, so the curves alone do not establish the causal benefit of adaptation."
            alt = "Training-loss curves for five Stage I model variants. Adapted and original fully tuned models reach much lower training loss than frozen-encoder or random-initialized variants. Each curve ends where its saved log ends; the runs are not independent replicates."
        else:
            caption = "Stage I validation trajectories. Average precision (AP; named eval_pr_auc in the source logs) is shown for five model variants on 213 region records, including 150 positives. The faint horizontal line marks positive-label prevalence (150/213 = 0.704225). The displayed y-axis runs from 0.65 to 0.85 and includes every recorded value. All chronological validation evaluations are plotted; the full adapted-run log ends with a separate reevaluation after restoration of the best checkpoint. That record repeats AP 0.810014 with epoch-7 metadata and is retained in plotted-data.csv with its restored-checkpoint role, but is not connected as a new epoch-7 model update. The best logged AP is 0.818051 for original frozen LAMAR and 0.810014 for adapted tuned LAMAR; no superiority of adaptation is inferred. Source scripts compute average precision rather than trapezoidal PR area. Runs have different lengths and configurations; no replicate uncertainty or significance tests are available."
            alt = "Five Stage I validation average-precision trajectories fluctuate across epochs. Original frozen LAMAR reaches the highest logged AP, about 0.818, while adapted tuned LAMAR peaks near 0.810. A faint reference line indicates validation positive-label prevalence, about 0.704."
        finish(fig, slug, theme, destination, [r for r in rows if r["metric"] == metric], caption, alt,
            {"sources": sources, "chart_type": "line with all logged observation markers; fixed variant styles",
             "analysis_unit": "one training run per variant; logged training intervals or validation evaluations",
             "n": {"training_region_records": 3835, "validation_region_records": 213, "validation_positives": 150},
             "transforms": [], "axis_range": list(ylim),
             "missing": "Logs have unequal endpoints; no extrapolation or imputation",
             "record_handling": "Post-training restored-checkpoint evaluation retained in data, excluded only from the chronological training curve"}, summary)


def test_pr(theme, destination):
    folder = ROOT / "results/model_1024_classifier/1024_orca_puf12/test_evaluation"
    path, metric_path = folder / "test_predictions.csv", folder / "test_metrics.json"
    with path.open(encoding="utf-8", newline="") as stream:
        records = list(csv.DictReader(stream))
    labels = np.asarray([int(r["label"]) for r in records])
    scores = np.asarray([float(r["prob_class_1"]) for r in records])
    if len(records) != 212 or set(labels) != {0, 1} or labels.sum() != 140:
        raise ValueError("Test prediction invariants changed; review before publication")
    if not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
        raise ValueError("Invalid test probability")
    order = np.argsort(-scores, kind="stable")
    sorted_scores, sorted_labels = scores[order], labels[order]
    endpoint = np.r_[np.flatnonzero(np.diff(sorted_scores)), len(scores) - 1]
    tp = np.cumsum(sorted_labels)[endpoint]
    precision = tp / (endpoint + 1)
    recall = tp / labels.sum()
    ap = float(np.sum(np.diff(np.r_[0., recall]) * precision))
    saved_ap = float(read_json(metric_path)["pr_auc"])
    if not np.isclose(ap, saved_ap, rtol=0, atol=1e-12):
        raise ValueError(f"Computed AP {ap} != saved AP {saved_ap}")
    rplot, pplot = np.r_[0., recall], np.r_[1., precision]
    rows = [{"threshold": "", "recall": 0., "precision": 1., "true_positives": 0,
             "predicted_positives": 0, "point_role": "plotting_endpoint_no_positive_predictions"}]
    rows += [{"threshold": float(sorted_scores[e]), "recall": float(r), "precision": float(p),
              "true_positives": int(t), "predicted_positives": int(e + 1), "point_role": "observed_score_threshold"}
             for e, r, p, t in zip(endpoint, recall, precision, tp)]
    fig, ax = plt.subplots(figsize=(10, 7.1))
    fig.subplots_adjust(left=.135, right=.96, top=.96, bottom=.28)
    ax.step(rplot, pplot, where="pre", color=theme.c["indigo700"], linewidth=2.2,
            label=f"Adapted model (AP {ap:.3f})")
    prevalence = float(labels.mean())
    ax.axhline(prevalence, color=theme.c["teal600"], linestyle="--", linewidth=1.7,
               label=f"Positive prevalence ({prevalence:.3f})")
    ax.set(xlabel="Recall", ylabel="Precision", xlim=(0, 1.015), ylim=(0, 1.025))
    ax.set_xticks(np.linspace(0, 1, 6))
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.legend(loc="upper center", bbox_to_anchor=(.5, -.17), ncol=1)
    finish(fig, SLUGS[3], theme, destination, rows,
        "Stage I test-split precision–recall curve, recomputed from all 212 saved region predictions (140 positive, 72 negative). Predictions are ordered by prob_class_1, with equal scores grouped at a single threshold. The staircase uses the precision at each inclusive score threshold; the (recall 0, precision 1) endpoint is a plotting convention, not an additional observation. Average precision is the sum of precision weighted by each increase in recall, reproducing the saved value 0.722267 to numerical tolerance; it is not trapezoidal PR-AUC. The dashed reference marks positive-label prevalence, 140/212 = 0.660377. No smoothing, uncertainty estimates, or exclusions are applied. Two exact full-region sequences are shared with the training split, and transcript-level independence cannot be verified without identifiers; this figure therefore describes the supplied test split, not an established independent external test.",
        "An empirical precision–recall staircase for 212 saved Stage I test predictions, with AP 0.722. A dashed reference line shows the positive-label prevalence of 0.660. The supplied split has exact sequence overlap with training and is not established as independent.",
        {"sources": [source(path, "all rows: label and prob_class_1"), source(metric_path, "pr_auc and sample counts")],
         "chart_type": "empirical precision–recall staircase",
         "analysis_unit": "region prediction", "n": {"regions": 212, "positive": 140, "negative": 72},
         "transforms": ["stable decreasing-score sort", "aggregate tied score thresholds", "precision=TP/(TP+FP)", "recall=TP/140"],
         "ap_definition": "sum of precision * increase in recall at distinct descending score thresholds",
         "ap": ap, "saved_ap": saved_ap, "ap_absolute_tolerance": 1e-12,
         "missing": "None in the label and positive-score columns", "axis_ranges": [0, 1]},
        [{"metric": "average_precision", "value": ap}, {"metric": "positive_prevalence", "value": prevalence}])


def ranking_reported(theme, destination):
    path = ROOT / "provenance/source-documents/01_WIKI.md"
    text = path.read_text(encoding="utf-8")
    match = re.search(r"Test results\s+(.*?)\nOn the test set", text, re.S)
    if match is None:
        raise ValueError("Cannot locate archived draft ranking test table")
    section = match.group(1)
    values = {name.strip(): float(value) for name, value in re.findall(r"\|\s*([^|\n]+?)\s*\|\s*([0-9.]+)\s*\|", section)}
    expected = {"Hit@1", "Hit@5", "Hit@10", "Hit@20", "Recall@5", "Recall@10", "Recall@20"}
    if not expected.issubset(values) or "Recall@1" in values:
        raise ValueError("Ranking draft fields changed; review before publication")
    rows = [{"metric": metric, "k": k, "reported_score": values[f"{metric}@{k}"],
             "evidence_status": "draft_reported_not_recomputed", "evaluated_positive_regions": 140}
            for metric, ks in (("Hit", [1, 5, 10, 20]), ("Recall", [5, 10, 20])) for k in ks]
    fig, ax = plt.subplots(figsize=(10, 6.7))
    fig.subplots_adjust(left=.135, right=.96, top=.955, bottom=.24)
    for metric, color, marker in (("Hit", theme.c["indigo700"], "o"), ("Recall", theme.c["teal600"], "s")):
        selected = [r for r in rows if r["metric"] == metric]
        ax.scatter([r["k"] for r in selected], [r["reported_score"] for r in selected],
                   color=color, marker=marker, s=105, edgecolors=theme.c["ink"], linewidths=.7,
                   label=f"{metric}@K (reported)", zorder=3)
    ax.set(xlabel="Top-ranked cytidines (K)", ylabel="Reported score", xlim=(0, 21), ylim=(0, 1.015))
    ax.set_xticks([1, 5, 10, 20])
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.legend(loc="upper center", bbox_to_anchor=(.5, -.17), ncol=2, columnspacing=1.2)
    finish(fig, SLUGS[4], theme, destination, rows,
        "Draft-reported Stage II ranking scores on 140 positive-containing test regions. Points reproduce the archived draft table: Hit@1/5/10/20 = 0.243/0.607/0.779/0.921 and Recall@5/10/20 = 0.373/0.573/0.792. Hit@K is the fraction of evaluated regions with at least one positive site among the top K cytidines; Recall@K is the mean within-region fraction of positive sites recovered. K retains its actual numeric spacing. No lines, interpolated values, error intervals, or Recall@1 value are added. The preserved evaluation script skips the 72 test regions without a positive site, so these metrics are conditional on 140 positive-containing regions rather than all 212 regions. The draft calls these units transcripts; the supplied records support only region-level terminology. No per-site model scores or ranking-output file was supplied to independently recompute this table. These reported values are not evidence for the performance of an end-to-end system gated by Stage I predictions.",
        "Discrete points show draft-reported Stage II Hit@K and Recall@K for K equal to 1, 5, 10, and 20. Hit rises from 0.243 at K 1 to 0.921 at K 20; Recall is absent at K 1 and rises from 0.373 at K 5 to 0.792 at K 20. Values cover only 140 positive-containing regions and were not independently recomputed.",
        {"sources": [source(path, "Stage II / Test results table: Hit@K and Recall@K")],
         "chart_type": "unconnected discrete points at original K values",
         "evidence_status": "draft-reported summary; per-site ranking output unavailable",
         "analysis_unit": "region with at least one labelled positive site",
         "n": {"evaluated_positive_regions": 140, "nonpositive_regions_skipped_by_existing_method": 72},
         "transforms": ["parse supplied draft table; no changes to rounded values"],
         "missing": "Recall@1 absent in source; not inferred",
         "method_scope": "conditional positive-region ranking, not Stage I-gated end-to-end evaluation"})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=ROOT / "wiki/figures")
    args = parser.parse_args()
    output = args.output_root.resolve()
    for slug in SLUGS:
        if (output / slug).exists():
            raise FileExistsError(f"Use a fresh output directory: {output / slug}")
    theme = Theme(defaults=HERE / "orca-figure-style.json")
    with theme.context():
        pretraining(theme, output)
        stage1_trajectories(theme, output)
        test_pr(theme, output)
        ranking_reported(theme, output)


if __name__ == "__main__":
    main()
