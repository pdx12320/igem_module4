#!/usr/bin/env python3
"""Audit the published bundle using only Python's standard library.

This checks saved data and predictions; it does not run or validate a model.
Outputs are deterministic and contain repository-relative paths only.
"""

import argparse
import ast
import collections
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import sys


SPLITS = ("train", "valid", "test")
EXPECTED = {
    "train": (3835, 2643, 1192),
    "valid": (213, 150, 63),
    "test": (212, 140, 72),
}
RESULT_BASE = Path("results/model_1024_classifier/1024_orca_puf12")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path, delimiter=","):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter=delimiter))


def read_json(path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def average_precision(labels, scores):
    """Non-interpolated AP, grouping tied scores at the same threshold."""
    positive_count = sum(labels)
    if not positive_count:
        return 0.0
    ordered = sorted(zip(scores, labels), reverse=True)
    true_positives = seen = 0
    result = 0.0
    for _, group in itertools.groupby(ordered, key=lambda pair: pair[0]):
        members = list(group)
        new_positives = sum(label for _, label in members)
        true_positives += new_positives
        seen += len(members)
        result += (new_positives / positive_count) * (true_positives / seen)
    return result


def roc_auc(labels, scores):
    """Pairwise ROC AUC with half credit for tied positive-negative pairs."""
    positive_count = sum(labels)
    negative_count = len(labels) - positive_count
    if not positive_count or not negative_count:
        return None
    negative_below = 0
    credit = 0.0
    for _, group in itertools.groupby(sorted(zip(scores, labels)), key=lambda x: x[0]):
        members = list(group)
        positives = sum(label for _, label in members)
        negatives = len(members) - positives
        credit += positives * (negative_below + 0.5 * negatives)
        negative_below += negatives
    return credit / (positive_count * negative_count)


def candidate_contexts(records):
    contexts = collections.Counter()
    for record in records:
        sequence = record["seq"].upper().replace("U", "T")
        padded = "N" * 50 + sequence + "N" * 50
        for position, base in enumerate(sequence):
            if base == "C":
                contexts[padded[position:position + 101]] += 1
    return contexts


def audit(root):
    report = {"scope": "Saved-artifact integrity and descriptive audit; no model execution", "errors": [], "warnings": []}

    def check(condition, message):
        if not condition:
            report["errors"].append(message)

    manifest = read_json(root / "provenance/import-manifest.json")
    entries = manifest["entries"]
    published_paths = []
    byte_identical = transformed = 0
    for entry in entries:
        relative = Path(entry["published_path"])
        check(not relative.is_absolute() and ".." not in relative.parts, "Unsafe manifest path")
        if relative.is_absolute() or ".." in relative.parts:
            continue
        published_paths.append(relative.as_posix())
        path = root / relative
        if entry.get("storage") == "git_history":
            check(relative.as_posix().startswith("provenance/source-documents/"),
                  "Only removed prose drafts may be recorded as historical")
            check(not path.exists(), f"Obsolete draft was restored: {relative}")
            check(entry.get("repository_revision") == "17a00c641f0a7ce726d65e16c2cca4b58a841431",
                  "Unexpected historical prose revision")
            continue
        check(path.is_file(), f"Missing manifest file: {relative}")
        if not path.is_file():
            continue
        digest = sha256(path)
        check(digest == entry["published_sha256"], f"Published checksum mismatch: {relative}")
        if entry["transformation"] == "byte-identical copy":
            byte_identical += 1
            check(digest == entry["source_sha256"], f"Source checksum mismatch: {relative}")
        else:
            transformed += 1
    check(len(published_paths) == len(set(published_paths)), "Duplicate published path in import manifest")
    report["import_manifest"] = {"entries": len(entries), "historical_prose_entries": sum(e.get("storage") == "git_history" for e in entries), "byte_identical_entries": byte_identical,
                                 "transformed_entries": transformed,
                                 "source_archive_available_required": False,
                                 "note": "Checks published bytes against recorded digests; transformed originals require the source archive for independent verification."}

    inventory = read_json(root / "provenance/archive-inventory.json")
    inventory_entries = inventory["entries"]
    inventory_published = {entry["published_path"] for entry in inventory_entries if entry.get("published_path")}
    check(inventory_published == set(published_paths), "Archive inventory and publication manifest disagree")
    members = [entry["source_member"] for entry in inventory_entries]
    check(len(members) == len(set(members)), "Duplicate source member in archive inventory")
    report["archive_inventory"] = {
        "members": len(inventory_entries), "published_members": len(inventory_published),
        "unpublished_members_retained_in_original_archive": sum(not entry.get("published_path") for entry in inventory_entries),
        "model_weight_members_retained_in_original_archive": sum(name.endswith(".safetensors") for name in members),
        "note": "Inventory describes archive contents; large unpublished members are not read or checked by this lightweight validator.",
    }

    json_count = 0
    for path in sorted(root.rglob("*.json")):
        if ".git" in path.parts:
            continue
        try:
            read_json(path)
            json_count += 1
        except (ValueError, OSError) as error:
            check(False, f"Invalid JSON: {path.relative_to(root)}: {error}")
    python_count = 0
    for directory in (root / "code", root / "scripts"):
        for path in sorted(directory.rglob("*.py")):
            try:
                ast.parse(path.read_text(encoding="utf-8"), filename=path.name)
                python_count += 1
            except (SyntaxError, UnicodeError) as error:
                check(False, f"Invalid Python syntax: {path.relative_to(root)}: {error}")
    report["syntax"] = {"json_files_parsed": json_count, "python_files_parsed_without_execution": python_count}

    regions = {}
    contexts = {}
    sequence_label_occurrences = collections.defaultdict(list)
    report["datasets"] = {}
    for split in SPLITS:
        rows = read_csv(root / f"data/1024nt_dataset/orca_puf12_{split}.csv")
        records = read_json(root / f"data/101nt_dataset/orca_puf12_{split}.json")
        check(bool(rows) and set(rows[0]) == {"seq", "label"}, f"CSV schema changed: {split}")
        check(len(rows) == len(records), f"CSV/JSON row count mismatch: {split}")
        labels = [int(row["label"]) for row in rows]
        for csv_line, (row, label) in enumerate(zip(rows, labels), start=2):
            sequence_label_occurrences[row["seq"]].append({"split": split, "csv_line": csv_line, "label": label})
        check(all(label in (0, 1) for label in labels), f"Nonbinary region label: {split}")
        counts = (len(rows), sum(labels), len(rows) - sum(labels))
        check(counts == EXPECTED[split], f"Frozen split counts changed: {split}: {counts}")
        label_length_mismatches = non_c_positives = row_mismatches = positive_sites = 0
        for row, record in zip(rows, records):
            check(set(record) == {"seq", "label"}, f"JSON record schema changed: {split}")
            sequence = record["seq"]
            per_base = record["label"]
            check(set(sequence) <= set("ACGTN"), f"Unexpected sequence alphabet: {split}")
            check(all(isinstance(value, (int, float)) and math.isfinite(value) and value >= 0 for value in per_base),
                  f"Invalid per-base value: {split}")
            label_length_mismatches += len(sequence) != len(per_base)
            non_c_positives += sum(value > 0 and base != "C" for base, value in zip(sequence, per_base))
            positive_sites += sum(value > 0 for value in per_base)
            row_mismatches += row["seq"] != sequence or int(row["label"]) != int(any(value > 0 for value in per_base))
        check(label_length_mismatches == 0, f"Per-base label length mismatch: {split}")
        check(non_c_positives == 0, f"Positive label at non-C base: {split}")
        check(row_mismatches == 0, f"CSV/JSON sequence or label disagreement: {split}")
        lengths = [len(row["seq"]) for row in rows]
        check(all(0 < length <= 1024 for length in lengths), f"Out-of-range region length: {split}")
        regions[split] = collections.Counter(row["seq"] for row in rows)
        contexts[split] = candidate_contexts(records)
        report["datasets"][split] = {
            "region_rows": len(rows), "positive_regions": sum(labels), "zero_positive_regions": len(rows) - sum(labels),
            "minimum_length_nt": min(lengths), "maximum_length_nt": max(lengths),
            "full_1024nt_regions": lengths.count(1024), "unique_region_sequences": len(regions[split]),
            "cytidine_candidate_occurrences": sum(contexts[split].values()), "positive_site_occurrences": positive_sites,
            "csv_json_row_disagreements": row_mismatches, "label_length_mismatches": label_length_mismatches,
            "positive_labels_at_non_C": non_c_positives,
        }
    conflicting_sequences = [
        {"sequence_sha256": hashlib.sha256(sequence.encode("ascii")).hexdigest(),
         "length_nt": len(sequence), "occurrences": occurrences}
        for sequence, occurrences in sequence_label_occurrences.items()
        if len({entry["label"] for entry in occurrences}) > 1
    ]
    report["identical_sequence_label_conflicts"] = {
        "unique_sequences": len(conflicting_sequences),
        "details": sorted(conflicting_sequences, key=lambda entry: entry["sequence_sha256"]),
        "note": "Same sequence has both binary labels. No labels changed; locus/isoform provenance is unavailable.",
    }
    report["cross_split_overlap"] = {}
    for left, right in itertools.combinations(SPLITS, 2):
        shared_regions = set(regions[left]) & set(regions[right])
        shared_contexts = set(contexts[left]) & set(contexts[right])
        report["cross_split_overlap"][f"{left}-{right}"] = {
            "shared_unique_region_sequences": len(shared_regions),
            "region_occurrences_left": sum(regions[left][seq] for seq in shared_regions),
            "region_occurrences_right": sum(regions[right][seq] for seq in shared_regions),
            "shared_unique_101nt_C_centered_contexts": len(shared_contexts),
            "context_occurrences_left": sum(contexts[left][seq] for seq in shared_contexts),
            "context_occurrences_right": sum(contexts[right][seq] for seq in shared_contexts),
        }

    positives = read_csv(root / "data/PUF12_positive_with_strand.tsv", delimiter="\t")
    check(len(positives) == 12908, "Frozen positive TSV row count changed")
    coordinates = {(row["chromosome"], row["position"], row["strand"]) for row in positives}
    check(len(coordinates) == len(positives), "Duplicate positive TSV genomic coordinate/strand")
    check(all(len(row["sequence_101nt"]) == 101 and row["sequence_101nt"][50] == "C" for row in positives),
          "Positive TSV context length or central C mismatch")
    alleles = collections.Counter(f'{row["strand"]}:{row["ref"]}>{row["alt"]}' for row in positives)
    check(dict(alleles) == {"-:G>A": 6588, "+:C>T": 6320}, "Positive TSV strand/allele counts changed")
    report["positive_tsv"] = {"rows": len(positives), "unique_coordinate_strand_tuples": len(coordinates),
                               "strand_allele_counts": dict(sorted(alleles.items())),
                               "all_contexts_101nt_centered_on_C": True,
                               "note": "No raw supporting-read/depth columns or mapping to split rows; draft selection thresholds cannot be reconstructed."}

    prediction_path = root / RESULT_BASE / "test_evaluation/test_predictions.csv"
    predictions = read_csv(prediction_path)
    original_test = read_csv(root / "data/1024nt_dataset/orca_puf12_test.csv")
    check(len(predictions) == len(original_test), "Prediction/test row count mismatch")
    check(all(a["seq"] == b["seq"] and a["label"] == b["label"] for a, b in zip(predictions, original_test)),
          "Prediction/test sequence or label alignment mismatch")
    labels = [int(row["label"]) for row in predictions]
    classes = [int(row["predicted_class"]) for row in predictions]
    scores = [float(row["prob_class_1"]) for row in predictions]
    check(all(math.isfinite(score) and 0 <= score <= 1 for score in scores), "Invalid prediction probability")
    check(all(math.isclose(float(row["prob_class_0"]) + score, 1.0, abs_tol=1e-9)
              for row, score in zip(predictions, scores)), "Prediction probabilities do not sum to one")
    check(all(predicted == int(float(row["prob_class_1"]) > float(row["prob_class_0"]))
              for row, predicted in zip(predictions, classes)), "Saved class disagrees with probability argmax")
    matrix = collections.Counter(zip(labels, classes))
    tn, fp, fn, tp = (matrix[(0, 0)], matrix[(0, 1)], matrix[(1, 0)], matrix[(1, 1)])
    metrics = {
        "accuracy": (tp + tn) / len(labels), "precision": tp / (tp + fp), "recall": tp / (tp + fn),
        "f1": 2 * tp / (2 * tp + fp + fn), "roc_auc": roc_auc(labels, scores),
        "pr_auc": average_precision(labels, scores), "true_negative": tn, "false_positive": fp,
        "false_negative": fn, "true_positive": tp, "num_samples": len(labels),
        "num_positive": sum(labels), "num_negative": len(labels) - sum(labels),
    }
    saved_metrics = read_json(root / RESULT_BASE / "test_evaluation/test_metrics.json")
    for key, value in metrics.items():
        check(math.isclose(value, saved_metrics[key], rel_tol=1e-10, abs_tol=1e-10), f"Saved metric mismatch: {key}")
    report["stage_i_test_recomputed_from_saved_predictions"] = metrics
    report["stage_i_test_positive_prevalence"] = sum(labels) / len(labels)
    report["metric_definition"] = "Historical pr_auc is non-interpolated average precision (AP), not trapezoidal PR area."
    report["stage_i_best_search"] = read_json(root / RESULT_BASE / "best_trial.json")
    report["stage_i_final_validation"] = read_json(root / RESULT_BASE / "final_model/validation_metrics.json")
    report["stage_i_final_hyperparameters"] = read_json(root / RESULT_BASE / "final_model/best_hyperparameters.json")

    comparison = {}
    for path in sorted((root / "results/model_1024_classifier").glob("*/final_model/checkpoint-*/trainer_state.json")):
        state = read_json(path)
        evaluations = [entry for entry in state["log_history"] if "eval_pr_auc" in entry]
        comparison[path.parts[-4]] = {
            "source": path.relative_to(root).as_posix(), "evaluation_points_in_saved_checkpoint": len(evaluations),
            "best_logged_validation_AP": max(entry["eval_pr_auc"] for entry in evaluations),
            "saved_checkpoint_step": state["global_step"],
        }
    report["stage_i_checkpoint_comparisons"] = comparison
    pretrain_path = root / "results/model_continual_pretrain/transcriptome_pretrain/checkpoint-100000/trainer_state.json"
    history = [entry for entry in read_json(pretrain_path)["log_history"] if "eval_masked_accuracy" in entry]
    report["continued_pretraining"] = {"evaluation_points": len(history), "first_saved_evaluation": history[0],
                                       "last_saved_evaluation": history[-1], "original_model_0_505_baseline": "draft-only"}

    report["warnings"] = [
        "Exact region and C-centered context sequences occur across splits. Four unique region sequences have conflicting binary labels across their occurrences. Transcript/gene IDs and the split builder are absent; transcript-disjoint generalization cannot be verified.",
        "Stage II evaluates only regions with a labelled positive (150 validation, 140 test), excluding zero-positive regions. Draft ranking metrics are not an end-to-end gated pipeline evaluation.",
        "Per-candidate Stage II scores and machine-readable ranking evaluation results are absent. Draft ranking values cannot be independently recomputed from this bundle.",
        "Historical Stage I tokenization uses max_length=103 while training/evaluation declare 1026. The original tokenized cache is absent, so effective historical training context is unresolved.",
        "This publication repository omits large model weights and pretraining inputs that are retained in the original archive (see archive-inventory.json). The local LAMAR implementation, split-construction code, tokenized cache and pinned training environment are missing from the supplied archive. Historical training/evaluation is not runnable from this repository alone.",
        "Stage II training_state.json was invalid JSON (trailing comma); the original is preserved as training_state.json.txt rather than silently repaired.",
        "The drafts' original-model masked accuracy 0.505, transcriptome size, positive-label thresholds and Stage II AP around 0.2 are not independently recoverable from saved evaluation artifacts.",
        "Saved Stage I search and final-run settings differ from draft hyperparameters. Stage II source uses ranking loss, a scalar center-token head and fixed settings, not the draft binary-classifier/Optuna description.",
        "These checks establish artifact consistency, not model retraining, biological validation, probability calibration or off-target safety.",
    ]
    report["status"] = "pass_with_documented_limitations" if not report["errors"] else "failed"
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, help="Write a new JSON report; refuses to overwrite an existing path")
    args = parser.parse_args()
    try:
        report = audit(args.root.resolve())
    except (OSError, ValueError, KeyError, TypeError, IndexError, ZeroDivisionError) as error:
        report = {"status": "failed", "errors": [str(error)], "warnings": []}
    payload = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output:
        try:
            with args.output.open("x", encoding="utf-8") as handle:
                handle.write(payload)
        except OSError as error:
            print(f"Cannot create output: {error}", file=sys.stderr)
            return 2
    else:
        print(payload, end="")
    return 1 if report["errors"] else 0


if __name__ == "__main__":
    sys.exit(main())
