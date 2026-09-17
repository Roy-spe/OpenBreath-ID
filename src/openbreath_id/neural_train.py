"""Train and evaluate subject-disjoint neural respiratory verification encoders."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import random
from typing import Iterable

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .data import Recording, discover_recordings, load_signal
from .metrics import (
    LogisticCalibrator,
    calibration_metrics,
    equal_error_rate,
    minimum_detection_cost,
    rates_at_threshold,
    threshold_at_far,
)
from .neural_data import (
    PKBatchSampler,
    RespiratoryWindowDataset,
    build_window_references,
    preprocess_window,
)
from .neural_models import (
    ArcMarginHead,
    BilateralAttentionEncoder,
    BilateralInteractionEncoder,
    MultiScaleTwoChannelEncoder,
    TwoChannelEncoder,
    V3BilateralInteractionEncoder,
    V3BIEAblationEncoder,
    V3SharedTwoTowerEncoder,
    V3StackedChannelEncoder,
    V3UnilateralEncoder,
    supervised_contrastive_loss,
)
from .protocol import IdentitySplit, make_outer_splits
from .uncertainty import subject_cluster_bootstrap


def set_reproducible_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    return device


def build_encoder(
    model_name: str,
    *,
    embedding_dimension: int,
    branch_dimension: int,
    base_channels: int,
) -> nn.Module:
    if model_name == "two_channel":
        return TwoChannelEncoder(embedding_dimension, base_channels)
    if model_name == "bilateral":
        return BilateralInteractionEncoder(
            embedding_dimension,
            branch_dimension=branch_dimension,
            base_channels=base_channels,
        )
    if model_name == "multiscale":
        return MultiScaleTwoChannelEncoder(embedding_dimension, base_channels)
    if model_name == "bilateral_attention":
        return BilateralAttentionEncoder(
            embedding_dimension,
            branch_dimension=branch_dimension,
            base_channels=base_channels,
        )
    if model_name == "v3_stacked":
        return V3StackedChannelEncoder(embedding_dimension, base_channels)
    if model_name == "v3_bie":
        return V3BilateralInteractionEncoder(
            embedding_dimension,
            branch_dimension=branch_dimension,
            base_channels=base_channels,
        )
    if model_name == "v3_shared_two_tower":
        return V3SharedTwoTowerEncoder(
            embedding_dimension,
            branch_dimension=branch_dimension,
            base_channels=base_channels,
        )
    ablation_prefix = "v3_bie_"
    if model_name.startswith(ablation_prefix) and model_name[len(ablation_prefix) :] in V3BIEAblationEncoder.VARIANTS:
        return V3BIEAblationEncoder(
            model_name[len(ablation_prefix) :],
            embedding_dimension,
            branch_dimension=branch_dimension,
            base_channels=base_channels,
        )
    if model_name == "v3_channel1":
        return V3UnilateralEncoder(
            embedding_dimension, base_channels, channel_index=0
        )
    if model_name == "v3_channel2":
        return V3UnilateralEncoder(
            embedding_dimension, base_channels, channel_index=1
        )
    raise ValueError(
        "model_name must be 'two_channel', 'bilateral', 'multiscale', or "
        "'bilateral_attention', 'v3_stacked', 'v3_bie', 'v3_shared_two_tower', "
        "a v3_bie ablation, 'v3_channel1', or 'v3_channel2'"
    )


def _autocast_context(device: torch.device, enabled: bool):
    if device.type == "cuda" and enabled:
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def train_one_epoch(
    encoder: nn.Module,
    head: ArcMarginHead | None,
    loader: DataLoader,
    sampler: PKBatchSampler,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    *,
    device: torch.device,
    epoch: int,
    use_amp: bool,
    loss_name: str,
    contrastive_temperature: float,
    supcon_weight: float,
) -> tuple[float, float]:
    encoder.train()
    if head is not None:
        head.train()
    sampler.set_epoch(epoch)
    losses: list[float] = []
    correct = 0
    examples = 0
    for windows, labels in loader:
        windows = windows.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with _autocast_context(device, use_amp):
            embeddings = encoder(windows)
            if loss_name == "arcface":
                if head is None:
                    raise ValueError("ArcFace training requires a classification head")
                logits = head(embeddings, labels)
                loss = F.cross_entropy(logits, labels)
                predictions = logits.detach().argmax(dim=1)
            elif loss_name == "supcon":
                loss = supervised_contrastive_loss(
                    embeddings, labels, temperature=contrastive_temperature
                )
                similarities = embeddings.detach() @ embeddings.detach().T
                similarities.fill_diagonal_(float("-inf"))
                predictions = labels[similarities.argmax(dim=1)]
            elif loss_name == "arcface_supcon":
                if head is None:
                    raise ValueError("combined training requires a classification head")
                logits = head(embeddings, labels)
                arcface_loss = F.cross_entropy(logits, labels)
                contrastive_loss = supervised_contrastive_loss(
                    embeddings, labels, temperature=contrastive_temperature
                )
                loss = arcface_loss + supcon_weight * contrastive_loss
                predictions = logits.detach().argmax(dim=1)
            else:
                raise ValueError(
                    "loss_name must be 'arcface', 'supcon', or 'arcface_supcon'"
                )
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(
            list(encoder.parameters())
            + (list(head.parameters()) if head is not None else []),
            max_norm=5.0,
        )
        scaler.step(optimizer)
        scaler.update()
        losses.append(float(loss.detach()))
        correct += int((predictions == labels).sum())
        examples += len(labels)
    return float(np.mean(losses)), float(correct / examples)


def _normalize_vector(values: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(values))
    return values / norm if norm > 1e-12 else np.zeros_like(values)


@torch.inference_mode()
def _embed_selected_windows(
    encoder: nn.Module,
    signal: np.ndarray,
    starts: list[int],
    *,
    device: torch.device,
    batch_size: int,
    normalization: str,
    canonicalize_polarity: bool,
    channel_view: str = "bilateral",
) -> np.ndarray:
    encoder.eval()
    outputs: list[np.ndarray] = []
    for offset in range(0, len(starts), batch_size):
        batch_starts = starts[offset : offset + batch_size]
        windows = np.stack(
            [
                preprocess_window(
                    signal[start : start + 180],
                    normalization=normalization,
                    canonicalize_polarity=canonicalize_polarity,
                    channel_view=channel_view,
                )
                for start in batch_starts
            ]
        )
        tensor = torch.from_numpy(windows).to(device, non_blocking=True)
        outputs.append(encoder(tensor).float().cpu().numpy())
    return np.concatenate(outputs, axis=0)


def _evenly_select_indices(length: int, maximum: int) -> np.ndarray:
    if maximum <= 0 or length <= maximum:
        return np.arange(length, dtype=int)
    return np.linspace(0, length - 1, maximum, dtype=int)


@torch.inference_mode()
def make_neural_trials(
    encoder: nn.Module,
    recordings: list[Recording],
    subjects: Iterable[str],
    *,
    device: torch.device,
    probe_seconds: int,
    guard_minutes: int,
    max_probes_per_subject: int,
    inference_batch_size: int,
    normalization: str,
    canonicalize_polarity: bool,
    channel_view: str = "bilateral",
) -> tuple[
    np.ndarray,
    np.ndarray,
    tuple[str, ...],
    dict[str, int],
    tuple[str, ...],
]:
    if probe_seconds <= 0 or probe_seconds % 30:
        raise ValueError("probe_seconds must be a positive multiple of 30")
    by_subject = {
        recording.subject_id: recording
        for recording in recordings
        if recording.session_id == "primary"
    }
    enrollment_windows = 10
    guard_windows = guard_minutes * 2
    probe_windows = probe_seconds // 30
    enrollment_subjects: list[str] = []
    enrollments: list[np.ndarray] = []
    probes: list[np.ndarray] = []
    probe_subjects: list[str] = []
    counts: dict[str, int] = {}

    for subject in sorted(set(subjects)):
        recording = by_subject.get(subject)
        if recording is None:
            continue
        signal = load_signal(recording.path)
        total_windows = len(signal) // 180
        probe_start = enrollment_windows + guard_windows
        groups = (total_windows - probe_start) // probe_windows
        if groups < 1:
            continue
        selected_groups = _evenly_select_indices(groups, max_probes_per_subject)
        enrollment_starts = [index * 180 for index in range(enrollment_windows)]
        probe_starts = [
            (probe_start + int(group) * probe_windows + within) * 180
            for group in selected_groups
            for within in range(probe_windows)
        ]
        embeddings = _embed_selected_windows(
            encoder,
            signal,
            enrollment_starts + probe_starts,
            device=device,
            batch_size=inference_batch_size,
            normalization=normalization,
            canonicalize_polarity=canonicalize_polarity,
            channel_view=channel_view,
        )
        enrollment = _normalize_vector(embeddings[:enrollment_windows].mean(axis=0))
        probe_values = embeddings[enrollment_windows:].reshape(
            len(selected_groups), probe_windows, -1
        ).mean(axis=1)
        probe_values = np.stack([_normalize_vector(value) for value in probe_values])
        enrollment_subjects.append(subject)
        enrollments.append(enrollment)
        probes.append(probe_values)
        probe_subjects.extend([subject] * len(probe_values))
        counts[subject] = len(probe_values)

    if len(enrollments) < 2:
        raise ValueError("at least two identities with enrollment and probe windows are required")
    enrollment_matrix = np.stack(enrollments)
    probe_matrix = np.concatenate(probes)
    score_matrix = probe_matrix @ enrollment_matrix.T
    label_matrix = np.asarray(probe_subjects)[:, None] == np.asarray(enrollment_subjects)[None, :]
    return (
        score_matrix.reshape(-1),
        label_matrix.astype(np.int8).reshape(-1),
        tuple(enrollment_subjects),
        counts,
        tuple(
            subject
            for subject in probe_subjects
            for _ in range(len(enrollment_subjects))
        ),
    )


def evaluate_unseen_identities(
    encoder: nn.Module,
    recordings: list[Recording],
    split: IdentitySplit,
    *,
    device: torch.device,
    probe_seconds: int,
    guard_minutes: int,
    max_probes_per_subject: int,
    inference_batch_size: int,
    normalization: str,
    canonicalize_polarity: bool,
    channel_view: str = "bilateral",
    bootstrap_replicates: int = 2000,
    bootstrap_confidence: float = 0.95,
    bootstrap_seed: int = 2027,
) -> dict[str, object]:
    common = {
        "device": device,
        "probe_seconds": probe_seconds,
        "guard_minutes": guard_minutes,
        "max_probes_per_subject": max_probes_per_subject,
        "inference_batch_size": inference_batch_size,
        "normalization": normalization,
        "canonicalize_polarity": canonicalize_polarity,
        "channel_view": channel_view,
    }
    validation_scores, validation_labels, validation_subjects, _, _ = make_neural_trials(
        encoder, recordings, split.validation, **common
    )
    (
        test_scores,
        test_labels,
        test_subjects,
        probe_counts,
        test_trial_subjects,
    ) = make_neural_trials(
        encoder, recordings, split.test, **common
    )
    threshold = threshold_at_far(validation_scores, validation_labels, target_far=0.01)
    test_tar, test_far = rates_at_threshold(test_scores, test_labels, threshold)
    calibrator = LogisticCalibrator.fit(validation_scores, validation_labels)
    probabilities = calibrator.predict_proba(test_scores)
    uncertainty = subject_cluster_bootstrap(
        test_scores,
        test_labels,
        test_trial_subjects,
        probabilities=probabilities,
        operating_threshold=threshold,
        replicates=bootstrap_replicates,
        confidence_level=bootstrap_confidence,
        seed=bootstrap_seed,
    )
    return {
        "probe_seconds": probe_seconds,
        "guard_minutes": guard_minutes,
        "validation_identities": len(validation_subjects),
        "test_identities": len(test_subjects),
        "validation_eer": equal_error_rate(validation_scores, validation_labels),
        "test_eer": equal_error_rate(test_scores, test_labels),
        "test_normalized_min_dcf_p01": minimum_detection_cost(test_scores, test_labels),
        "validation_selected_threshold": threshold,
        "test_tar_at_validation_1pct_far_threshold": test_tar,
        "test_far_at_validation_1pct_far_threshold": test_far,
        "test_trials": len(test_scores),
        "test_genuine_trials": int(test_labels.sum()),
        "test_impostor_trials": int((test_labels == 0).sum()),
        "test_probes_per_subject": probe_counts,
        "calibrator": {
            "slope": calibrator.slope,
            "intercept": calibrator.intercept,
        },
        "subject_cluster_bootstrap": uncertainty,
        **calibration_metrics(probabilities, test_labels),
    }


def validation_eer_for_selection(
    encoder: nn.Module,
    recordings: list[Recording],
    split: IdentitySplit,
    *,
    device: torch.device,
    probe_seconds: int,
    guard_minutes: int,
    max_probes_per_subject: int,
    inference_batch_size: int,
    normalization: str,
    canonicalize_polarity: bool,
    channel_view: str = "bilateral",
) -> float:
    """Compute model-selection EER without loading or scoring test identities."""

    scores, labels, _, _, _ = make_neural_trials(
        encoder,
        recordings,
        split.validation,
        device=device,
        probe_seconds=probe_seconds,
        guard_minutes=guard_minutes,
        max_probes_per_subject=max_probes_per_subject,
        inference_batch_size=inference_batch_size,
        normalization=normalization,
        canonicalize_polarity=canonicalize_polarity,
        channel_view=channel_view,
    )
    return equal_error_rate(scores, labels)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("reports/neural_fold0.json"))
    parser.add_argument("--fold", type=int, default=0, choices=range(5))
    parser.add_argument(
        "--model",
        choices=(
            "two_channel",
            "bilateral",
            "multiscale",
            "bilateral_attention",
            "v3_stacked",
            "v3_bie",
            "v3_shared_two_tower",
            "v3_bie_no_asymmetry",
            "v3_bie_no_interaction",
            "v3_bie_no_global",
            "v3_bie_independent_nostrils",
            "v3_channel1",
            "v3_channel2",
        ),
        default="two_channel",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batches-per-epoch", type=int, default=200)
    parser.add_argument("--subjects-per-batch", type=int, default=16)
    parser.add_argument("--windows-per-subject", type=int, default=4)
    parser.add_argument("--maximum-training-windows-per-recording", type=int, default=0)
    parser.add_argument("--embedding-dimension", type=int, default=256)
    parser.add_argument("--branch-dimension", type=int, default=64)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--arcface-scale", type=float, default=30.0)
    parser.add_argument("--arcface-margin", type=float, default=0.20)
    parser.add_argument(
        "--loss",
        choices=("arcface", "supcon", "arcface_supcon"),
        default="arcface",
    )
    parser.add_argument("--contrastive-temperature", type=float, default=0.07)
    parser.add_argument("--supcon-weight", type=float, default=0.5)
    parser.add_argument(
        "--normalization",
        choices=("none", "zscore", "robust", "shared_robust"),
        default="robust",
    )
    parser.add_argument(
        "--augmentation",
        choices=(
            "none",
            "mild",
            "channel_dropout",
            "channel_swap",
            "robust_mixed",
        ),
        default="none",
        help="Training-window augmentation; validation and test windows are never augmented.",
    )
    parser.add_argument("--channel-polarity", choices=("raw", "correlation"), default="correlation")
    parser.add_argument(
        "--channel-view",
        choices=("bilateral", "channel_1", "channel_2"),
        default="bilateral",
        help="Restrict preprocessing to one channel for unilateral experiments.",
    )
    parser.add_argument("--probe-seconds", type=int, nargs="+", default=(60, 30))
    parser.add_argument(
        "--selection-probe-seconds",
        type=int,
        nargs="+",
        default=(60,),
        help="Validation probe durations used for checkpoint selection.",
    )
    parser.add_argument(
        "--selection-probe-weights",
        type=float,
        nargs="+",
        default=(1.0,),
        help="Weights paired with --selection-probe-seconds.",
    )
    parser.add_argument("--guard-minutes", type=int, default=30)
    parser.add_argument("--max-probes-per-subject", type=int, default=120)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--bootstrap-confidence", type=float, default=0.95)
    parser.add_argument("--inference-batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--cache-recordings", type=int, default=64)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument(
        "--validation-only",
        action="store_true",
        help="Train and select a checkpoint without loading or scoring outer-test identities.",
    )
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument(
        "--split-seed",
        type=int,
        default=2027,
        help="Fixed identity-partition seed, independent of training randomness.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.epochs < 1 or args.batches_per_epoch < 1:
        raise SystemExit("epochs and batches per epoch must be positive")
    if args.guard_minutes < 0:
        raise SystemExit("guard minutes cannot be negative")
    if len(args.selection_probe_seconds) != len(args.selection_probe_weights):
        raise SystemExit("selection probe durations and weights must have equal length")
    if any(seconds <= 0 or seconds % 30 for seconds in args.selection_probe_seconds):
        raise SystemExit("selection probe durations must be positive multiples of 30 seconds")
    if any(weight <= 0 for weight in args.selection_probe_weights):
        raise SystemExit("selection probe weights must be positive")
    if args.bootstrap_replicates < 1:
        raise SystemExit("bootstrap replicates must be positive")
    if not 0.0 < args.bootstrap_confidence < 1.0:
        raise SystemExit("bootstrap confidence must be between zero and one")
    required_view = {
        "v3_channel1": "channel_1",
        "v3_channel2": "channel_2",
    }.get(args.model)
    if required_view is not None and args.channel_view != required_view:
        raise SystemExit(f"{args.model} requires --channel-view {required_view}")
    set_reproducible_seed(args.seed)
    device = resolve_device(args.device)
    recordings = [
        recording
        for recording in discover_recordings(args.dataset_root, states=("wake",))
        if recording.session_id == "primary"
    ]
    subjects = sorted({recording.subject_id for recording in recordings})
    split = make_outer_splits(subjects, seed=args.split_seed)[args.fold]
    label_by_subject = {subject: index for index, subject in enumerate(split.train)}
    references = build_window_references(
        recordings,
        split.train,
        maximum_windows_per_recording=args.maximum_training_windows_per_recording,
    )
    dataset = RespiratoryWindowDataset(
        references,
        label_by_subject,
        normalization=args.normalization,
        canonicalize_polarity=args.channel_polarity == "correlation",
        augmentation=args.augmentation,
        channel_view=args.channel_view,
        cache_recordings=args.cache_recordings,
    )
    sampler = PKBatchSampler(
        references,
        subjects_per_batch=args.subjects_per_batch,
        windows_per_subject=args.windows_per_subject,
        batches_per_epoch=args.batches_per_epoch,
        seed=args.seed,
    )
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    encoder = build_encoder(
        args.model,
        embedding_dimension=args.embedding_dimension,
        branch_dimension=args.branch_dimension,
        base_channels=args.base_channels,
    ).to(device)
    head = (
        ArcMarginHead(
            args.embedding_dimension,
            len(split.train),
            scale=args.arcface_scale,
            margin=args.arcface_margin,
        ).to(device)
        if args.loss in {"arcface", "arcface_supcon"}
        else None
    )
    parameters = list(encoder.parameters()) + (
        list(head.parameters()) if head is not None else []
    )
    encoder_parameters = sum(parameter.numel() for parameter in encoder.parameters())
    trainable_parameters = sum(parameter.numel() for parameter in parameters)
    optimizer = torch.optim.AdamW(
        parameters,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    use_amp = device.type == "cuda" and not args.no_amp
    # BF16 has FP32-like exponent range, so gradient scaling is unnecessary.
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    checkpoint_path = args.output.with_suffix(".pt")
    history: list[dict[str, object]] = []
    best_validation_score = float("inf")
    best_validation_eers: dict[int, float] = {}
    best_epoch = 0

    print(
        f"device={device} model={args.model} train_identities={len(split.train)} "
        f"windows={len(references)} encoder_parameters={encoder_parameters:,}"
    )
    for epoch in range(args.epochs):
        loss, training_accuracy = train_one_epoch(
            encoder,
            head,
            loader,
            sampler,
            optimizer,
            scaler,
            device=device,
            epoch=epoch,
            use_amp=use_amp,
            loss_name=args.loss,
            contrastive_temperature=args.contrastive_temperature,
            supcon_weight=args.supcon_weight,
        )
        validation_eers = {
            seconds: validation_eer_for_selection(
                encoder,
                recordings,
                split,
                device=device,
                probe_seconds=seconds,
                guard_minutes=args.guard_minutes,
                max_probes_per_subject=min(args.max_probes_per_subject, 30),
                inference_batch_size=args.inference_batch_size,
                normalization=args.normalization,
                canonicalize_polarity=args.channel_polarity == "correlation",
                channel_view=args.channel_view,
            )
            for seconds in args.selection_probe_seconds
        }
        validation_score = float(
            sum(
                weight * validation_eers[seconds]
                for seconds, weight in zip(
                    args.selection_probe_seconds,
                    args.selection_probe_weights,
                    strict=True,
                )
            )
        )
        history.append(
            {
                "epoch": epoch + 1,
                "training_loss": loss,
                "training_batch_retrieval_accuracy": training_accuracy,
                "validation_eer_60s": validation_eers.get(60),
                "validation_eer_by_probe_seconds": {
                    str(seconds): value for seconds, value in validation_eers.items()
                },
                "validation_selection_score": validation_score,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            }
        )
        validation_description = " ".join(
            f"validation_eer_{seconds}s={100 * value:.2f}%"
            for seconds, value in validation_eers.items()
        )
        print(
            f"epoch={epoch + 1} loss={loss:.4f} "
            f"train_acc={100 * training_accuracy:.2f}% "
            f"{validation_description} selection_score={validation_score:.4f}"
        )
        if validation_score < best_validation_score:
            best_validation_score = validation_score
            best_validation_eers = validation_eers.copy()
            best_epoch = epoch + 1
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "encoder": encoder.state_dict(),
                    "head": head.state_dict() if head is not None else None,
                    "args": vars(args),
                    "fold": args.fold,
                    "epoch": best_epoch,
                    "validation_eer_60s": best_validation_eers.get(60),
                    "validation_eer_by_probe_seconds": {
                        str(seconds): value
                        for seconds, value in best_validation_eers.items()
                    },
                    "validation_selection_score": best_validation_score,
                },
                checkpoint_path,
            )
        scheduler.step()

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    encoder.load_state_dict(checkpoint["encoder"])
    evaluations = (
        []
        if args.validation_only
        else [
            evaluate_unseen_identities(
                encoder,
                recordings,
                split,
                device=device,
                probe_seconds=seconds,
                guard_minutes=args.guard_minutes,
                max_probes_per_subject=args.max_probes_per_subject,
                inference_batch_size=args.inference_batch_size,
                normalization=args.normalization,
                canonicalize_polarity=args.channel_polarity == "correlation",
                channel_view=args.channel_view,
                bootstrap_replicates=args.bootstrap_replicates,
                bootstrap_confidence=args.bootstrap_confidence,
                bootstrap_seed=args.seed + 1000 * args.fold + seconds,
            )
            for seconds in args.probe_seconds
        ]
    )
    payload = {
        "protocol": "subject-disjoint primary-session Wake-to-Wake verification",
        "fold": args.fold,
        "split_seed": args.split_seed,
        "training_seed": args.seed,
        "model": args.model,
        "embedding_dimension": args.embedding_dimension,
        "encoder_parameters": encoder_parameters,
        "trainable_parameters": trainable_parameters,
        "training_loss": (
            "ArcFace cross-entropy over training identities only"
            if args.loss == "arcface"
            else (
                "supervised contrastive loss over P x K training-identity batches"
                if args.loss == "supcon"
                else "ArcFace cross-entropy plus weighted supervised contrastive loss"
            )
        ),
        "loss_name": args.loss,
        "supcon_weight": args.supcon_weight if args.loss == "arcface_supcon" else None,
        "train_identities": len(split.train),
        "validation_identities": len(split.validation),
        "test_identities": len(split.test),
        "training_window_seconds": 30,
        "training_windows_indexed": len(references),
        "normalization": args.normalization,
        "augmentation": args.augmentation,
        "channel_polarity": args.channel_polarity,
        "channel_view": args.channel_view,
        "device": str(device),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "best_epoch": best_epoch,
        "selection_probe_seconds": args.selection_probe_seconds,
        "selection_probe_weights": args.selection_probe_weights,
        "best_validation_selection_score": best_validation_score,
        "best_validation_eer_60s": best_validation_eers.get(60),
        "best_validation_eer_by_probe_seconds": {
            str(seconds): value for seconds, value in best_validation_eers.items()
        },
        "checkpoint": str(checkpoint_path.resolve()),
        "evaluation_status": (
            "validation-only development run; outer-test identities not scored"
            if args.validation_only
            else "best validation-selected checkpoint scored once on outer-test identities"
        ),
        "history": history,
        "evaluations": evaluations,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(evaluations, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
