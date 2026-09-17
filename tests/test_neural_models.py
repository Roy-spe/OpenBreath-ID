import torch

from openbreath_id.neural_models import (
    ArcMarginHead,
    BilateralAttentionEncoder,
    BilateralInteractionEncoder,
    MultiScaleTwoChannelEncoder,
    TwoChannelEncoder,
    V3BilateralInteractionEncoder,
    V3BIEAblationEncoder,
    V3StackedChannelEncoder,
    V3UnilateralEncoder,
    bilateral_channels,
    supervised_contrastive_loss,
)


def test_neural_encoders_emit_normalized_embeddings() -> None:
    values = torch.randn(4, 2, 180)
    models = (
        TwoChannelEncoder(embedding_dimension=32, base_channels=8),
        BilateralInteractionEncoder(
            embedding_dimension=32, branch_dimension=8, base_channels=4
        ),
        MultiScaleTwoChannelEncoder(embedding_dimension=32, base_channels=4),
        BilateralAttentionEncoder(
            embedding_dimension=32, branch_dimension=8, base_channels=2
        ),
        V3StackedChannelEncoder(embedding_dimension=32, base_channels=8),
        V3BilateralInteractionEncoder(
            embedding_dimension=32, branch_dimension=8, base_channels=8
        ),
        V3UnilateralEncoder(
            embedding_dimension=32, base_channels=8, channel_index=0
        ),
    )
    for model in models:
        embeddings = model(values)
        assert embeddings.shape == (4, 32)
        assert torch.allclose(torch.linalg.vector_norm(embeddings, dim=1), torch.ones(4))


def test_arc_margin_head_returns_training_class_logits() -> None:
    embeddings = torch.nn.functional.normalize(torch.randn(6, 16), dim=1)
    labels = torch.tensor([0, 0, 1, 1, 2, 2])
    head = ArcMarginHead(16, 3)
    logits = head(embeddings, labels)
    assert logits.shape == (6, 3)
    assert torch.isfinite(logits).all()


def test_supervised_contrastive_loss_is_finite_with_pk_batch() -> None:
    embeddings = torch.nn.functional.normalize(torch.randn(6, 16), dim=1)
    labels = torch.tensor([0, 0, 1, 1, 2, 2])
    loss = supervised_contrastive_loss(embeddings, labels)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert loss > 0


def test_v3_channel_construction_tracks_missing_nostril() -> None:
    values = torch.randn(3, 2, 180)
    values[1, 0] = 0.0
    stacked, availability = bilateral_channels(values)
    assert stacked.shape == (3, 6, 180)
    assert availability.tolist()[1] == [0.0, 1.0]
    assert torch.equal(stacked[:, 2], values[:, 0] + values[:, 1])
    assert torch.equal(stacked[:, 3], values[:, 0] - values[:, 1])


def test_v3_encoders_are_parameter_matched_at_proposal_capacity() -> None:
    stacked = V3StackedChannelEncoder(
        embedding_dimension=256, base_channels=72
    )
    bilateral = V3BilateralInteractionEncoder(
        embedding_dimension=256, branch_dimension=128, base_channels=40
    )
    stacked_parameters = sum(parameter.numel() for parameter in stacked.parameters())
    bilateral_parameters = sum(parameter.numel() for parameter in bilateral.parameters())
    unilateral = V3UnilateralEncoder(
        embedding_dimension=256, base_channels=72, channel_index=0
    )
    unilateral_parameters = sum(
        parameter.numel() for parameter in unilateral.parameters()
    )
    relative_difference = abs(stacked_parameters - bilateral_parameters) / max(
        stacked_parameters, bilateral_parameters
    )
    assert 2_000_000 <= stacked_parameters <= 3_000_000
    assert 2_000_000 <= bilateral_parameters <= 3_000_000
    assert relative_difference < 0.05
    assert abs(unilateral_parameters - bilateral_parameters) / bilateral_parameters < 0.05


def test_unilateral_encoder_ignores_the_other_channel() -> None:
    torch.manual_seed(3)
    model = V3UnilateralEncoder(
        embedding_dimension=16, base_channels=8, channel_index=0
    ).eval()
    first = torch.randn(2, 2, 180)
    second = first.clone()
    second[:, 1] = 100.0 * torch.randn_like(second[:, 1])
    with torch.inference_mode():
        assert torch.equal(model(first), model(second))


def test_v3_bie_ablation_variants_emit_normalized_embeddings() -> None:
    values = torch.randn(3, 2, 180)
    parameter_counts = {}
    for variant in V3BIEAblationEncoder.VARIANTS:
        model = V3BIEAblationEncoder(
            variant, embedding_dimension=32, branch_dimension=8, base_channels=8
        )
        embeddings = model(values)
        assert embeddings.shape == (3, 32)
        assert torch.allclose(
            torch.linalg.vector_norm(embeddings, dim=1), torch.ones(3), atol=1e-6
        )
        parameter_counts[variant] = sum(
            parameter.numel() for parameter in model.parameters()
        )
    assert parameter_counts["independent_nostrils"] > parameter_counts["no_global"]


def test_no_interaction_ablation_remains_side_aware_via_asymmetry() -> None:
    torch.manual_seed(19)
    model = V3BIEAblationEncoder(
        "no_interaction", embedding_dimension=16, branch_dimension=8, base_channels=8
    ).eval()
    # Global is symmetric but asymmetry is signed, so disabling only interaction
    # is intentionally still side-aware and should generally change on swap.
    values = torch.randn(2, 2, 180)
    with torch.inference_mode():
        assert not torch.equal(model(values), model(values[:, [1, 0]]))
