"""
Smoke tests for prediction heads.

Migrated from the old `if __name__ == "__main__"` block in `models/heads.py`
so these checks run under pytest and keep the module import-clean.
"""

import torch


class TestHeadsSmoke:
    def test_prediction_heads_smoke(self):
        from models.heads import MultiTaskHead, NextVisitPredictionHead, ProgressionSlopeHead

        torch.manual_seed(0)

        batch_size = 4
        seq_len = 10
        d_model = 256

        hidden_states = torch.randn(batch_size, seq_len, d_model)
        attention_mask = torch.ones(batch_size, seq_len)

        # Mask some positions to simulate variable-length sequences
        attention_mask[0, 8:] = 0
        attention_mask[1, 6:] = 0
        attention_mask[2, 9:] = 0

        n_targets = 4  # NP1TOT, NP2TOT, NP3TOT, NP4TOT
        next_visit_head = NextVisitPredictionHead(d_model, n_targets, [128, 64])
        next_visit_preds = next_visit_head(hidden_states)

        assert next_visit_preds.shape == (batch_size, seq_len, n_targets)
        assert torch.isfinite(next_visit_preds).all()

        for pooling in ["mean", "last", "max"]:
            slope_head = ProgressionSlopeHead(d_model, [128, 64], pooling=pooling)
            slope_preds = slope_head(hidden_states, attention_mask)

            assert slope_preds.shape == (batch_size,)
            assert torch.isfinite(slope_preds).all()

        multi_head = MultiTaskHead(d_model, n_targets, [128, 64], [128, 64], pooling="mean")
        outputs = multi_head(hidden_states, attention_mask)

        assert outputs["next_visit"].shape == (batch_size, seq_len, n_targets)
        assert outputs["slope"].shape == (batch_size,)
        assert torch.isfinite(outputs["next_visit"]).all()
        assert torch.isfinite(outputs["slope"]).all()
