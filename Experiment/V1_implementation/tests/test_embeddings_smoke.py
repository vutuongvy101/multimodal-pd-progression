"""
Smoke tests for embedding modules.

These tests were migrated from the old `if __name__ == "__main__":` block in
`models/embeddings.py` to keep library code import-clean and to make the checks
run under pytest.
"""

import torch


class TestEmbeddingsSmoke:
    def test_embeddings_and_visit_token_builder_smoke(self):
        from models.embeddings import (
            SinusoidalTimeEncoding,
            StaticFeatureEmbedding,
            VisitFeatureEmbedding,
            VisitTokenBuilder,
        )

        batch_size = 4
        seq_len = 10
        n_motor_features = 35
        n_nonmotor_features = 20
        n_med_features = 5
        n_static_features = 25
        d_model = 256

        # Create test data
        motor_values = torch.randn(batch_size, seq_len, n_motor_features)
        motor_mask = torch.bernoulli(torch.ones_like(motor_values) * 0.1)

        nonmotor_values = torch.randn(batch_size, seq_len, n_nonmotor_features)
        nonmotor_mask = torch.bernoulli(torch.ones_like(nonmotor_values) * 0.2)

        med_values = torch.randn(batch_size, seq_len, n_med_features)
        med_mask = torch.bernoulli(torch.ones_like(med_values) * 0.05)

        static_values = torch.randn(batch_size, n_static_features)
        static_mask = torch.bernoulli(torch.ones_like(static_values) * 0.15)

        time_months = torch.linspace(0, 48, seq_len).unsqueeze(0).expand(batch_size, -1)

        # Create embedding modules
        static_embedding = StaticFeatureEmbedding(n_static_features, d_model, [128])
        motor_embedding = VisitFeatureEmbedding(n_motor_features, d_model, [128])
        nonmotor_embedding = VisitFeatureEmbedding(n_nonmotor_features, d_model, [128])
        med_embedding = VisitFeatureEmbedding(n_med_features, d_model, [64])
        time_encoding = SinusoidalTimeEncoding(d_model, max_time=120.0)

        # Forward pass through individual embeddings
        static_emb = static_embedding(static_values, static_mask)
        motor_emb = motor_embedding(motor_values, motor_mask)
        nonmotor_emb = nonmotor_embedding(nonmotor_values, nonmotor_mask)
        med_emb = med_embedding(med_values, med_mask)
        time_emb = time_encoding(time_months)

        assert static_emb.shape == (batch_size, d_model)
        assert motor_emb.shape == (batch_size, seq_len, d_model)
        assert nonmotor_emb.shape == (batch_size, seq_len, d_model)
        assert med_emb.shape == (batch_size, seq_len, d_model)
        assert time_emb.shape == (batch_size, seq_len, d_model)

        # Test VisitTokenBuilder with all modalities
        enabled_all = ["static", "motor", "nonmotor", "medication"]
        visit_builder_all = VisitTokenBuilder(d_model, enabled_all)
        embeddings_dict = {
            "static": static_emb,
            "motor": motor_emb,
            "nonmotor": nonmotor_emb,
            "medication": med_emb,
        }

        visit_tokens = visit_builder_all(embeddings_dict, seq_len)
        visit_tokens = visit_tokens + time_emb
        assert visit_tokens.shape == (batch_size, seq_len, d_model)

        # Test with subset of modalities (ablation)
        enabled_subset = ["static", "motor"]
        visit_builder_subset = VisitTokenBuilder(d_model, enabled_subset)
        embeddings_subset = {"static": static_emb, "motor": motor_emb}

        visit_tokens_subset = visit_builder_subset(embeddings_subset, seq_len)
        visit_tokens_subset = visit_tokens_subset + time_emb
        assert visit_tokens_subset.shape == (batch_size, seq_len, d_model)
