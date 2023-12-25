
cfg = AutoEncoderConfig(
    site="toy_model",
    d_data=64,
    layer=1,
    gram_shmidt_trail=512,
    num_to_resample=4,
    l1_coeff=14e-4,
    dict_mult=1,
    batch_size=128,
    beta2=0.999,
    lr=3e-4,
)


def main():
    encoder = AutoEncoder(cfg)
    toycfg = ToyModelConfig(
        d_data=cfg.d_data,
        n_features=32,
        num_correlation_rounds=10,
        batch_size=cfg.batch_size,
        blank_correlations=False,
        initial_features=3,
        features_per_round=2,
        features_per_round_negative=4,
        seed=50,
        correlation_drop=0.9,
        source_prob_drop=0.85,
    )

    toy = ToyModel(toycfg)
    toy.f_means[:] = 1
    train(encoder, cfg, toy)


if __name__ == "__main__":
    main()
