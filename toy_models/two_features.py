from nqgl.sae.toy_models.toy_model import ToyModel, ToyModelConfig
import torch
import torch.nn.functional as F

def gram_schmit(vecs, trail=None):
    if trail is None:
        trail = vecs.shape[0]
    v_orth = torch.zeros_like(vecs)
    for i in range(vecs.shape[0]):
        v_orth[i] = vecs[i]
        for j in range(max(0, i - trail), i):
            v_orth[i] -= torch.dot(v_orth[j], v_orth[i]) * v_orth[j] / torch.dot(v_orth[j], v_orth[j])
        if v_orth[i].norm() < 1e-6:
            n_succesfully_reset = i
            break
        v_orth[i] = F.normalize(v_orth[i], dim=-1)
        # v_ = vecs[i] - v_bar * torch.dot(v_bar, vecs[i])
        # # print(v_.shape)
        # v_orth[i] = v_ / v_.norm(dim=-1, keepdim=True)
    return v_orth


def get_simple_hierarchy_model(d_data, n_features = 3, rounds = 1):
    cfg = ToyModelConfig(
        d_data=d_data,
        n_features=n_features,
        features_per_round=1,
        features_per_round_negative=1,
        num_correlation_rounds=rounds,
        batch_size=512,
        initial_features=1,
        seed = 55,
        blank_correlations = True,
    )
    model = ToyModel(cfg)

    model.f_means[:] += 1
    # model.f_stds[:] = 0.1
    # model.features[0][0] = 1
    # model.features[0][1] = 0
    # model.features[1][0] = 0
    # model.features[1][1] = 1
    model.features = gram_schmit(model.features, trail=d_data)
    # model.features[2][:] = 0
    model.f_probs[0] = 0.5
    model.f_probs[1] = 0
    model.f_probs[2] = 0.5

    model.correlations[0][:, 1] = 0
    model.correlations[0][0][1] = 0.5
    model.correlations[0][0][0] = 0.5
    model.correlations[0][2][2] = 0.5
    print(model.correlations)
    samples = model.get_sample(500)
    return model



def main():
    m = get_simple_hierarchy_model(3)
    test = m.features @ m.features.T
    print(test)
    print(m.get_sample(18))

if __name__ == "__main__":
    main()