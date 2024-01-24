from nqgl.sae.sae import AutoEncoder, AutoEncoderConfig
from nqgl.sae.hsae.hsae import HierarchicalAutoEncoder, HierarchicalAutoEncoderLayer
from nqgl.sae.toy_models.toy_model import ToyModel
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
import matplotlib.cm as cm
import cmasher as cmr
import matplotlib.colors as mcolors


def get_adjusted_feature_directions(toy: ToyModel, ae: AutoEncoder):
    ground_truth_features = toy.features * toy.f_means.view(-1, 1)
    v = torch.eye(ae.cfg.d_dict, device="cuda")
    # learned_features = ae.decode(v) - ae.decoder_bias()
    learned_features = ae.W_dec
    learned_features = F.normalize(learned_features, dim=-1)
    ground_truth_adjusted_features = ground_truth_features - ae.unscale(ae.b_dec)

    ground_truth_directions = F.normalize(ground_truth_adjusted_features, dim=-1)
    return ground_truth_directions, learned_features


def get_adjusted_feature_directions_from_hierarchical(toy: ToyModel, ae: HierarchicalAutoEncoder, layer, feature_index):
    ground_truth_features = toy.features * toy.f_means.view(-1, 1)

    learned_features, bias = ae.get_features_and_bias(layer, feature_index)
    learned_features = F.normalize(learned_features, dim=-1)
    ground_truth_adjusted_features = ground_truth_features - bias

    ground_truth_directions = F.normalize(ground_truth_adjusted_features, dim=-1)
    return ground_truth_directions, learned_features



def get_adjusted_feature_directions2(toy: ToyModel, ae: AutoEncoder):
    ground_truth_features = toy.features * toy.f_means.view(-1, 1)
    v = torch.eye(ae.cfg.d_dict, device="cuda")

    # learned_features = ae.decode(v) - ae.decoder_bias()
    learned_features = ae.b_dec + ae.W_dec
    learned_features = F.normalize(learned_features, dim=-1)
    ground_truth_adjusted_features = ground_truth_features

    ground_truth_directions = F.normalize(ground_truth_adjusted_features, dim=-1)
    return ground_truth_directions, learned_features


import seaborn as sns


def visualize_by_heatmap(
    toy: ToyModel, ae: AutoEncoder, special_features=[], special_labels=[]
):
    # TODO selection of rows and columns
    # TODO ax parameter so you can plot multiple
    # TODO color scheme where 0 is always black?
    plt.ion()

    ground_truth, learned = get_adjusted_feature_directions(toy, ae)
    # y-axis @ x-axis
    features_similarity = learned @ ground_truth.transpose(-2, -1)
    bias_times_gt = F.normalize(ae.b_dec.view(1, -1), dim=-1) @ F.normalize(
        toy.features, dim=-1
    ).transpose(-2, -1)

    features_similarity = torch.cat((bias_times_gt, features_similarity), dim=0)

    mat, yticks = permuted_heatmap(features_similarity)
    show_as_heatmap(mat, yticks)


def visualize_by_heatmap2(
    toy: ToyModel, ae: AutoEncoder, special_features=[], special_labels=[]
):
    # TODO selection of rows and columns
    # TODO ax parameter so you can plot multiple
    # TODO color scheme where 0 is always black?
    plt.ion()

    ground_truth, learned = get_adjusted_feature_directions(toy, ae)
    ground_truth2, learned2 = get_adjusted_feature_directions2(toy, ae)
    ground_truth = torch.cat((ground_truth, ground_truth2), dim=0)

    # y-axis @ x-axis
    features_similarity = learned @ ground_truth.transpose(-2, -1)
    bias_times_gt = F.normalize(ae.b_dec.view(1, -1), dim=-1) @ F.normalize(
        toy.features, dim=-1
    ).transpose(-2, -1)
    bias_times_gt = torch.cat((bias_times_gt, bias_times_gt), dim=1)

    features_similarity = torch.cat((bias_times_gt, features_similarity), dim=0)

    mat, yticks, perm = permuted_heatmap(
        features_similarity[:, : features_similarity.shape[1] // 2], return_perm=True
    )
    ytick_names = list(range(-1, mat.shape[0]))
    show_as_heatmap(features_similarity[perm], yticks)

    # mat, yticks = permuted_heatmap(features_similarity)
    # show_as_heatmap(mat, yticks)


def permuted_heatmap(
    mat: torch.Tensor, special_features=None, special_labels=[], return_perm=False
):
    # Step 1: Compute max values and their indices
    max_values, max_indices = mat.max(dim=-1)

    # Step 2: Sort by magnitude (descending order)
    mag_sort_indices = torch.argsort(max_values)
    sorted_by_magnitude = mat[mag_sort_indices]

    # Step 3: Now sort by indices of max elements within each row
    # We create a tuple (max_indices, -row_index) to ensure that ties in max_indices are resolved by the row index
    indices_and_row = list(
        zip(max_indices[mag_sort_indices], -torch.arange(len(max_indices)))
    )
    sorted_indices = sorted(
        range(len(indices_and_row)), key=lambda i: indices_and_row[i], reverse=False
    )

    # Apply final sorting
    fullsort = mag_sort_indices[sorted_indices]
    final_sorted_mat = sorted_by_magnitude[sorted_indices]
    ytick_names = list(range(-1, mat.shape[0]))
    ytick_names[0] = "bias only"
    yticks = [ytick_names[i] for i in fullsort]
    if return_perm:
        return final_sorted_mat, yticks, fullsort
    return final_sorted_mat, yticks


def show_as_heatmap(mat, yticks):
    # Plotting
    plt.clf()
    print(mat.shape)
    cpumat = mat.cpu().detach().numpy()
    sns.heatmap(cpumat, yticklabels=yticks, **color_map_0_black(cpumat))
    plt.title("Feature Cosine Similarities")
    plt.xlabel("Ground Truth")
    plt.ylabel("Learned")
    plt.draw()
    plt.pause(0.25)


def color_map_0_black(mat: torch.Tensor):
    v = max(abs(mat.min()), abs(mat.max()))
    normalize = mcolors.TwoSlopeNorm(vmin=-1.25 * v, vcenter=0, vmax=1 * v)
    colors = cmr.redshift
    return {
        "center": 0.0,
        # "norm": normalize,
        "cmap": colors,
    }


def chf(sae, toy: ToyModel):
    samples, feature_acts = toy.next(return_active_features=True)
    acts = sae.encode(samples)

    for i in range(samples.shape[0]):
        sample = samples[i]
        dict_active = acts[i]
        features_active = feature_acts[i]
        active_acts = (dict_active > 0).to_sparse().indicies()
        for mfi in active_acts:
            ablated = dict_active.clone()
            ablated[mfi] = 0
            sample_mfi_ablated = sae.decode()
