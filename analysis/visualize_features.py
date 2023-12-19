from nqgl.sae.sae import AutoEncoder, AutoEncoderConfig
from nqgl.sae.toy_models.toy_model import ToyModel
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt

def get_adjusted_feature_directions(toy :ToyModel, ae :AutoEncoder):
    
    ground_truth_features = toy.features
    v = torch.eye(ae.cfg.d_dict, device="cuda")
    # learned_features = ae.decode(v) - ae.decoder_bias()
    learned_features = ae.W_dec
    learned_features = F.normalize(learned_features, dim=-1)
    ground_truth_adjusted_features = ground_truth_features - ae.unscale(ae.b_dec)

    ground_truth_directions = F.normalize(ground_truth_adjusted_features, dim=-1)
    return ground_truth_directions, learned_features


import seaborn as sns
def visualize_by_heatmap(toy :ToyModel, ae :AutoEncoder, special_features = [], special_labels = []):
    # TODO selection of rows and columns
    # TODO ax parameter so you can plot multiple
    # TODO color scheme where 0 is always black?
    plt.ion()   

    ground_truth, learned = get_adjusted_feature_directions(toy, ae)
                            # y-axis @ x-axis
    features_similarity = learned @ ground_truth.transpose(-2, -1)
    bias_times_gt = F.normalize(ae.b_dec.view(1,-1), dim=-1) @ F.normalize(toy.features, dim=-1).transpose(-2, -1)
    
    features_similarity = torch.cat((
        bias_times_gt,
        features_similarity), 
    dim=0)

    permuted_heatmap(F.relu(features_similarity))


def permuted_heatmap(mat :torch.Tensor, special_features = None, special_labels = []):
    # Step 1: Compute max values and their indices
    max_values, max_indices = mat.max(dim=-1)
    
    # Step 2: Sort by magnitude (descending order)
    mag_sort_indices = torch.argsort(max_values)
    sorted_by_magnitude = mat[mag_sort_indices]

    # Step 3: Now sort by indices of max elements within each row
    # We create a tuple (max_indices, -row_index) to ensure that ties in max_indices are resolved by the row index
    indices_and_row = list(zip(max_indices[mag_sort_indices], -torch.arange(len(max_indices))))
    sorted_indices = sorted(range(len(indices_and_row)), key=lambda i: indices_and_row[i], reverse=False)
    
    # Apply final sorting
    final_sorted_mat = sorted_by_magnitude[sorted_indices]

    # Plotting
    plt.clf()
    print(final_sorted_mat.shape)
    ytick_names = list(range(-1, mat.shape[0]))
    ytick_names[0] = "bias only"
    yticks = [ytick_names[i] for i in sorted_indices]
    sns.heatmap(
        final_sorted_mat.cpu().detach().numpy(), 
        yticklabels=yticks)
    plt.title("Feature Cosine Similarities")
    plt.xlabel("Ground Truth")
    plt.ylabel("Learned")
    plt.draw()
    plt.pause(0.25)


    # magsort = mat.max(dim=-1).values.argsort()

    # maxmat = mat[magsort]
    # max_args = maxmat.argmax(dim=-1)
    # maxsort = max_args.argsort()
    # argsort_max = maxsort[magsort]

    # inv_argsort = torch.zeros_like(argsort_max, device=argsort_max.device)
    # inv_argsort[argsort_max] = torch.arange(inv_argsort.shape[0], device=argsort_max.device)
    # features_similarity_sorted = mat[argsort_max, :]
    # plt.clf()
    # print(features_similarity_sorted.shape)
    # ytick_names = list(range(-1, mat.shape[0]))
    # ytick_names[0] = "bias only"
    # yticks = [ytick_names[i.item()] for i in argsort_max]

    # sns.heatmap(features_similarity_sorted.cpu().detach().numpy(), yticklabels=yticks)
    # plt.draw()
    # plt.pause(0.05)
    





# def 


# for i in range(20, 15, -1):
#     f1 = f[i, :]
#     print(f1.shape)
#     print(m @ f1)
#     print(f1 @ m.T)
#     print((m @ f1).shape)

#     import matplotlib.pyplot as plt

#     # Your existing code...

#     result = m @ f1

#     # Plotting the bar graph
#     plt.bar(range(len(result)), result.cpu().detach().numpy())
#     plt.xlabel('Index')
#     plt.ylabel('Value')
#     plt.title('m @ f1')
#     plt.show()
# # heatmap of m @ f

# vm = torch.eye(m.shape[-2])
# vm = vm.cuda()
# vm = torch.nn.functional.normalize(vm, dim=-2)
# vm = vm @ m
# vin = ae.encode(vm)



# import matplotlib.pyplot as plt

# # Your existing code...

# # Plotting the bar graph
# result = vin[0]
# print(vin.shape)
# print(sum(vin))
# plt.bar(range(len(result)), result.cpu().detach().numpy())
# for i in range(200, 0, -1):
#     plt.plot(vin[i, :].cpu().detach().numpy(), label=f"{i}")
# plt.xlabel('Index')
# plt.ylabel('Value')
# plt.title('m @ f1')
# plt.show()



def chf(sae, toy :ToyModel):
    samples, feature_acts = toy.next(return_active_features = True)
    acts = sae.encode(samples)

    for i in range(samples.shape[0]):
        sample = samples[i]
        dict_active = acts[i]
        features_active = feature_acts[i]
        active_acts = (dict_active>0).to_sparse().indicies()
        for mfi in active_acts:
            ablated = dict_active.clone()
            ablated[mfi] = 0
            sample_mfi_ablated = sae.decode()