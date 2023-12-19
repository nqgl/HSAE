import torch
import torch.nn as nn
import torch.nn.functional as F

class BaseSAE(nn.Module):

    @torch.no_grad()
    def neurons_to_reset(self, to_be_reset :torch.Tensor):
        if to_be_reset.sum() > 0:
            self.neurons_to_be_reset = torch.argwhere(to_be_reset).squeeze(1)
            w_enc_norms = self.W_enc[:, ~ to_be_reset].norm(dim=0)
            # print("w_enc_norms", w_enc_norms.shape)
            # print("to_be_reset", self.to_be_reset.sum())
            self.alive_norm_along_feature_axis = torch.mean(torch.mean(w_enc_norms))
        else:
            self.neurons_to_be_reset = None
    
    @torch.no_grad()
    def re_init_neurons(self, x_diff):
        self.re_init_neurons_gram_shmidt_precise_iterative_argmax(x_diff)


    def queue_neurons_to_reset(self, to_be_reset :torch.Tensor):
        if to_be_reset.sum() > 0:
            self.neurons_to_be_reset = torch.argwhere(to_be_reset).squeeze(1)
            w_enc_norms = self.W_enc[:, ~ to_be_reset].norm(dim=0)
            # print("w_enc_norms", w_enc_norms.shape)
            # print("to_be_reset", self.to_be_reset.sum())
            self.alive_norm_along_feature_axis = torch.mean(torch.mean(w_enc_norms))
        else:
            self.neurons_to_be_reset = None



    @torch.no_grad()
    def re_init_neurons(self, x_diff):
        self.re_init_neurons_gram_shmidt_precise_iterative_argmax(x_diff)

    @torch.no_grad()
    def re_init_neurons_gram_shmidt_precise_iterative_argmax(self, x_diff):
        n_reset = min(x_diff.shape[0], self.cfg.d_data // 2, self.cfg.num_to_resample)
        v_orth = torch.zeros_like(x_diff)
        n_succesfully_reset = n_reset
        for i in range(n_reset):
            magnitudes = x_diff.norm(dim=-1)
            i_max = torch.argmax(magnitudes)
            v_orth[i] = x_diff[i_max]
            for j in range(max(0, i - self.cfg.gram_shmidt_trail), i):
                v_orth[i] -= torch.dot(v_orth[j], v_orth[i]) * v_orth[j] / torch.dot(v_orth[j], v_orth[j])
            if v_orth[i].norm() < 1e-6:
                n_succesfulselfly_reset = i
                break
            v_orth[i] = F.normalize(v_orth[i], dim=-1)
            x_diff -= (x_diff @ v_orth[i]).unsqueeze(1) * v_orth[i] / torch.dot(v_orth[i], v_orth[i])
            # v_ = x_diff[i] - v_bar * torch.dot(v_bar, x_diff[i])
            # # print(v_.shape)
            # v_orth[i] = v_ / v_.norm(dim=-1, keepdim=True)
        self.reset_neurons(v_orth[:n_succesfully_reset])


    @torch.no_grad()
    def re_init_neurons_gram_shmidt_precise_topk(self, x_diff):
        t = self.cfg.gram_shmidt_trail
        n_reset = min(x_diff.shape[0], self.cfg.d_data // 2, self.cfg.num_to_resample)
        v_orth = torch.zeros_like(x_diff)
        # print(x_diff.shape)
        # v_orth[0] = F.normalize(x_diff[0], dim=-1)
        magnitudes = x_diff.norm(dim=-1)
        indices = torch.topk(magnitudes, n_reset).indices
        x_diff = x_diff[indices]
        n_succesfully_reset = n_reset
        for i in range(n_reset):
            v_orth[i] = x_diff[i]
            for j in range(max(0, i - t), i):
                v_orth[i] -= torch.dot(v_orth[j], v_orth[i]) * v_orth[j] / torch.dot(v_orth[j], v_orth[j])
            if v_orth[i].norm() < 1e-6:
                n_succesfully_reset = i
                break
            v_orth[i] = F.normalize(v_orth[i], dim=-1)
            # v_ = x_diff[i] - v_bar * torch.dot(v_bar, x_diff[i])
            # # print(v_.shape)
            # v_orth[i] = v_ / v_.norm(dim=-1, keepdim=True)
        self.reset_neurons(v_orth[:n_succesfully_reset])


    
    @torch.no_grad()
    def re_init_neurons_gram_shmidt_precise(self, x_diff):
        t = self.cfg.gram_shmidt_trail
        n_reset = min(x_diff.shape[0], self.cfg.d_data // 2, self.cfg.num_to_resample)
        v_orth = torch.zeros_like(x_diff)
        # print(x_diff.shape)
        # v_orth[0] = F.normalize(x_diff[0], dim=-1)
        n_succesfully_reset = n_reset
        for i in range(n_reset):
            v_orth[i] = x_diff[i]
            for j in range(max(0, i - t), i):
                v_orth[i] -= torch.dot(v_orth[j], v_orth[i]) * v_orth[j] / torch.dot(v_orth[j], v_orth[j])
            if v_orth[i].norm() < 1e-6:
                n_succesfully_reset = i
                break
            v_orth[i] = F.normalize(v_orth[i], dim=-1)
            # v_ = x_diff[i] - v_bar * torch.dot(v_bar, x_diff[i])
            # # print(v_.shape)
            # v_orth[i] = v_ / v_.norm(dim=-1, keepdim=True)
        self.reset_neurons(v_orth[:n_succesfully_reset])
