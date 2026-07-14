import torch
from torch import nn
from torch.nn import functional as F
import math

def get_embedding(hidden_state, attention_mask, cls_pooling=False):
    if cls_pooling:
        return hidden_state[:, 0, :]

    cls_token = hidden_state[:, 0:1, :]

    attention_scores = torch.bmm(cls_token, hidden_state.transpose(1, 2))
    attention_scores = attention_scores.squeeze(1)

    hidden_dim = hidden_state.size(-1)
    attention_scores = attention_scores / math.sqrt(hidden_dim)

    extended_attention_mask = (1.0 - attention_mask) * -1e9
    attention_scores = attention_scores + extended_attention_mask
    weights = F.softmax(attention_scores, dim=-1)
    weights_expanded = weights.unsqueeze(-1)  # (batch_size, seq_len, 1)

    pooled_vector = torch.sum(hidden_state * weights_expanded, dim=1)

    return pooled_vector


class Trainer:
    def __init__(self, student, tokenizer, projectors, args):
        super().__init__()

        self.student = student.train()
        self.tokenizer = tokenizer
        self.projector = projectors
        self.device = student.device
        self.n_layer = student.config.num_hidden_layers

        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')
        self.mse_loss = nn.MSELoss(reduction='mean')

        self.args = args

        self.step = 0

    def compute_scores(self, q_reps, p_reps, temperature=1.0):
        scores = torch.matmul(q_reps, p_reps.transpose(0, 1)) / temperature
        scores = scores.view(q_reps.size(0), -1)
        return scores

    def std_norm(self, embeddings):
        std_embeddings = embeddings.std(dim=-1, keepdim=True).detach()
        norm_embeddings = embeddings / std_embeddings
        return norm_embeddings

    def mse_dim_weight_loss(self, student_embeddings, teacher_embeddings, power=1.0):
        std_embeddings = teacher_embeddings.std(dim=(0))
        mean_std = std_embeddings.mean()
        std_scaled = std_embeddings / mean_std
        weights = std_scaled ** power

        squared_diff = (student_embeddings - teacher_embeddings) ** 2
        weighted_squared_diff = squared_diff * weights
        weighted_mse_loss = weighted_squared_diff.mean()

        return weighted_mse_loss

    def structure_loss(self, student_embeddings, teacher_embeddings):
        student_embeddings = F.normalize(student_embeddings, p=2, dim=-1)
        teacher_embeddings = F.normalize(teacher_embeddings, p=2, dim=-1)

        student_similarity = student_embeddings @ student_embeddings.transpose(-1, -2)
        teacher_similarity = teacher_embeddings @ teacher_embeddings.transpose(-1, -2)

        loss = F.mse_loss(student_similarity, teacher_similarity)

        return loss

    def cosine_loss(self, student_embeddings, teacher_embeddings):
        cos_sim = F.cosine_similarity(student_embeddings, teacher_embeddings, dim=-1)
        cos_sim_loss = 1 - cos_sim
        return cos_sim_loss.mean()

    def distill_loss(self, student_outputs, teacher_embedings=None, attention_mask=None):
        hidden_states = student_outputs.hidden_states

        if teacher_embedings is not None:
            if student_outputs.hidden_states is not None:
                TAMD_loss, LASD_loss = 0, 0

                n_layer_distill = 2
                for i in range(self.n_layer - n_layer_distill + 1, self.n_layer + 1):
                    h_embed = self.get_embedding(hidden_states[i], attention_mask)
                    h_proj = self.projector[i - 1](h_embed)
                    TAMD_loss += self.cosine_loss(h_proj, teacher_embedings)

                TAMD_loss = TAMD_loss / n_layer_distill

                for i in range(1, self.n_layer):
                    h_embedding = self.get_embedding(hidden_states[i], attention_mask)
                    next_h_embedding = self.get_embedding(hidden_states[i+1], attention_mask)
                    LASD_loss += self.structure_loss(h_embedding, next_h_embedding)
                LASD_loss = LASD_loss / (self.n_layer - 1)

        return TAMD_loss, LASD_loss

    def compute_InfoNCE_loss(self, scores, target):
        return self.cross_entropy(scores, target)

    def get_embedding(self, hidden_state, attention_mask):
        return get_embedding(hidden_state, attention_mask)

    def compute_loss(self, student_inputs, teacher_embedings=None):
        inputs = {key: value.to(self.device) for key, value in student_inputs.items()}

        student_q_outputs = self.student(**inputs, return_dict=True)
        student_p_outputs = self.student(**inputs, return_dict=True)

        student_q_embeddings = self.get_embedding(student_q_outputs.last_hidden_state,
                                                  inputs['attention_mask'])
        student_p_embeddings = self.get_embedding(student_p_outputs.last_hidden_state,
                                                  inputs['attention_mask'])
        q_dense_vecs = nn.functional.normalize(student_q_embeddings, dim=-1)
        p_dense_vecs = nn.functional.normalize(student_p_embeddings, dim=-1)

        targets = torch.arange(q_dense_vecs.size(0), device=self.device, dtype=torch.long)

        # dense loss
        dense_scores = self.compute_scores(q_dense_vecs, p_dense_vecs, self.args.temperature)
        simcse_loss = self.compute_InfoNCE_loss(dense_scores, targets)


        TAMD_loss, LASD_loss = self.distill_loss(student_q_outputs,
                                                 teacher_embedings,
                                                 inputs['attention_mask'])


        loss = self.args.lambda1 * simcse_loss + self.args.lambda2 * TAMD_loss + self.args.lambda3 * LASD_loss


        self.step += 1

        return loss, simcse_loss