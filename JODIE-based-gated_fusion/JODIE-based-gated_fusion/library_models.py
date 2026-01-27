'''
This is a supporting library with the code of the model.

Paper: Predicting Dynamic Embedding Trajectory in Temporal Interaction Networks. S. Kumar, X. Zhang, J. Leskovec. ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD), 2019. 
'''

from __future__ import division
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
from torch import optim
import numpy as np
import math, random
import sys
from collections import defaultdict
import os
import gpustat
from itertools import chain
from tqdm import tqdm, trange, tqdm_notebook, tnrange
import csv
import json

PATH = "./"

try:
    get_ipython
    trange = tnrange
    tqdm = tqdm_notebook
except NameError:
    pass

total_reinitialization_count = 0

# A NORMALIZATION LAYER
class NormalLinear(nn.Linear):
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.normal_(0, stdv)
        if self.bias is not None:
            self.bias.data.normal_(0, stdv)


# THE JODIE MODULE
class JODIE(nn.Module):
    def __init__(self, args, num_features, num_users, num_items):
        super(JODIE,self).__init__()

        print("*** Initializing the JODIE model ***")
        self.modelname = args.model
        self.embedding_dim = args.embedding_dim
        self.num_users = num_users
        self.num_items = num_items
        self.user_static_embedding_size = num_users
        self.item_static_embedding_size = num_items

        print("Initializing user and item embeddings")
        self.initial_user_embedding = nn.Parameter(torch.Tensor(args.embedding_dim))
        self.initial_item_embedding = nn.Parameter(torch.Tensor(args.embedding_dim))

        # Timediff embedding parameters
        self.timediff_embed_dim = args.timediff_embed_dim
        self.timediff_embed_layers = args.timediff_embed_layers
        self.timediff_activation = args.timediff_activation
        self.timediff_separate_encoder = getattr(args, 'timediff_separate_encoder', False)
        
        # Calculate effective timediff dimension
        if self.timediff_embed_layers == 0:
            # No embedding: use scalar timediff directly
            effective_timediff_dim = 1
        else:
            # Use embedding dimension
            effective_timediff_dim = self.timediff_embed_dim
        self.effective_timediff_dim = effective_timediff_dim
        
        # Build timediff embedding layers only if timediff_embed_layers > 0
        if self.timediff_embed_layers > 0:
            print("Initializing timediff embedding layers")
            # Get activation function
            if self.timediff_activation == 'relu':
                activation_fn = nn.ReLU()
            elif self.timediff_activation == 'leaky_relu':
                activation_fn = nn.LeakyReLU()
            else:
                activation_fn = nn.ReLU()
            
            # Build timediff embedding MLP
            # Activation is applied after every layer
            timediff_layers = []
            for layer_idx in range(self.timediff_embed_layers):
                if layer_idx == 0:
                    # First layer: 1 -> embedding_dim
                    timediff_layers.append(nn.Linear(1, self.timediff_embed_dim))
                else:
                    # Subsequent layers: embedding_dim -> embedding_dim
                    timediff_layers.append(nn.Linear(self.timediff_embed_dim, self.timediff_embed_dim))
                
                # Apply activation after each layer
                timediff_layers.append(activation_fn)
            
            if self.timediff_separate_encoder:
                # Separate encoders for user and item
                self.user_timediff_embedding = nn.Sequential(*timediff_layers)
                self.item_timediff_embedding = nn.Sequential(*[
                    nn.Linear(1, self.timediff_embed_dim) if i == 0 else 
                    nn.Linear(self.timediff_embed_dim, self.timediff_embed_dim) if isinstance(timediff_layers[i], nn.Linear) else 
                    activation_fn
                    for i in range(len(timediff_layers))
                ])
            else:
                # Shared encoder
                self.timediff_embedding = nn.Sequential(*timediff_layers)
        
        rnn_input_size_items = rnn_input_size_users = self.embedding_dim + effective_timediff_dim + num_features

        print("Initializing user and item RNNs")
        self.item_rnn = nn.RNNCell(rnn_input_size_users, self.embedding_dim)
        self.user_rnn = nn.RNNCell(rnn_input_size_items, self.embedding_dim)

        print("Initializing linear layers")
        self.linear_layer1 = nn.Linear(self.embedding_dim, 50)
        self.linear_layer2 = nn.Linear(50, 2)
        self.prediction_layer = nn.Linear(self.user_static_embedding_size + self.item_static_embedding_size + self.embedding_dim * 2, self.item_static_embedding_size + self.embedding_dim)
        self.embedding_layer = NormalLinear(1, self.embedding_dim)
        
        # U-U and I-I view
        print("Initializing U-U and I-I RNNs")
        self.uu_rnn = nn.RNNCell(self.embedding_dim + effective_timediff_dim, self.embedding_dim)
        self.ii_rnn = nn.RNNCell(self.embedding_dim + effective_timediff_dim, self.embedding_dim)

        # Gate for local/global fusion (dimension matches RNN input)
        gate_in_dim = 2 * (self.embedding_dim + effective_timediff_dim)
        gate_out_dim = self.embedding_dim + effective_timediff_dim
        self.uu_gate = nn.Linear(gate_in_dim, gate_out_dim)
        self.ii_gate = nn.Linear(gate_in_dim, gate_out_dim)

        
        print("*** JODIE initialization complete ***\n\n")
        
    def forward(self, user_embeddings, item_embeddings, timediffs=None, features=None, select=None):
        timediff_embedded = self._embed_timediff(timediffs, select)
        
        if select == 'item_update':
            input1 = torch.cat([user_embeddings, timediff_embedded, features], dim=1)
            item_embedding_output = self.item_rnn(input1, item_embeddings)
            return F.normalize(item_embedding_output)

        elif select == 'user_update':
            input2 = torch.cat([item_embeddings, timediff_embedded, features], dim=1)
            user_embedding_output = self.user_rnn(input2, user_embeddings)
            return F.normalize(user_embedding_output)

        elif select == 'project':
            user_projected_embedding = self.context_convert(user_embeddings, timediffs, features)
            #user_projected_embedding = torch.cat([input3, item_embeddings], dim=1)
            return user_projected_embedding

    def context_convert(self, embeddings, timediffs, features):
        new_embeddings = embeddings * (1 + self.embedding_layer(timediffs))
        return new_embeddings

    def _embed_timediff(self, timediffs, view: str):
        """
        view: 'user_update' / 'item_update' / 'uu' / 'ii'
        """
        if self.timediff_embed_layers == 0:
            return timediffs

        if self.timediff_separate_encoder:
            if view in ('user_update', 'uu'):
                return self.user_timediff_embedding(timediffs)
            elif view in ('item_update', 'ii'):
                return self.item_timediff_embedding(timediffs)

            # default to user encoder when view is ambiguous
            return self.user_timediff_embedding(timediffs)

        return self.timediff_embedding(timediffs)

    def predict_label(self, user_embeddings):
        X_out = nn.ReLU()(self.linear_layer1(user_embeddings))
        X_out = self.linear_layer2(X_out)
        return X_out

    def predict_item_embedding(self, user_embeddings):
        X_out = self.prediction_layer(user_embeddings)
        return X_out
    
    def _sim_weighted_agg(self, neigh_embs, neigh_sims, neigh_mask):
        """
        neigh_embs: (B, L, D)
        neigh_sims: (B, L)
        neigh_mask: (B, L) bool
        return: (B, D)
        """
        w = neigh_sims * neigh_mask.float()
        w = w / (w.sum(dim=1, keepdim=True) + 1e-12)
        return torch.sum(w.unsqueeze(-1) * neigh_embs, dim=1)

    def encode_sim_view(self, prev_state, pos_embs, pos_sims, pos_mask, timediffs, view: str):
        """
        Single-window encoder (legacy): aggregates one neighbor set and updates RNN.
        prev_state: (B, D) - uu_state[u] or ii_state[i]
        pos_embs:  (B, L, D) - neighbor embeddings (already padded)
        pos_sims:  (B, L)
        pos_mask:  (B, L)
        timediffs: (B, 1)
        view: 'uu' or 'ii'
        """
        
        x = self._sim_weighted_agg(pos_embs, pos_sims, pos_mask)  # (B, D)
        timediff_embedded = self._embed_timediff(timediffs, view)

        input_vec = torch.cat([x, timediff_embedded], dim=1)

        if view == 'uu':
            uu_embedding_output = self.uu_rnn(input_vec, prev_state)
            return F.normalize(uu_embedding_output)
        elif view == 'ii':
            ii_embedding_output = self.ii_rnn(input_vec, prev_state)
            return F.normalize(ii_embedding_output)

    def encode_sim_view_fused(self, prev_state, pos_local_embs, pos_local_sims, pos_local_mask,
                              pos_global_embs, pos_global_sims, pos_global_mask,
                              timediffs, view: str):
        """
        Dual-window encoder: aggregates local/global, applies gated fusion, then updates RNN.
        """
        local_agg = self._sim_weighted_agg(pos_local_embs, pos_local_sims, pos_local_mask)
        global_agg = self._sim_weighted_agg(pos_global_embs, pos_global_sims, pos_global_mask)
        timediff_embedded = self._embed_timediff(timediffs, view)

        local_input = torch.cat([local_agg, timediff_embedded], dim=1)
        global_input = torch.cat([global_agg, timediff_embedded], dim=1)

        gate_input = torch.cat([local_input, global_input], dim=1)
        if view == 'uu':
            gate = torch.sigmoid(self.uu_gate(gate_input))
            fused_input = gate * local_input + (1 - gate) * global_input
            uu_embedding_output = self.uu_rnn(fused_input, prev_state)
            return F.normalize(uu_embedding_output)
        elif view == 'ii':
            gate = torch.sigmoid(self.ii_gate(gate_input))
            fused_input = gate * local_input + (1 - gate) * global_input
            ii_embedding_output = self.ii_rnn(fused_input, prev_state)
            return F.normalize(ii_embedding_output)
        
    def infonce(self, query, pos_key, neg_keys, neg_mask, tau: float):
        """
        query:    (B, D)
        pos_key:  (B, D)
        neg_keys: (B, N, D)
        neg_mask: (B, N) bool
        """
        query = F.normalize(query, dim=-1)
        pos_key = F.normalize(pos_key, dim=-1)
        neg_keys = F.normalize(neg_keys, dim=-1)

        pos_logit = torch.sum(query * pos_key, dim=-1, keepdim=True) / tau      # (B,1)
        neg_logit = torch.sum(query.unsqueeze(1) * neg_keys, dim=-1) / tau      # (B,N)
        neg_logit = neg_logit.masked_fill(~neg_mask, -1e9)

        logits = torch.cat([pos_logit, neg_logit], dim=1)                        # (B,1+N)
        labels = torch.zeros(query.size(0), dtype=torch.long, device=query.device)
        return F.cross_entropy(logits, labels)

# INITIALIZE T-BATCH VARIABLES
def reinitialize_tbatches():
    global current_tbatches_interactionids, current_tbatches_user, current_tbatches_item, current_tbatches_timestamp, current_tbatches_feature, current_tbatches_label, current_tbatches_previous_item
    global tbatchid_user, tbatchid_item, current_tbatches_user_timediffs, current_tbatches_item_timediffs, current_tbatches_user_timediffs_next

    # list of users of each tbatch up to now
    current_tbatches_interactionids = defaultdict(list)
    current_tbatches_user = defaultdict(list)
    current_tbatches_item = defaultdict(list)
    current_tbatches_timestamp = defaultdict(list)
    current_tbatches_feature = defaultdict(list)
    current_tbatches_label = defaultdict(list)
    current_tbatches_previous_item = defaultdict(list)
    current_tbatches_user_timediffs = defaultdict(list)
    current_tbatches_item_timediffs = defaultdict(list)
    current_tbatches_user_timediffs_next = defaultdict(list)

    # the latest tbatch a user is in
    tbatchid_user = defaultdict(lambda: -1)

    # the latest tbatch a item is in
    tbatchid_item = defaultdict(lambda: -1)

    global total_reinitialization_count
    total_reinitialization_count +=1


# CALCULATE LOSS FOR THE PREDICTED USER STATE 
def calculate_state_prediction_loss(model, tbatch_interactionids, user_embeddings_time_series, y_true, loss_function):
    # PREDCIT THE LABEL FROM THE USER DYNAMIC EMBEDDINGS
    prob = model.predict_label(user_embeddings_time_series[tbatch_interactionids,:])
    y = Variable(torch.LongTensor(y_true).cuda()[tbatch_interactionids])
    
    loss = loss_function(prob, y)

    return loss


# SAVE TRAINED MODEL TO DISK
def save_model(model, optimizer, args, epoch, user_embeddings, item_embeddings, train_end_idx, user_embeddings_time_series=None, item_embeddings_time_series=None, path=PATH):
    print("*** Saving embeddings and model ***")
    state = {
            'user_embeddings': user_embeddings.data.cpu().numpy(),
            'item_embeddings': item_embeddings.data.cpu().numpy(),
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer' : optimizer.state_dict(),
            'train_end_idx': train_end_idx
            }

    if user_embeddings_time_series is not None:
        state['user_embeddings_time_series'] = user_embeddings_time_series.data.cpu().numpy()
        state['item_embeddings_time_series'] = item_embeddings_time_series.data.cpu().numpy()

    directory = os.path.join(path, 'saved_models/%s/' % args.network)
    if not os.path.exists(directory):
        os.makedirs(directory)

    filename = f'saved_models/{args.network}/{args.model}_{epoch}_wt{args.window_type}_ws{args.window_size}_tk{args.top_k}_th{args.threshold}_bk{args.bottom_k}_md{args.min_deg}_clw{args.cl_weight}_clt{args.tau}.pth.tar'
    torch.save(state, filename)
    print(f"*** Saved embeddings and model to file: {filename} ***\n\n")


# LOAD PREVIOUSLY TRAINED AND SAVED MODEL
def load_model(model, optimizer, args, epoch):
    modelname = args.model
    filename = f'saved_models/{args.network}/{modelname}_{epoch}_wt{args.window_type}_ws{args.window_size}_tk{args.top_k}_th{args.threshold}_bk{args.bottom_k}_md{args.min_deg}_clw{args.cl_weight}_clt{args.tau}.pth.tar'
    try:
        checkpoint = torch.load(filename, weights_only=False)
    except:
        checkpoint = torch.load(filename)
    
    print(f"Loading saved embeddings and model: {filename}")
    args.start_epoch = checkpoint['epoch']
    user_embeddings = Variable(torch.from_numpy(checkpoint['user_embeddings']).cuda())
    item_embeddings = Variable(torch.from_numpy(checkpoint['item_embeddings']).cuda())
    try:
        train_end_idx = checkpoint['train_end_idx'] 
    except KeyError:
        train_end_idx = None

    try:
        user_embeddings_time_series = Variable(torch.from_numpy(checkpoint['user_embeddings_time_series']).cuda())
        item_embeddings_time_series = Variable(torch.from_numpy(checkpoint['item_embeddings_time_series']).cuda())
    except:
        user_embeddings_time_series = None
        item_embeddings_time_series = None

    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])

    return [model, optimizer, user_embeddings, item_embeddings, user_embeddings_time_series, item_embeddings_time_series, train_end_idx]


# SET USER AND ITEM EMBEDDINGS TO THE END OF THE TRAINING PERIOD 
def set_embeddings_training_end(user_embeddings, item_embeddings, user_embeddings_time_series, item_embeddings_time_series, user_data_id, item_data_id, train_end_idx):
    userid2lastidx = {}
    for cnt, userid in enumerate(user_data_id[:train_end_idx]):
        userid2lastidx[userid] = cnt
    itemid2lastidx = {}
    for cnt, itemid in enumerate(item_data_id[:train_end_idx]):
        itemid2lastidx[itemid] = cnt

    try:
        embedding_dim = user_embeddings_time_series.size(1)
    except:
        embedding_dim = user_embeddings_time_series.shape[1]
    for userid in userid2lastidx:
        user_embeddings[userid, :embedding_dim] = user_embeddings_time_series[userid2lastidx[userid]]
    for itemid in itemid2lastidx:
        item_embeddings[itemid, :embedding_dim] = item_embeddings_time_series[itemid2lastidx[itemid]]

    user_embeddings.detach_()
    item_embeddings.detach_()


# SELECT THE GPU WITH MOST FREE MEMORY TO SCHEDULE JOB 
def select_free_gpu():
    mem = []
    gpus = list(set(range(torch.cuda.device_count()))) # list(set(X)) is done to shuffle the array
    for i in gpus:
        gpu_stats = gpustat.GPUStatCollection.new_query()
        mem.append(gpu_stats.jsonify()["gpus"][i]["memory.used"])
    return str(gpus[np.argmin(mem)])

def compute_metrics(ranks, relevances=None, k_list=[1, 5, 10]):
    metrics = []
    ranks = np.array(ranks)

    # If no relevance scores are provided, assume binary relevance (1 for correct item)
    if relevances is None:
        relevances = np.ones_like(ranks)

    # MRR
    metrics.append(np.mean(1.0 / ranks))

    # Recall @k
    for k in k_list:
        metrics.append(np.mean(ranks <= k))

    # NDCG @k
    for k in k_list:
        ndcgs = []
        for rank, rel in zip(ranks, relevances):
            if rank <= k:
                dcg = rel / np.log2(rank + 1)
            else:
                dcg = 0.0
            idcg = rel / np.log2(1 + 1) if rel > 0 else 1e-10
            ndcg = dcg / idcg
            ndcgs.append(ndcg)
        metrics.append(np.mean(ndcgs))

    return metrics

def eval_model(args, model, user_embeddings, item_embeddings, user_embeddings_static, item_embeddings_static, 
              start_idx, end_idx,
              
              num_users, num_items, user_sequence_id, item_sequence_id, feature_sequence,
              user_timediffs_sequence, item_timediffs_sequence, timestamp_sequence, user_previous_itemid_sequence):
    
    # Performance record
    ranks = []
    model.eval()
    with torch.no_grad():
        for j in trange(start_idx, end_idx, ncols = 50):
            # LOAD INTERACTION J
            userid = user_sequence_id[j]
            itemid = item_sequence_id[j]
            feature = feature_sequence[j]
            user_timediff = user_timediffs_sequence[j]
            item_timediff = item_timediffs_sequence[j]
            timestamp = timestamp_sequence[j]
            itemid_previous = user_previous_itemid_sequence[j]

            # LOAD USER AND ITEM EMBEDDING
            user_embedding_input = user_embeddings[torch.LongTensor([userid]).cuda()]
            user_embedding_static_input = user_embeddings_static[torch.LongTensor([userid]).cuda()]
            item_embedding_input = item_embeddings[torch.LongTensor([itemid]).cuda()]
            item_embedding_static_input = item_embeddings_static[torch.LongTensor([itemid]).cuda()]
            feature_tensor = torch.Tensor(feature).cuda().unsqueeze(0)
            user_timediffs_tensor = torch.Tensor([user_timediff]).cuda().unsqueeze(0)
            item_timediffs_tensor = torch.Tensor([item_timediff]).cuda().unsqueeze(0)
            item_embedding_previous = item_embeddings[torch.LongTensor([itemid_previous]).cuda()]

            # PROJECT USER EMBEDDING
            user_projected_embedding = model.forward(user_embedding_input, item_embedding_previous, timediffs=user_timediffs_tensor, features=feature_tensor, select='project')
            user_item_embedding = torch.cat([user_projected_embedding, item_embedding_previous, item_embeddings_static[torch.LongTensor([itemid_previous]).cuda()], user_embedding_static_input], dim=1)

            # PREDICT ITEM EMBEDDING
            predicted_item_embedding = model.predict_item_embedding(user_item_embedding)

            # CALCULATE DISTANCE OF PREDICTED ITEM EMBEDDING TO ALL ITEMS
            euclidean_distances = nn.PairwiseDistance()(predicted_item_embedding.repeat(num_items, 1), torch.cat([item_embeddings, item_embeddings_static], dim=1)).squeeze(-1)

            # CALCULATE RANK OF THE TRUE ITEM AMONG ALL ITEMS
            true_item_distance = euclidean_distances[itemid]
            euclidean_distances_smaller = (euclidean_distances < true_item_distance).data.cpu().numpy()
            true_item_rank = np.sum(euclidean_distances_smaller) + 1

            ranks.append(true_item_rank)

            user_embedding_output = model.forward(user_embedding_input, item_embedding_input,
                                                timediffs=user_timediffs_tensor, features=feature_tensor,
                                                select='user_update')
            item_embedding_output = model.forward(user_embedding_input, item_embedding_input,
                                                timediffs=item_timediffs_tensor, features=feature_tensor,
                                                select='item_update')

            # SAVE EMBEDDINGS
            item_embeddings[itemid,:] = item_embedding_output.squeeze(0)
            user_embeddings[userid,:] = user_embedding_output.squeeze(0)

            item_embeddings.detach_()
            user_embeddings.detach_()

    return compute_metrics(ranks)

# Padding functions for ragged arrays (U-U and I-I view)
def pad_2d_int(ragged, pad_val=-1):
    B = len(ragged)
    L = max((len(x) for x in ragged), default=0)
    out = np.full((B, L), pad_val, dtype=np.int64)
    mask = np.zeros((B, L), dtype=np.bool_)
    for i, arr in enumerate(ragged):
        if len(arr) == 0:
            continue
        out[i, :len(arr)] = arr
        mask[i, :len(arr)] = True
    return out, mask

def pad_2d_float(ragged, pad_val=0.0):
    B = len(ragged)
    L = max((len(x) for x in ragged), default=0)
    out = np.full((B, L), pad_val, dtype=np.float32)
    mask = np.zeros((B, L), dtype=np.bool_)
    for i, arr in enumerate(ragged):
        if len(arr) == 0:
            continue
        out[i, :len(arr)] = arr
        mask[i, :len(arr)] = True
    return out, mask