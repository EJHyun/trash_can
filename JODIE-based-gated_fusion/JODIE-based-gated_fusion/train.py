'''
This code trains the JODIE model for the given dataset. 
The task is: interaction prediction.

How to run: 
$ python jodie.py --network reddit --model jodie --epochs 50

Paper: Predicting Dynamic Embedding Trajectory in Temporal Interaction Networks. S. Kumar, X. Zhang, J. Leskovec. ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD), 2019. 
'''

import time
from datetime import datetime

from library_data import *
import library_models as lib
from library_models import *
import wandb

# Set Seed
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # GPU 시드 고정
    torch.cuda.manual_seed_all(seed)  # 다중 GPU 시드 고정
    torch.backends.cudnn.deterministic = True  # 연산을 결정적으로
    torch.backends.cudnn.benchmark = False  # 성능을 희생하더라도 결과 재현 가능하게
    os.environ['PYTHONHASHSEED'] = str(seed)

set_seed(42)

# INITIALIZE PARAMETERS
parser = argparse.ArgumentParser()
parser.add_argument('--network', required=True, help='Name of the network/dataset')
parser.add_argument('--model', default="JODIE", help='Model name to save output in file')
parser.add_argument('--gpu', default=-1, type=int, help='ID of the gpu to run on. If set to -1 (default), the GPU with most free memory will be chosen.')
parser.add_argument('--epochs', default=50, type=int, help='Number of epochs to train the model')
parser.add_argument('--embedding_dim', default=128, type=int, help='Number of dimensions of the dynamic embedding')
parser.add_argument('--train_proportion', default=0.8, type=float, help='Fraction of interactions (from the beginning) that are used for training.The next 10% are used for validation and the next 10% for testing')

# precompute parameters
parser.add_argument('--window_type', type=str, default='local', choices=['local', 'global', 'fusion'], help='window type: local (sliding), global (cumulative), or fusion (local+global gated)')
parser.add_argument('--window_size', type=float, default=0.05, help='time window size')
parser.add_argument('--top_k', type=int, default=5, help='number of similar neighbors to consider')
parser.add_argument('--threshold', type=float, default=0.8, help='similarity threshold for positive neighbors')
parser.add_argument('--bottom_k', type=int, default=5, help='number of negative neighbors to consider')
parser.add_argument('--min_deg', type=int, default=2, help='minimum degree for precomputing negative neighbors')

# CL parameters
parser.add_argument('--cl_weight', type=float, default=1, help='weight for contrastive learning loss')
parser.add_argument('--tau', type=float, default=0.2, help='temperature parameter for contrastive learning loss')

# Timediff embedding parameters
parser.add_argument('--timediff_embed_dim', type=int, default=16, help='Dimension of timediff embedding')
parser.add_argument('--timediff_embed_layers', type=int, default=1, help='Number of layers in timediff embedding MLP')
parser.add_argument('--timediff_activation', type=str, default='relu', choices=['relu', 'leaky_relu'], help='Activation function for timediff embedding MLP')
parser.add_argument('--timediff_scale_method', type=str, default='standard', choices=['standard', 'log', 'log_minmax'], help='Timediff scaling method: standard (scale), log (log1p), log_minmax (log1p then 0-1 normalize)')
parser.add_argument('--timediff_separate_encoder', action='store_true', help='Use separate encoders for user and item timediff embeddings')
args = parser.parse_args()
print(args)

# Initialize Weights & Biases
encoder_type = "sep_enc" if args.timediff_separate_encoder else "shared_enc"
wandb_run_name = f"{args.network}__scale={args.timediff_scale_method}__layers={args.timediff_embed_layers}__act={args.timediff_activation}__enc={encoder_type}__wt{args.window_type}__ws{args.window_size}__tk{args.top_k}__th{args.threshold}__bk{args.bottom_k}__md{args.min_deg}__clw{args.cl_weight}__clt{args.tau}"

wandb.init(
    project="ORCA",
    name=wandb_run_name,
    config={
        'dataset': args.network,
        'model': args.model,
        'timediff_embed_dim': args.timediff_embed_dim,
        'timediff_embed_layers': args.timediff_embed_layers,
        'timediff_activation': args.timediff_activation,
        'timediff_scale_method': args.timediff_scale_method,
        'timediff_separate_encoder': args.timediff_separate_encoder,
        'window_type': args.window_type,
        'window_size': args.window_size,
        'top_k': args.top_k,
        'threshold': args.threshold,
        'bottom_k': args.bottom_k,
        'min_deg': args.min_deg,
        'cl_weight': args.cl_weight,
        'tau': args.tau,
    },
    tags=[args.network, args.model, f"scale_{args.timediff_scale_method}", f"act_{args.timediff_activation}"]
)

# Early stopping parameters
patience_limit = 20
patience_counter = 0
best_val_mrr = float('-inf')

# SCALE FACTOR (make the scale of the CL loss similar to the prediction loss)
scale_factor = 0.001
if args.network == 'wikipedia':
    scale_factor = 0.001
elif args.network == 'lastfm':
    scale_factor = 0.001
elif args.network == 'douban_movie':
    scale_factor = 0.001
elif args.network == 'yoochoose':
    scale_factor = 0.001
elif args.network == 'amazon_video':
    scale_factor = 0.001

args.datapath = "data/%s.csv" % args.network
if args.train_proportion > 0.8:
    sys.exit('Training sequence proportion cannot be greater than 0.8.')

# SET GPU
if args.gpu == -1:
    args.gpu = select_free_gpu()
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

# LOAD DATA
[user2id, user_sequence_id, user_timediffs_sequence, user_previous_itemid_sequence,
 item2id, item_sequence_id, item_timediffs_sequence, 
 timestamp_sequence, feature_sequence, y_true] = load_network(args)
num_interactions = len(user_sequence_id)
num_users = len(user2id) 
num_items = len(item2id) + 1 # one extra item for "none-of-these"
num_features = len(feature_sequence[0])
print("*** Network statistics:\n  %d users\n  %d items\n  %d interactions\n  %d/%d true labels ***\n\n" % (num_users, num_items, num_interactions, sum(y_true), len(y_true)))

# Load neighbors if using precomputed similar neighbors
def _load_precomputed(window_type: str):
    path = f'precomputed_neighbors/{args.network}/wt{window_type}__ws{args.window_size}__tk{args.top_k}__th{args.threshold}__bk{args.bottom_k}__md{args.min_deg}.npz'
    try:
        data = np.load(path, allow_pickle=True)
    except FileNotFoundError:
        sys.exit(f'Precomputed similar neighbors not found. Please run "precompute_sim_neighbors.py" with the same parameters first to generate the precomputed similar neighbors at {path}.')
    return data, path

if args.window_type == 'fusion':
    sim_data_local, precomputed_path_local = _load_precomputed('local')
    sim_data_global, precomputed_path_global = _load_precomputed('global')

    uu_pos_neighbors_local = sim_data_local['uu_pos_neighbors']
    uu_pos_sims_local = sim_data_local['uu_pos_sims']
    uu_pos_neighbors_global = sim_data_global['uu_pos_neighbors']
    uu_pos_sims_global = sim_data_global['uu_pos_sims']
    uu_neg_neighbors_global = sim_data_global['uu_neg_neighbors']

    ii_pos_neighbors_local = sim_data_local['ii_pos_neighbors']
    ii_pos_sims_local = sim_data_local['ii_pos_sims']
    ii_pos_neighbors_global = sim_data_global['ii_pos_neighbors']
    ii_pos_sims_global = sim_data_global['ii_pos_sims']
    ii_neg_neighbors_global = sim_data_global['ii_neg_neighbors']
else:
    sim_data, precomputed_path = _load_precomputed(args.window_type)
    uu_pos_neighbors = sim_data['uu_pos_neighbors']
    uu_pos_sims = sim_data['uu_pos_sims']
    uu_neg_neighbors = sim_data['uu_neg_neighbors']
    ii_pos_neighbors = sim_data['ii_pos_neighbors']
    ii_pos_sims = sim_data['ii_pos_sims']
    ii_neg_neighbors = sim_data['ii_neg_neighbors']

# SET TRAINING, VALIDATION, TESTING, and TBATCH BOUNDARIES
train_end_idx = validation_start_idx = int(num_interactions * args.train_proportion) 
test_start_idx = int(num_interactions * (args.train_proportion+0.1))
test_end_idx = int(num_interactions * (args.train_proportion+0.2))

# SET BATCHING TIMESPAN
'''
Timespan is the frequency at which the batches are created and the JODIE model is trained. 
As the data arrives in a temporal order, the interactions within a timespan are added into batches (using the T-batch algorithm). 
The batches are then used to train JODIE. 
Longer timespans mean more interactions are processed and the training time is reduced, however it requires more GPU memory.
Longer timespan leads to less frequent model updates. 
'''
timespan = timestamp_sequence[-1] - timestamp_sequence[0]
tbatch_timespan = timespan / 500 

# INITIALIZE MODEL AND PARAMETERS
model = JODIE(args, num_features, num_users, num_items).cuda()
MSELoss = nn.MSELoss()

# INITIALIZE EMBEDDING
initial_user_embedding = nn.Parameter(F.normalize(torch.rand(args.embedding_dim).cuda(), dim=0)) # the initial user and item embeddings are learned during training as well
initial_item_embedding = nn.Parameter(F.normalize(torch.rand(args.embedding_dim).cuda(), dim=0))
model.initial_user_embedding = initial_user_embedding
model.initial_item_embedding = initial_item_embedding

user_embeddings = initial_user_embedding.repeat(num_users, 1) # initialize all users to the same embedding 
item_embeddings = initial_item_embedding.repeat(num_items, 1) # initialize all items to the same embedding
item_embedding_static = Variable(torch.eye(num_items).cuda()) # one-hot vectors for static embeddings
user_embedding_static = Variable(torch.eye(num_users).cuda()) # one-hot vectors for static embeddings

# U-U / I-I view initial states
uu_state = initial_user_embedding.repeat(num_users, 1)
ii_state = initial_item_embedding.repeat(num_items, 1) 

# INITIALIZE MODEL
learning_rate = 1e-3
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

# RUN THE JODIE MODEL
'''
THE MODEL IS TRAINED FOR SEVERAL EPOCHS. IN EACH EPOCH, JODIES USES THE TRAINING SET OF INTERACTIONS TO UPDATE ITS PARAMETERS.
'''
print("*** Training the JODIE model for %d epochs ***" % args.epochs)
# variables to help using tbatch cache between epochs
is_first_epoch = True
cached_tbatches_user = {}
cached_tbatches_item = {}
cached_tbatches_interactionids = {}
cached_tbatches_feature = {}
cached_tbatches_user_timediffs = {}
cached_tbatches_item_timediffs = {}
cached_tbatches_previous_item = {}

with trange(args.epochs, ncols = 50) as progress_bar1:
    for ep in progress_bar1:
        progress_bar1.set_description('Epoch %d of %d' % (ep, args.epochs))

        epoch_start_time = time.time()

        model.train()
        optimizer.zero_grad()
        reinitialize_tbatches()
        total_loss = 0
        total_prediction_loss = 0
        total_smoothness_loss_user = 0
        total_smoothness_loss_item = 0
        total_cl_loss_uu = 0
        total_cl_loss_ii = 0
        loss = 0
        total_interaction_count = 0

        tbatch_start_time = None
        tbatch_to_insert = -1
        tbatch_full = False

        # TRAIN TILL THE END OF TRAINING INTERACTION IDX
        for j in trange(train_end_idx, ncols = 50):
            if is_first_epoch:
                # READ INTERACTION J
                userid = user_sequence_id[j]
                itemid = item_sequence_id[j]
                feature = feature_sequence[j]
                user_timediff = user_timediffs_sequence[j]
                item_timediff = item_timediffs_sequence[j]

                # CREATE T-BATCHES: ADD INTERACTION J TO THE CORRECT T-BATCH
                tbatch_to_insert = max(lib.tbatchid_user[userid], lib.tbatchid_item[itemid]) + 1
                lib.tbatchid_user[userid] = tbatch_to_insert
                lib.tbatchid_item[itemid] = tbatch_to_insert

                lib.current_tbatches_user[tbatch_to_insert].append(userid)
                lib.current_tbatches_item[tbatch_to_insert].append(itemid)
                lib.current_tbatches_feature[tbatch_to_insert].append(feature)
                lib.current_tbatches_interactionids[tbatch_to_insert].append(j)
                lib.current_tbatches_user_timediffs[tbatch_to_insert].append(user_timediff)
                lib.current_tbatches_item_timediffs[tbatch_to_insert].append(item_timediff)
                lib.current_tbatches_previous_item[tbatch_to_insert].append(user_previous_itemid_sequence[j])

            timestamp = timestamp_sequence[j]
            if tbatch_start_time is None:
                tbatch_start_time = timestamp

            # AFTER ALL INTERACTIONS IN THE TIMESPAN ARE CONVERTED TO T-BATCHES, FORWARD PASS TO CREATE EMBEDDING TRAJECTORIES AND CALCULATE PREDICTION LOSS
            if (timestamp - tbatch_start_time > tbatch_timespan) or (j == train_end_idx - 1): # if the next interaction is outside the timespan, or if we are at the last interaction
                tbatch_start_time = timestamp # RESET START TIME FOR THE NEXT TBATCHES

                # ITERATE OVER ALL T-BATCHES
                if not is_first_epoch:
                    lib.current_tbatches_user = cached_tbatches_user[timestamp]
                    lib.current_tbatches_item = cached_tbatches_item[timestamp]
                    lib.current_tbatches_interactionids = cached_tbatches_interactionids[timestamp]
                    lib.current_tbatches_feature = cached_tbatches_feature[timestamp]
                    lib.current_tbatches_user_timediffs = cached_tbatches_user_timediffs[timestamp]
                    lib.current_tbatches_item_timediffs = cached_tbatches_item_timediffs[timestamp]
                    lib.current_tbatches_previous_item = cached_tbatches_previous_item[timestamp]


                for i in range(len(lib.current_tbatches_user)):
                    total_interaction_count += len(lib.current_tbatches_interactionids[i])

                    # LOAD THE CURRENT TBATCH
                    if is_first_epoch:
                        lib.current_tbatches_user[i] = torch.LongTensor(lib.current_tbatches_user[i]).cuda()
                        lib.current_tbatches_item[i] = torch.LongTensor(lib.current_tbatches_item[i]).cuda()
                        lib.current_tbatches_interactionids[i] = torch.LongTensor(lib.current_tbatches_interactionids[i]).cuda()
                        lib.current_tbatches_feature[i] = torch.Tensor(lib.current_tbatches_feature[i]).cuda()

                        lib.current_tbatches_user_timediffs[i] = torch.Tensor(lib.current_tbatches_user_timediffs[i]).cuda()
                        lib.current_tbatches_item_timediffs[i] = torch.Tensor(lib.current_tbatches_item_timediffs[i]).cuda()
                        lib.current_tbatches_previous_item[i] = torch.LongTensor(lib.current_tbatches_previous_item[i]).cuda()

                    tbatch_userids = lib.current_tbatches_user[i] # Recall "lib.current_tbatches_user[i]" has unique elements
                    tbatch_itemids = lib.current_tbatches_item[i] # Recall "lib.current_tbatches_item[i]" has unique elements
                    tbatch_interactionids = lib.current_tbatches_interactionids[i]
                    feature_tensor = Variable(lib.current_tbatches_feature[i]) # Recall "lib.current_tbatches_feature[i]" is list of list, so "feature_tensor" is a 2-d tensor
                    user_timediffs_tensor = Variable(lib.current_tbatches_user_timediffs[i]).unsqueeze(1)
                    item_timediffs_tensor = Variable(lib.current_tbatches_item_timediffs[i]).unsqueeze(1)
                    tbatch_itemids_previous = lib.current_tbatches_previous_item[i]
                    item_embedding_previous = item_embeddings[tbatch_itemids_previous,:]

                    # PROJECT USER EMBEDDING TO CURRENT TIME
                    user_embedding_input = user_embeddings[tbatch_userids,:]
                    user_projected_embedding = model.forward(user_embedding_input, item_embedding_previous, timediffs=user_timediffs_tensor, features=feature_tensor, select='project')
                    user_item_embedding = torch.cat([user_projected_embedding, item_embedding_previous, item_embedding_static[tbatch_itemids_previous,:], user_embedding_static[tbatch_userids,:]], dim=1)

                    # PREDICT NEXT ITEM EMBEDDING                            
                    predicted_item_embedding = model.predict_item_embedding(user_item_embedding)

                    # CALCULATE PREDICTION LOSS
                    item_embedding_input = item_embeddings[tbatch_itemids,:]
                    prediction_loss = MSELoss(predicted_item_embedding, torch.cat([item_embedding_input, item_embedding_static[tbatch_itemids,:]], dim=1).detach())
                    loss += prediction_loss
                    total_prediction_loss += prediction_loss.item()

                    # UPDATE DYNAMIC EMBEDDINGS AFTER INTERACTION
                    user_embedding_output = model.forward(user_embedding_input, item_embedding_input, timediffs=user_timediffs_tensor, features=feature_tensor, select='user_update')
                    item_embedding_output = model.forward(user_embedding_input, item_embedding_input, timediffs=item_timediffs_tensor, features=feature_tensor, select='item_update')

                    # U-U / I-I View Contrastive Learning
                    idx_list = lib.current_tbatches_interactionids[i]
                    B = len(idx_list)

                    # -----------------------
                    # U-U CL
                    # -----------------------
                    if args.window_type == 'fusion':
                        uu_pos_local_list   = [uu_pos_neighbors_local[idx]   for idx in idx_list]
                        uu_pos_local_s_list = [uu_pos_sims_local[idx]        for idx in idx_list]
                        uu_pos_global_list  = [uu_pos_neighbors_global[idx]  for idx in idx_list]
                        uu_pos_global_s_list= [uu_pos_sims_global[idx]       for idx in idx_list]
                        uu_neg_global_list  = [uu_neg_neighbors_global[idx]  for idx in idx_list]

                        uu_valid = [((len(pl) > 0 or len(pg) > 0) and len(ng) > 0) for pl, pg, ng in zip(uu_pos_local_list, uu_pos_global_list, uu_neg_global_list)]
                        if any(uu_valid):
                            keep = [k for k, ok in enumerate(uu_valid) if ok]

                            q_user = user_embedding_output[keep, :]
                            uu_prev = uu_state[tbatch_userids[keep], :]

                            pos_local_ids_pad, pos_local_mask = pad_2d_int([uu_pos_local_list[k] for k in keep], pad_val=-1)
                            pos_local_sims_pad, _ = pad_2d_float([uu_pos_local_s_list[k] for k in keep], pad_val=0.0)
                            pos_global_ids_pad, pos_global_mask = pad_2d_int([uu_pos_global_list[k] for k in keep], pad_val=-1)
                            pos_global_sims_pad, _ = pad_2d_float([uu_pos_global_s_list[k] for k in keep], pad_val=0.0)

                            pos_local_ids_t   = torch.LongTensor(pos_local_ids_pad).cuda()
                            pos_local_mask_t  = torch.BoolTensor(pos_local_mask).cuda()
                            pos_local_sims_t  = torch.FloatTensor(pos_local_sims_pad).cuda()
                            pos_global_ids_t  = torch.LongTensor(pos_global_ids_pad).cuda()
                            pos_global_mask_t = torch.BoolTensor(pos_global_mask).cuda()
                            pos_global_sims_t = torch.FloatTensor(pos_global_sims_pad).cuda()

                            pos_local_embs = user_embeddings[pos_local_ids_t.clamp(min=0), :].detach()
                            pos_local_embs = pos_local_embs * pos_local_mask_t.unsqueeze(-1).float()
                            pos_global_embs = user_embeddings[pos_global_ids_t.clamp(min=0), :].detach()
                            pos_global_embs = pos_global_embs * pos_global_mask_t.unsqueeze(-1).float()

                            uu_out = model.encode_sim_view_fused(
                                uu_prev,
                                pos_local_embs, pos_local_sims_t, pos_local_mask_t,
                                pos_global_embs, pos_global_sims_t, pos_global_mask_t,
                                timediffs=user_timediffs_tensor[keep], view='uu'
                            )

                            neg_ids_pad, neg_mask = pad_2d_int([uu_neg_global_list[k] for k in keep], pad_val=-1)
                            neg_ids_t  = torch.LongTensor(neg_ids_pad).cuda()
                            neg_mask_t = torch.BoolTensor(neg_mask).cuda()

                            neg_keys = user_embeddings[neg_ids_t.clamp(min=0), :].detach()
                            neg_keys = neg_keys * neg_mask_t.unsqueeze(-1).float()

                            cl_loss_uu = model.infonce(q_user, uu_out, neg_keys, neg_mask_t, tau=args.tau)
                            total_cl_loss_uu += (args.cl_weight * scale_factor) * cl_loss_uu.item()
                            loss += (args.cl_weight * scale_factor) * cl_loss_uu

                            uu_state[tbatch_userids[keep]] = uu_out
                    else:
                        uu_pos_ids_list  = [uu_pos_neighbors[idx] for idx in idx_list]
                        uu_pos_sims_list = [uu_pos_sims[idx]      for idx in idx_list]
                        uu_neg_ids_list  = [uu_neg_neighbors[idx] for idx in idx_list]

                        uu_valid = [(len(p) > 0 and len(n) > 0) for p, n in zip(uu_pos_ids_list, uu_neg_ids_list)]
                        if any(uu_valid):
                            keep = [k for k, ok in enumerate(uu_valid) if ok]

                            q_user = user_embedding_output[keep, :]
                            uu_prev = uu_state[tbatch_userids[keep], :]

                            pos_ids_pad, pos_mask = pad_2d_int([uu_pos_ids_list[k] for k in keep], pad_val=-1)
                            pos_sims_pad, _ = pad_2d_float([uu_pos_sims_list[k] for k in keep], pad_val=0.0)

                            pos_ids_t  = torch.LongTensor(pos_ids_pad).cuda()
                            pos_mask_t = torch.BoolTensor(pos_mask).cuda()
                            pos_sims_t = torch.FloatTensor(pos_sims_pad).cuda()

                            pos_embs = user_embeddings[pos_ids_t.clamp(min=0), :].detach()
                            pos_embs = pos_embs * pos_mask_t.unsqueeze(-1).float()

                            uu_out = model.encode_sim_view(uu_prev, pos_embs, pos_sims_t, pos_mask_t, timediffs=user_timediffs_tensor[keep], view='uu')

                            neg_ids_pad, neg_mask = pad_2d_int([uu_neg_ids_list[k] for k in keep], pad_val=-1)
                            neg_ids_t  = torch.LongTensor(neg_ids_pad).cuda()
                            neg_mask_t = torch.BoolTensor(neg_mask).cuda()

                            neg_keys = user_embeddings[neg_ids_t.clamp(min=0), :].detach()
                            neg_keys = neg_keys * neg_mask_t.unsqueeze(-1).float()

                            cl_loss_uu = model.infonce(q_user, uu_out, neg_keys, neg_mask_t, tau=args.tau)
                            total_cl_loss_uu += (args.cl_weight * scale_factor) * cl_loss_uu.item()
                            loss += (args.cl_weight * scale_factor) * cl_loss_uu

                            uu_state[tbatch_userids[keep]] = uu_out

                    # -----------------------
                    # I-I CL
                    # -----------------------
                    if args.window_type == 'fusion':
                        ii_pos_local_list   = [ii_pos_neighbors_local[idx]   for idx in idx_list]
                        ii_pos_local_s_list = [ii_pos_sims_local[idx]        for idx in idx_list]
                        ii_pos_global_list  = [ii_pos_neighbors_global[idx]  for idx in idx_list]
                        ii_pos_global_s_list= [ii_pos_sims_global[idx]       for idx in idx_list]
                        ii_neg_global_list  = [ii_neg_neighbors_global[idx]  for idx in idx_list]

                        ii_valid = [((len(pl) > 0 or len(pg) > 0) and len(ng) > 0) for pl, pg, ng in zip(ii_pos_local_list, ii_pos_global_list, ii_neg_global_list)]
                        if any(ii_valid):
                            keep = [k for k, ok in enumerate(ii_valid) if ok]

                            q_item = item_embedding_output[keep, :]
                            ii_prev = ii_state[tbatch_itemids[keep], :]

                            pos_local_ids_pad, pos_local_mask = pad_2d_int([ii_pos_local_list[k] for k in keep], pad_val=-1)
                            pos_local_sims_pad, _ = pad_2d_float([ii_pos_local_s_list[k] for k in keep], pad_val=0.0)
                            pos_global_ids_pad, pos_global_mask = pad_2d_int([ii_pos_global_list[k] for k in keep], pad_val=-1)
                            pos_global_sims_pad, _ = pad_2d_float([ii_pos_global_s_list[k] for k in keep], pad_val=0.0)

                            pos_local_ids_t   = torch.LongTensor(pos_local_ids_pad).cuda()
                            pos_local_mask_t  = torch.BoolTensor(pos_local_mask).cuda()
                            pos_local_sims_t  = torch.FloatTensor(pos_local_sims_pad).cuda()
                            pos_global_ids_t  = torch.LongTensor(pos_global_ids_pad).cuda()
                            pos_global_mask_t = torch.BoolTensor(pos_global_mask).cuda()
                            pos_global_sims_t = torch.FloatTensor(pos_global_sims_pad).cuda()

                            pos_local_embs = item_embeddings[pos_local_ids_t.clamp(min=0), :].detach()
                            pos_local_embs = pos_local_embs * pos_local_mask_t.unsqueeze(-1).float()
                            pos_global_embs = item_embeddings[pos_global_ids_t.clamp(min=0), :].detach()
                            pos_global_embs = pos_global_embs * pos_global_mask_t.unsqueeze(-1).float()

                            ii_out = model.encode_sim_view_fused(
                                ii_prev,
                                pos_local_embs, pos_local_sims_t, pos_local_mask_t,
                                pos_global_embs, pos_global_sims_t, pos_global_mask_t,
                                timediffs=item_timediffs_tensor[keep], view='ii'
                            )

                            neg_ids_pad, neg_mask = pad_2d_int([ii_neg_global_list[k] for k in keep], pad_val=-1)
                            neg_ids_t  = torch.LongTensor(neg_ids_pad).cuda()
                            neg_mask_t = torch.BoolTensor(neg_mask).cuda()

                            neg_keys = item_embeddings[neg_ids_t.clamp(min=0), :].detach()
                            neg_keys = neg_keys * neg_mask_t.unsqueeze(-1).float()

                            cl_loss_ii = model.infonce(q_item, ii_out, neg_keys, neg_mask_t, tau=args.tau)
                            total_cl_loss_ii += (args.cl_weight * scale_factor) * cl_loss_ii.item()
                            loss += (args.cl_weight * scale_factor) * cl_loss_ii

                            ii_state[tbatch_itemids[keep]] = ii_out
                    else:
                        ii_pos_ids_list  = [ii_pos_neighbors[idx] for idx in idx_list]
                        ii_pos_sims_list = [ii_pos_sims[idx]      for idx in idx_list]
                        ii_neg_ids_list  = [ii_neg_neighbors[idx] for idx in idx_list]

                        ii_valid = [(len(p) > 0 and len(n) > 0) for p, n in zip(ii_pos_ids_list, ii_neg_ids_list)]
                        if any(ii_valid):
                            keep = [k for k, ok in enumerate(ii_valid) if ok]

                            q_item = item_embedding_output[keep, :]
                            ii_prev = ii_state[tbatch_itemids[keep], :]

                            pos_ids_pad, pos_mask = pad_2d_int([ii_pos_ids_list[k] for k in keep], pad_val=-1)
                            pos_sims_pad, _ = pad_2d_float([ii_pos_sims_list[k] for k in keep], pad_val=0.0)

                            pos_ids_t  = torch.LongTensor(pos_ids_pad).cuda()
                            pos_mask_t = torch.BoolTensor(pos_mask).cuda()
                            pos_sims_t = torch.FloatTensor(pos_sims_pad).cuda()

                            pos_embs = item_embeddings[pos_ids_t.clamp(min=0), :].detach()
                            pos_embs = pos_embs * pos_mask_t.unsqueeze(-1).float()

                            ii_out = model.encode_sim_view(ii_prev, pos_embs, pos_sims_t, pos_mask_t, timediffs=item_timediffs_tensor[keep], view='ii')

                            neg_ids_pad, neg_mask = pad_2d_int([ii_neg_ids_list[k] for k in keep], pad_val=-1)
                            neg_ids_t  = torch.LongTensor(neg_ids_pad).cuda()
                            neg_mask_t = torch.BoolTensor(neg_mask).cuda()

                            neg_keys = item_embeddings[neg_ids_t.clamp(min=0), :].detach()
                            neg_keys = neg_keys * neg_mask_t.unsqueeze(-1).float()

                            cl_loss_ii = model.infonce(q_item, ii_out, neg_keys, neg_mask_t, tau=args.tau)
                            total_cl_loss_ii += (args.cl_weight * scale_factor) * cl_loss_ii.item()
                            loss += (args.cl_weight * scale_factor) * cl_loss_ii

                            ii_state[tbatch_itemids[keep]] = ii_out

                    item_embeddings[tbatch_itemids,:] = item_embedding_output
                    user_embeddings[tbatch_userids,:] = user_embedding_output  

                    # CALCULATE LOSS TO MAINTAIN TEMPORAL SMOOTHNESS
                    smoothness_loss_i = MSELoss(item_embedding_output, item_embedding_input.detach())
                    smoothness_loss_u = MSELoss(user_embedding_output, user_embedding_input.detach())
                    loss += smoothness_loss_i + smoothness_loss_u
                    total_smoothness_loss_item += smoothness_loss_i.item()
                    total_smoothness_loss_user += smoothness_loss_u.item()

                # BACKPROPAGATE ERROR AFTER END OF T-BATCH
                total_loss += loss.item()
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                # RESET LOSS FOR NEXT T-BATCH
                loss = 0
                item_embeddings.detach_() # Detachment is needed to prevent double propagation of gradient
                user_embeddings.detach_()
                uu_state.detach_()
                ii_state.detach_()
                
                # REINITIALIZE
                if is_first_epoch:
                    cached_tbatches_user[timestamp] = lib.current_tbatches_user
                    cached_tbatches_item[timestamp] = lib.current_tbatches_item
                    cached_tbatches_interactionids[timestamp] = lib.current_tbatches_interactionids
                    cached_tbatches_feature[timestamp] = lib.current_tbatches_feature
                    cached_tbatches_user_timediffs[timestamp] = lib.current_tbatches_user_timediffs
                    cached_tbatches_item_timediffs[timestamp] = lib.current_tbatches_item_timediffs
                    cached_tbatches_previous_item[timestamp] = lib.current_tbatches_previous_item
                    
                    reinitialize_tbatches()
                    tbatch_to_insert = -1

        is_first_epoch = False # as first epoch ends here
        print("Last epoch took {} minutes".format((time.time()-epoch_start_time)/60))
        # END OF ONE EPOCH
        print(f"\n{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Epoch {ep} finished.")
        print("Loss breakdown:")
        print(f"  - Total Loss: {total_loss:.4f}")
        print(f"  - Prediction MSE Loss: {total_prediction_loss:.4f} ({(total_prediction_loss/total_loss)*100:.2f}%)")
        print(f"  - U-U CL Loss: {total_cl_loss_uu:.4f} ({(total_cl_loss_uu/total_loss)*100:.2f}%)")
        print(f"  - I-I CL Loss: {total_cl_loss_ii:.4f} ({(total_cl_loss_ii/total_loss)*100:.2f}%)")
        print(f"  - User Smoothness Loss: {total_smoothness_loss_user:.4f} ({(total_smoothness_loss_user/total_loss)*100:.2f}%)")
        print(f"  - Item Smoothness Loss: {total_smoothness_loss_item:.4f} ({(total_smoothness_loss_item/total_loss)*100:.2f}%)")

        

        # Validation
        print("\n***** Evaluating the model on the validation set (no t-batching) *****")
        val_mrr = eval_model(args, model, user_embeddings, item_embeddings, user_embedding_static, item_embedding_static,
                             validation_start_idx, test_start_idx,

                             num_users, num_items, user_sequence_id, item_sequence_id, feature_sequence,
                             user_timediffs_sequence, item_timediffs_sequence, timestamp_sequence, user_previous_itemid_sequence)[0]
        print("Validation MRR of %d epoch: %f" % (ep, val_mrr))

        # early stop
        improved = val_mrr > best_val_mrr
        if improved:
            best_val_mrr = val_mrr
            patience_counter = 0
            print(f"  ↳ New best! epoch={ep}, MRR={best_val_mrr:.6f}")
            # SAVE CURRENT MODEL TO DISK TO BE USED IN EVALUATION.
            save_model(model, optimizer, args, 'best', user_embeddings, item_embeddings, train_end_idx, None, None)
        else:
            patience_counter += 1
            print(f"  ↳ No improvement. Patience counter: {patience_counter}/{patience_limit}")
            if patience_counter >= patience_limit:
                print("\n***** Early stopping triggered.*****\n")
                wandb.log({})
                wandb.log({'early_stopping_epoch': ep,
                           'best_val_mrr': best_val_mrr,
                })
                break

        # RE-INITIALIZE USER AND ITEM EMBEDDINGS
        user_embeddings = initial_user_embedding.repeat(num_users, 1)
        item_embeddings = initial_item_embedding.repeat(num_items, 1)
        uu_state = initial_user_embedding.repeat(num_users, 1)
        ii_state = initial_item_embedding.repeat(num_items, 1)

        # Log to W&B
        wandb.log({
            'epoch': ep,
            'total_loss': total_loss,
            'prediction_loss': total_prediction_loss,
            'uu_cl_loss': total_cl_loss_uu,
            'ii_cl_loss': total_cl_loss_ii,
            'user_smoothness_loss': total_smoothness_loss_user,
            'item_smoothness_loss': total_smoothness_loss_item,
            'val_mrr': val_mrr,
        })

# END OF ALL EPOCHS. SAVE FINAL MODEL DISK TO BE USED IN EVALUATION.
print("\n\n*** Training complete. ***\n\n")
wandb.finish()
