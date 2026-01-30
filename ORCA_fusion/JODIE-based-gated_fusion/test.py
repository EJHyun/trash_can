from library_data import *
from library_models import *
import datetime

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
parser.add_argument('--network', required=True, help='Network name')
parser.add_argument('--model', default='jodie', help="Model name")
parser.add_argument('--gpu', default=-1, type=int, help='ID of the gpu to run on. If set to -1 (default), the GPU with most free memory will be chosen.')
parser.add_argument('--embedding_dim', default=128, type=int, help='Number of dimensions')
parser.add_argument('--train_proportion', default=0.8, type=float, help='Proportion of training interactions')

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

# Timediff embedding parameters (must match training)
parser.add_argument('--timediff_embed_dim', type=int, default=16, help='Dimension of timediff embedding')
parser.add_argument('--timediff_embed_layers', type=int, default=1, help='Number of layers in timediff embedding MLP')
parser.add_argument('--timediff_activation', type=str, default='relu', choices=['relu', 'leaky_relu'], help='Activation function for timediff embedding MLP')
parser.add_argument('--timediff_scale_method', type=str, default='standard', choices=['standard', 'log', 'log_minmax'], help='Timediff scaling method: standard (scale), log (log1p), log_minmax (log1p then 0-1 normalize)')
parser.add_argument('--timediff_separate_encoder', action='store_true', help='Use separate encoders for user and item timediff embeddings')
args = parser.parse_args()
args.datapath = f"data/{args.network}.csv"
if args.train_proportion > 0.8:
    sys.exit('Training sequence proportion cannot be greater than 0.8.')

# SET GPU
# args.gpu = select_free_gpu()
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

if not os.path.exists(f'results/{args.network}'):
    os.makedirs(f'results/{args.network}')
output_fname = f'results/{args.network}/{args.model}_best_wt{args.window_type}_ws{args.window_size}_tk{args.top_k}_th{args.threshold}_bk{args.bottom_k}_md{args.min_deg}_clw{args.cl_weight}_clt{args.tau}.txt'

# LOAD NETWORK
[user2id, user_sequence_id, user_timediffs_sequence, user_previous_itemid_sequence, \
 item2id, item_sequence_id, item_timediffs_sequence, \
 timestamp_sequence, \
 feature_sequence, \
 y_true] = load_network(args)
num_interactions = len(user_sequence_id)
num_features = len(feature_sequence[0])
num_users = len(user2id)
num_items = len(item2id) + 1
print("*** Network statistics:\n  %d users\n  %d items\n  %d interactions\n  %d/%d true labels ***\n\n" % (num_users, num_items, num_interactions, sum(y_true), len(y_true)))

# SET TRAINING, VALIDATION, TESTING, and TBATCH BOUNDARIES
train_end_idx = validation_start_idx = int(num_interactions * args.train_proportion) 
test_start_idx = int(num_interactions * (args.train_proportion+0.1))
test_end_idx = int(num_interactions * (args.train_proportion+0.2))

# SET BATCHING TIMESPAN
'''
Timespan indicates how frequently the model is run and updated. 
All interactions in one timespan are processed simultaneously. 
Longer timespans mean more interactions are processed and the training time is reduced, however it requires more GPU memory.
At the end of each timespan, the model is updated as well. So, longer timespan means less frequent model updates. 
'''
timespan = timestamp_sequence[-1] - timestamp_sequence[0]
tbatch_timespan = timespan / 500 

# INITIALIZE MODEL PARAMETERS
model = JODIE(args, num_features, num_users, num_items).cuda()
MSELoss = nn.MSELoss()

# INITIALIZE MODEL
learning_rate = 1e-3
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

# LOAD THE MODEL
model, optimizer, user_embeddings_dystat, item_embeddings_dystat, user_embeddings_timeseries, item_embeddings_timeseries, train_end_idx_training = load_model(model, optimizer, args, 'best')
if train_end_idx != train_end_idx_training:
    sys.exit('Training proportion during training and testing are different. Aborting.')

# LOAD THE EMBEDDINGS: DYNAMIC AND STATIC
item_embeddings = item_embeddings_dystat[:, :args.embedding_dim]
item_embeddings = item_embeddings.clone()
item_embedding_static = Variable(torch.eye(num_items).cuda()) # one-hot vectors for static embeddings

user_embeddings = user_embeddings_dystat[:, :args.embedding_dim]
user_embeddings = user_embeddings.clone()
user_embedding_static = Variable(torch.eye(num_users).cuda()) # one-hot vectors for static embeddings 

print("***** Evaluating the model on the test set (no t-batching) *****")
print('start time:', datetime.datetime.now())
performance_dict = dict()
performance_dict['test'] = eval_model(args, model, user_embeddings, item_embeddings, user_embedding_static, item_embedding_static, 
                                      test_start_idx, test_end_idx,
                                      num_users, num_items, user_sequence_id, item_sequence_id, feature_sequence,
                                      user_timediffs_sequence, item_timediffs_sequence, timestamp_sequence, user_previous_itemid_sequence)

print('end time:', datetime.datetime.now())
# PRINT AND SAVE THE PERFORMANCE METRICS
fw = open(output_fname, "a")
metrics = ['Mean Reciprocal Rank', 'Recall@1', 'Recall@5', 'Recall@10', 'NDCG@1', 'NDCG@5', 'NDCG@10']

print('\n\n*** Test performance of best epoch ***')
fw.write('\n\n*** Test performance of best epoch ***\n')
for i in range(len(metrics)):
    print(metrics[i] + ': ' + str(performance_dict['test'][i]))
    fw.write("Test: " + metrics[i] + ': ' + str(performance_dict['test'][i]) + "\n")

fw.flush()
fw.close()