import torch
import json
import pickle
import numpy as np
import pandas as pd
import os
import torch.utils.data as Data
from pytorch_pretrained_bert import BertTokenizer, BertModel
import tqdm
sig = "0622"

nu = 500
raw_data_path = '../source/hetrec2011-lastfm-2k/'
processed_data_path = "../dataset-generate/data5_lastfm/un_1500-bn_10000/train_item_user_test_rate"

os.makedirs(processed_data_path + "/storage", exist_ok=True)
attr_save_path = processed_data_path + f"/{sig}_attr.npy"
# arm embedding

def read_arms(processed_data_path):
    arm_info = dict()
    file_name = processed_data_path + '/arm_info.txt'
    with open(file_name, 'r') as fr:
        for line in fr:
            j_s = json.loads(line)
            aid = j_s['a_id']
            fv = j_s['fv']
            arm_info[aid] = torch.tensor(fv, device='cuda').squeeze()
    return arm_info

movies = pd.read_csv(raw_data_path + '/artists.dat', header=0, index_col=0, sep='\t')['name'].to_dict()
item_names = {key: item.strip('"')  for key, item in movies.items()}  # item num to item idx  title

def load_raw_id_2_matrix_index(processed_data_path):
    raw_id_2_matrix_index = np.load(processed_data_path + "_raw_id_2_matrix_index.npy").item()
    matrix_index_2_raw_id = dict([val, key] for key, val in raw_id_2_matrix_index.items())
    return raw_id_2_matrix_index, matrix_index_2_raw_id

def load_matrix_index_2_arm_index(processed_data_path):
    matrix_index_2_arm_index = np.load(processed_data_path + "_matrix_index_2_arm_index.npy").item()
    arm_index_2_matrix_index = dict([val, key] for key, val in matrix_index_2_arm_index.items())
    return matrix_index_2_arm_index, arm_index_2_matrix_index
processed_data_path_replay = "/mnt/qzli_hdd/ConUCB/cluster_original/dataset-generate/data5_lastfm/un_1500-bn_10000_0616_replay"
raw_id_2_matrix_index, matrix_index_2_raw_id = load_raw_id_2_matrix_index(processed_data_path_replay)
matrix_index_2_arm_index, arm_index_2_matrix_index = load_matrix_index_2_arm_index(processed_data_path_replay)

def load_source(attr, attr_info, raw_data_path, processed_data_path):
    arm_info = read_arms(processed_data_path)
    movies = pd.read_csv(raw_data_path + '/artists.dat', header=0, index_col=0, sep='\t')['name'].to_dict()
    tags_name = pd.read_csv(raw_data_path + '/tags.dat', header=0, index_col=0, sep='\t', engine='python')[
        'tagValue'].to_dict()
    arm_index = {item.strip('"'): key for key, item in movies.items()}  # item name to item idx  title
    tags = pd.read_csv(raw_data_path + '/user_taggedartists-timestamps.dat', header=0, index_col=None, sep='\t', engine='python')

    for index, row in tags.iterrows():  # tag's movie
        if len(str(tags_name[row['tagID']]).split(' ')) > 1 or len(str(tags_name[row['tagID']]).split('-')) > 1:
            continue
        try:
            movie = movies[row['artistID']]  # tag movie's name str
        except KeyError:
            continue
        try:  # arm_index[movie] index
            attr_info[len(attr_info)] = arm_info[matrix_index_2_arm_index[raw_id_2_matrix_index[arm_index[movie]]]]
        except:
            continue
        attr[len(attr)] = str(tags_name[row['tagID']]).strip('"')
    with open(processed_data_path + '/semantic_attr.pkl', 'wb') as f:
        pickle.dump(attr, f)
    np.save(processed_data_path + '/storage/suparms.npy', attr_info)
    return attr, attr_info  # tag_str item

attr = {}
attr_info = {}
attr, attr_info = load_source(attr, attr_info, raw_data_path, processed_data_path)
value_text = [attr[attr_i] for attr_i in attr]
value_text_set = set(value_text)


class M(torch.nn.Module):
    def __init__(self):
        super(M, self).__init__()
        self.project = torch.nn.Sequential(
            torch.nn.Linear(768, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 50),
        )

    def forward(self, x):
        return self.project(x)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
emb = BertModel.from_pretrained('bert-base-uncased').embeddings.word_embeddings
emb.requires_grad_(False)
emb = emb.cuda()


if os.path.exists(processed_data_path + f'/model.pt'):
    m = torch.load(processed_data_path + f'/model.pt').cuda()
else:
    arm_info = read_arms(processed_data_path)
    if os.path.exists(processed_data_path + '/storage/suparms.npy') \
            and os.path.exists(processed_data_path + '/semantic_attr.pkl'):
        attr_info = np.load(processed_data_path + '/storage/suparms.npy', allow_pickle=True).item()
        with open(processed_data_path + '/semantic_attr.pkl', 'rb') as f:
            attr = pickle.load(f)
    else:
        attr, attr_info = {}, {}
        attr, attr_info = load_source(attr, attr_info, raw_data_path, processed_data_path)  # saved  attr attr_info
        print('preload success:', len(attr))  # tag, embedding
    # Load pre-trained model tokenizer (vocabulary)
    X_train, y_train, original, tokens_tensor = [], [], [], []
    for index, item in tqdm.tqdm(attr.items()):  # index item
        if index % 20000 == 19999:
            print(index, ':', item)  # eg 299999 sweet

        tokenized_text = tokenizer.tokenize(item)
        # Convert token to vocabulary indices
        indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)  # index
        # Convert inputs to PyTorch tensors
        if len(indexed_tokens) == 1 and not item.startswith('UNKNOWN') and index in attr_info:
            original.append(tokenized_text[0])  # text
            tokens_tensor.append(indexed_tokens[0])  # token
            try:
                y_train.append(torch.from_numpy(attr_info[index]).float().cuda())
            except:
                y_train.append(attr_info[index].float().view(-1).cuda())
    tokens_tensor = torch.tensor([tokens_tensor]).cuda()
    X_train = emb(tokens_tensor).squeeze()  # token to embedding
    y_train = torch.stack(y_train)
    m = M().cuda()
    opt = torch.optim.Adam(m.parameters(), lr=1e-4)
    y = y_train / y_train.pow(2).sum(-1, keepdims=True).sqrt()

    dataset = Data.TensorDataset(X_train, y)  # tag embedding, item embedding
    loader = Data.DataLoader(
        dataset=dataset,
        batch_size=102400,
        shuffle=True
    )
    for epoch in range(200):
        for step, (batch_x, batch_y) in enumerate(loader):
            X = m(batch_x)
            X = X / X.pow(2).sum(-1, keepdims=True).sqrt()  # normalization
            loss = 1. - (X * batch_y).sum(-1).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()

    torch.save(m, processed_data_path + '/model.pt')

finds, store = dict(), set()

inds = []

for file in ["genre.tsv", "instrument.tsv", "mood_theme.tsv"]:
    raw_data_path_file = f"{raw_data_path}/raw_30s_cleantags/{file}"
    df = pd.read_csv(raw_data_path_file, sep='\t')
    for index, row in df.iterrows():  # tag's movie
        word = row['tag']
        try:
            ind = tokenizer.convert_tokens_to_ids([word])[0]  # word
        except:
            pass
        inds.append(ind)

inds = torch.tensor(inds, device='cuda').long()
embeddings = m(emb(inds))
embeddings = embeddings / embeddings.pow(2).sum(-1, keepdims=True).sqrt()  # shape 109406, 50

for user_each_plot in range(500):
    word_embedding_folder = f"../../datasets/data4_lastfm/un_1500-bn_10000/train_item_user_test_rate/1008_save_theta_cluster/0/[Kmeans+del+0.1]++T_ratio0.1topk_[SKmeans+Dyn+1.0+salpha+0.25+salpha_p+1.4142135623730951+kalpha+0.01+ktalpha+0.25+lamb0.5+tlamb0.1+T_ratio0.1+is_split_over_user_num]/semantic/user_{user_each_plot}"
    word_embedding_folder = f"../../datasets/data4_lastfmun_1500-bn_10000/train_item_user_test_rate/1008_save_theta_cluster/0/[Kmeans+del+0.1]+topk_[Kmeans_user_ts_cluster+Dyn+1.0+kalpha+0.01+ktalpha+0.25+lamb0.5+tlamb0.1+user_ts_a0.01+is_split_over_user_num]/semantic/user_{user_each_plot}"
    user_id = 0
    file_name_list_ori = os.listdir(word_embedding_folder)
    file_name_list = []
    for file_name in file_name_list_ori:
        if file_name[-4:] == '.npy':
            file_name_list.append(file_name)

    file_name_list.sort(key=lambda x: (int(x.split('.')[0]), int(x.split('.')[1])) )

    for index, dirs in enumerate(file_name_list):
        if index == 0:
            f = open(word_embedding_folder + f'/results_{sig}.csv', 'w', encoding='utf-8')
            f.write('\t'.join(['time', 'index', 'attr', 'items', 'user']) + '\n')
        elif index % 10000 == 9999:
            f.close()
            f = open(word_embedding_folder + f'/results_{sig}.csv', 'a', encoding='utf-8')
            print(index, ':', dirs)
        attr = np.load(word_embedding_folder + '/' + dirs, allow_pickle=True).item()  # arm_index
        r = attr['reward']
        items = []
        for i in attr['items'][0]:
            movies_id = matrix_index_2_raw_id[arm_index_2_matrix_index[i]]
            try:
                items.append(movies[movies_id])
            except:
                print("no movie")
                continue
        if len(items) == 0: continue

        fv = torch.tensor(attr['fv']).cuda().view(-1)
        user = np.argmax(attr['reward'])
        sims = torch.mv(embeddings, fv)
        topk = torch.topk(sims, 10)
        indices = inds[topk.indices]
        results = tokenizer.convert_ids_to_tokens(indices.tolist())
        results = list(set(results))
        if len(results) < 5:
            results = results + ['' for _ in range(5 - len(results))]
        f.write('\t'.join(dirs.split('.')[:2] + results[:5]) + '\t' +
                ' ; '.join([i for i in items]) + '\t' + str(user) + f'\t{r}\n')
        print('\t'.join([str(int(dirs.split('.')[0]) // nu)] + results[:5])
         + '\t' + ' ; '.join([i for i in items]) + '\t' + str(user) + f'\t{r}\n')
    f.close()
