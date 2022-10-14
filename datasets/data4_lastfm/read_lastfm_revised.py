import scipy.sparse as spp
import numpy as np
import pandas as pd
import os
from ExtractFeatures import extract_features, get_review_matrix
from GenerateInfo import generate_info
from tqdm import tqdm


def load_sparse_matrix(source_prefix='../source/hetrec2011-lastfm-2k'):
    userID = {}
    artistID = {}
    tagID = {}
    # generate sparse matrix
    rows = []
    cols = []
    data = []

    category_list = {}
    attributes = {}
    df = pd.read_csv(source_prefix + '/user_taggedartists.dat', sep='\t', header=0, index_col=None)
    for index, row in df.iterrows():
        if row['artistID'] not in artistID:
            artistID[row['artistID']] = len(artistID)
        if row['userID'] not in userID:
            userID[row['userID']] = len(userID)
        if row['tagID'] not in tagID:
            tagID[row['tagID']] = len(tagID)

        artist = artistID[row['artistID']]
        user = userID[row['userID']]
        tag = tagID[row['tagID']]
        cols.append(artist)
        rows.append(user)
        data.append(1)
        if artist not in category_list:
            category_list[artist] = {}
        if tag not in category_list[artist]:
            category_list[artist][tag] = 1
        else:
            category_list[artist][tag] += 1
        if tag not in attributes:
            attributes[tag] = []
        else:
            attributes[tag].append(artist)

    for key in attributes:
        attributes[key] = len(set(attributes[key]))
    # attributes = sorted(attributes.items(), key=lambda x: x[1], reverse=True)

    for key in category_list:
        attrs = category_list[key]
        assert len(attrs) > 0
        attrs = {i: attributes[i] for i in attrs}
        attrs = sorted(attrs.items(), key=lambda x: x[1], reverse=True)[:20]
        attrs = [k[0] for k in attrs]
        category_list[key] = attrs

    # print(len(attributes), len(set(rows)), len(set(cols)))
    return spp.csr_matrix((np.asarray(data), (np.asarray(rows), np.asarray(cols)))), category_list


if __name__ == '__main__':
    source_data = '../../source/hetrec2011-lastfm-2k'
    user_num = 1500
    business_num = 10000
    target_prefix = f"un_{user_num}-bn_{business_num}"
    d = 50

    if not os.path.exists(target_prefix):
        os.mkdir(target_prefix)

    pprefix = ['train', 'validate', 'test']
    for i in range(len(pprefix)):
        if not os.path.exists(target_prefix + '/' + pprefix[i]):
            os.mkdir(target_prefix + '/' + pprefix[i])
    M, category_list = load_sparse_matrix(source_data)
    M, top_business = get_review_matrix(user_num, business_num, M)
    userMapList = list(range(user_num))
    np.random.shuffle(userMapList)

    for i in tqdm(range(len(pprefix))):
        U, V = extract_features(user_num // 3, business_num, d,
                                   M[userMapList[i * user_num // 3:(i + 1) * user_num // 3]])
        generate_info(U, V, user_num // 3, top_business, category_list, target_prefix + '/' + pprefix[i])
        np.save(target_prefix + '/' + pprefix[i] + '/user_item.npy', M[userMapList[i * user_num // 3:(i + 1) * user_num // 3]])
