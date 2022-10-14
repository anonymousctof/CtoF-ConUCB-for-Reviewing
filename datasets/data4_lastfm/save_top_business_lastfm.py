import scipy.sparse as spp
import numpy as np
import pandas as pd
import os
from ExtractFeatures import extract_features, get_review_matrix


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
    return spp.csr_matrix((np.asarray(data), (np.asarray(rows), np.asarray(cols)))), category_list, artistID, userID


if __name__ == '__main__':
    source_data = '../../source/hetrec2011-lastfm-2k'
    user_num = 1500
    business_num = 10000
    target_prefix = f"un_{user_num}-bn_{business_num}_0616_replay"
    d = 50

    if not os.path.exists(target_prefix):
        os.mkdir(target_prefix)
    tmp_M_cat_top_itemuserdict_save = f'{target_prefix}_M_cat_top_useritemdict.npy'

    if not os.path.exists(tmp_M_cat_top_itemuserdict_save):
        M, category_list, business_dict, user_dict = load_sparse_matrix(source_data)
        M, top_business = get_review_matrix(user_num, business_num, M)
        M_cat_top = [M, category_list, top_business, business_dict, user_dict]
        np.save(tmp_M_cat_top_itemuserdict_save, M_cat_top)
    else:
        reduced_matrix, category_list, top_business, business_dict, user_dict = np.load(tmp_M_cat_top_itemuserdict_save)

    def save_raw_id_2_matrix_index(business_dict):
        raw_id_2_matrix_index = {}
        for business_dict_each in business_dict:
            raw_id_2_matrix_index[int(business_dict_each)] = business_dict[business_dict_each]
        return raw_id_2_matrix_index
    raw_id_2_matrix_index = save_raw_id_2_matrix_index(business_dict)
    raw_id_2_matrix_index_file = target_prefix + "_raw_id_2_matrix_index.npy"
    # arm_index -> matrix_index -> raw_id
    np.save(raw_id_2_matrix_index_file, raw_id_2_matrix_index)
    matrix_index_2_arm_index = {}
    for i, arm_id in enumerate(top_business):
        matrix_index_2_arm_index[arm_id] = i
    matrix_index_2_arm_index_file = target_prefix + "_matrix_index_2_arm_index.npy"
    np.save(matrix_index_2_arm_index_file, matrix_index_2_arm_index)
