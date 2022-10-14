import scipy.sparse as spp
import numpy as np
import json
import os
from ExtractFeatures import extract_features, get_review_matrix
from GenerateInfo import generate_info


def load_sparse_matrix(source_prefix='../../source/ml-25m'):
    # key: business_id, item: list of categories or None
    categories = {}
    with open(source_prefix + '/movies.csv', encoding='utf-8') as bs:
        next(bs)
        lines = bs.readlines()
        for line in lines:
            dicts = line.split(',')
            if len(dicts) == 2:
                categories[dicts[0]] = None
            else:
                items = dicts[2].split('|')
                categories[dicts[0]] = []
                for item in items:
                    categories[dicts[0]].append(item.strip())

    # get tuples for sparse matrix
    user_dict, business_dict = {}, {}
    category_list = []
    data, rows, cols = [], [], []
    with open(source_prefix + '/ratings.csv', encoding='utf-8') as rf:
        next(rf)
        lines = rf.readlines()
        for line in lines:
            dicts = line.split(',')
            user_id = dicts[0]
            business_id = dicts[1]
            rating = float(dicts[2])
            rating = 1 if rating > 3 else 0

            # key: user id, item: user index
            if user_id not in user_dict:
                user_dict[user_id] = len(user_dict)
            row_index = user_dict[user_id]

            # key: business id, item: business index
            if business_id not in business_dict:
                business_dict[business_id] = len(business_dict)
                # a list of categories in the order of business index
                category_list.append(categories[business_id])
            col_index = business_dict[business_id]

            # add tuple to list
            data.append(rating)
            rows.append(row_index)
            cols.append(col_index)

    # generate sparse matrix
    data = np.asarray(data)
    rows = np.asarray(rows)
    cols = np.asarray(cols)
    # print(len(data), len(rows), len(cols))
    # print(len(user_dict), len(business_dict), len(category_list))
    return spp.csr_matrix((data, (rows, cols))), category_list, business_dict, user_dict


if __name__ == '__main__':
    user_num = 2000
    business_num = 25000
    target_prefix = f'ml_25m_0401a1_un{user_num}_in{business_num}_{"0616_replay"}'
    d = 50

    if not os.path.exists(target_prefix):
        os.mkdir(target_prefix)
    tmp_M_cat_top_itemuserdict_save = f'{target_prefix}_M_cat_top_useritemdict.npy'

    # f"tmp_M_cat_top_save_25_un{user_num}_in{business_num}.npy"
    if not os.path.exists(tmp_M_cat_top_itemuserdict_save):
        M, category_list, business_dict, user_dict = load_sparse_matrix()
        reduced_matrix, top_business = get_review_matrix(user_num, business_num, M)
        M_cat_top = [reduced_matrix, category_list, top_business, business_dict, user_dict]
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
    if not os.path.exists(target_prefix + "_top_business.npy"):
        np.save(target_prefix+"_top_business.npy", top_business)
    else:
        top_business = np.load(target_prefix + "_top_business.npy")

    print("max top_business", max(top_business))
    matrix_index_2_arm_index = {}
    for i, arm_id in enumerate(top_business):
        matrix_index_2_arm_index[arm_id] = i
    matrix_index_2_arm_index_file = target_prefix + "_matrix_index_2_arm_index.npy"
    np.save(matrix_index_2_arm_index_file, matrix_index_2_arm_index)
