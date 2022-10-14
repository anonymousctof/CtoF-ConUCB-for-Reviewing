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
    return spp.csr_matrix((data, (rows, cols))), category_list


if __name__ == '__main__':
    # target_prefix = 'ml_25m'
    seed = 12345
    sig = "0608"
    shuffle = False
    if shuffle == True:
        target_prefix_p = f'ml_shuff_u_{sig}'
    else:
        target_prefix_p = f'ml_{sig}'

    user_num = 2000
    train_num = 500
    val_num = 500
    test_num = 500
    assert user_num == train_num + val_num + test_num
    business_num = 25000
    d = 50
    target_prefix = f'{target_prefix_p}_un{user_num}_in{business_num}'

    if not os.path.exists(target_prefix):
        os.mkdir(target_prefix)

    tmp_file_M_cat_save = f'{sig}_M_cat.npy'

    if not os.path.exists(tmp_file_M_cat_save):
        M, category_list = load_sparse_matrix()

        to_save = (M, category_list)
        # M_cat_top = [reduced_matrix, category_list, top_business]
        np.save(tmp_file_M_cat_save, to_save)
    else:
        M, category_list = np.load(tmp_file_M_cat_save)

    print("category_list", category_list)
    print("len(category_list)", len(category_list))
    exit(1)
    reduced_matrix, top_business = get_review_matrix(user_num, business_num, M)
    np.random.seed(seed)
    userMapList = list(range(user_num))
    np.random.shuffle(userMapList)

    reduced_matrix_train = reduced_matrix[:train_num]
    reduced_matrix_val = reduced_matrix[train_num:train_num+val_num]
    reduced_matrix_test = reduced_matrix[train_num + val_num:train_num + val_num + test_num]

    print("reduced_matrix_train", reduced_matrix_train.shape)
    print("reduced_matrix_val", reduced_matrix_val.shape)
    print("reduced_matrix_test", reduced_matrix_test.shape)
    U_train, V_train = extract_features(train_num, business_num, d, reduced_matrix_train)
    U_val, V_val = extract_features(val_num, business_num, d, reduced_matrix_val)
    U_test, V_test = extract_features(test_num, business_num, d, reduced_matrix_test)

    print("U_train.shape, V_train.shape", U_train.shape, V_train.shape)
    print("U_val.shape, V_val.shape", U_val.shape, V_val.shape)
    print("U_test.shape, V_test.shape", U_test.shape, V_test.shape)

    pprefix = ['train', 'validate', 'test', 'train_item_user_test_rate']
    for i in range(len(pprefix)):
        if not os.path.exists(target_prefix + '/' + pprefix[i]):
            os.mkdir(target_prefix + '/' + pprefix[i])
    generate_info(U_train, V_train, train_num, top_business, category_list, target_prefix + "/train")
    np.save(target_prefix + "/train" + '/user_item.npy', reduced_matrix_train)
    generate_info(U_val, V_val, val_num, top_business, category_list, target_prefix + "/validate")
    np.save(target_prefix + "/validate" + '/user_item.npy', reduced_matrix_val)
    generate_info(U_test, V_test, test_num, top_business, category_list, target_prefix + "/test")
    np.save(target_prefix + "/test" + '/user_item.npy', reduced_matrix_test)

    # os.makedirs(target_prefix + '/train_item_user_test_rate', exist_ok=True)
    generate_info(U_train, V_train, train_num, top_business, category_list,
                  target_prefix + '/train_item_user_test_rate')
    np.save(target_prefix + '/train_item_user_test_rate/user_item.npy', reduced_matrix_test)
