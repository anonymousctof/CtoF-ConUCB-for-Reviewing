import argparse
import os

import numpy as np
import pandas as pd
import scipy.sparse as spp
from tqdm import tqdm
from ExtractFeatures import extract_features, get_review_matrix
from GenerateInfo import generate_info

def load_sparse_matrix(input_folder):
    userID = {}
    artistID = {}
    tagID = {}
    rows = []
    cols = []
    data = []
    artist_to_tagcount = {}
    tag_to_artists = {}
    df0 = pd.read_csv(os.path.join(input_folder, "delicious-ut/out.delicious_delicious-ut_delicious-ut"), sep=" ",
                      header=None, index_col=None, skiprows=2)
    df1 = pd.read_csv(os.path.join(input_folder, "delicious-ti/out.delicious_delicious-ti_delicious-ti"), sep=" ",
                      header=None, index_col=None, skiprows=2)
    df = pd.merge(df0, df1, left_index=True, right_index=True)
    for _, row in tqdm(df.iterrows()):
        if row[5] not in artistID:
            artistID[row[5]] = len(artistID)  # item
        if row[0] not in userID:
            userID[row[0]] = len(userID)  # user
        if row[1] not in tagID:
            tagID[row[1]] = len(tagID)  # tag
        artist = artistID[row[5]]
        user = userID[row[0]]
        tag = tagID[row[1]]
        cols.append(artist)
        rows.append(user)
        data.append(1)
        if artist not in artist_to_tagcount:
            artist_to_tagcount[artist] = {}
        if tag not in artist_to_tagcount[artist]:
            artist_to_tagcount[artist][tag] = 1
        else:
            artist_to_tagcount[artist][tag] += 1
        if tag not in tag_to_artists:
            tag_to_artists[tag] = []
        else:
            tag_to_artists[tag].append(artist)

    # tag_to_artists  # tag 的 target unused
    tag_to_artistcount = {}
    for tag in tag_to_artists:
        tag_to_artistcount[tag] = len(set(tag_to_artists[tag]))

    # artist_to_tags artist's tag
    artist_to_tags = {}
    for artist in artist_to_tagcount:
        related_tags = artist_to_tagcount[artist]  # artist tag
        related_tag_to_artistcount = {tag: tag_to_artistcount[tag] for tag in related_tags}

        # Limit the number of tags.
        related_tag_to_artistcount = dict(sorted(related_tag_to_artistcount.items(), key=lambda x: x[1], reverse=True)[:20])
        related_tags = list(related_tag_to_artistcount)
        artist_to_tags[artist] = related_tags

    print(len(tag_to_artists), len(set(rows)), len(set(cols)))
    meta_data = {"rating_num": len(data), "user_num": len(rows), "item_num": len(cols),
                 "user_num_of_user_dict": len(userID), "item_num_of_bussiness_dict": len(artistID),
                 "key_term": len(tagID)}
    print(meta_data)
    import json
    json.dump(meta_data, open(f"{input_folder}/meta.json", 'w'))

    return spp.csr_matrix((np.asarray(data), (np.asarray(rows), np.asarray(cols)))), artist_to_tags

if __name__ == "__main__":
    seed = 12345
    sig = "0619"
    cut_value = None
    if cut_value is not None:
        sig = sig + f"cut_{cut_value}"
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", dest="input", type=str, help="Path to the input folder.", default=None)
    parser.add_argument("-o", "--output", dest="output", type=str, help="Path to the output folder.", default=None)
    d = 50

    target_prefix_p = f'shuff_u_{sig}'


    user_num = 1500

    train_num, val_num, test_num = 500, 500, 500
    business_num_list = [5000, 10000, 15000, 2000]
    # business_num_list = []
    assert train_num + val_num + test_num == user_num
    for business_num in business_num_list:
        args = parser.parse_args()
        if not args.input:
            args.input = "../../source/delicious/"
        if not args.output:
            args.output = f"./{target_prefix_p}_un_{user_num}-bn_{business_num}/"
        os.makedirs(args.output, exist_ok=True)
        print(args.input)
        print(args.output)

        # Extract data.
        tmp_file_full_matrix_artist_to_tag = "./tmp_file_full_matrix_artist_to_tag.npy"

        if not os.path.exists(tmp_file_full_matrix_artist_to_tag):
            full_matrix, artist_to_tags = load_sparse_matrix(args.input)
            to_save = (full_matrix, artist_to_tags)
            np.save(tmp_file_full_matrix_artist_to_tag, to_save)
        else:
            full_matrix, artist_to_tags = np.load(tmp_file_full_matrix_artist_to_tag)

        M, category_list = full_matrix, artist_to_tags
        print("M.shape", M.shape)  # 57944 767447
        M, top_business = get_review_matrix(user_num, business_num, M)
        print("M.shape", M.shape)

        if cut_value is not None:
            M = np.clip(M, -10, cut_value)

        # print("category_list", category_list)
        print("category_list", len(category_list))

        print("top_business", top_business)
        print("top_business", len(top_business))
        print("max_business", max(top_business))
        np.random.seed(seed)
        userMapList = list(range(user_num))
        np.random.shuffle(userMapList)
        print(M.shape)

        reduced_matrix_train = M[userMapList[:train_num]]

        reduced_matrix_val = M[userMapList[train_num:train_num + val_num]]
        reduced_matrix_test = M[userMapList[train_num+val_num: train_num + val_num + test_num]]
        assert reduced_matrix_train.shape == (train_num, business_num), f"reduced_matrix_train{reduced_matrix_train.shape}-train,d {train_num, d}"
        assert reduced_matrix_val.shape == (val_num, business_num)
        assert reduced_matrix_test.shape == (test_num, business_num)
        target_prefix = args.output

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

        generate_info(U_train, V_train, train_num, top_business, category_list,
                      target_prefix + '/train_item_user_test_rate')
        np.save(target_prefix + '/train_item_user_test_rate/user_item.npy', reduced_matrix_test)
