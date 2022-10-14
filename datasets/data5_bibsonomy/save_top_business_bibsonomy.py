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
    print("input_folder", input_folder)
    print("current path", os.getcwd())
    df0 = pd.read_csv(os.path.join(input_folder, "bibsonomy-2ut/out.bibsonomy-2ut"), sep=" ", header=None, index_col=None)
    df1 = pd.read_csv(os.path.join(input_folder, "bibsonomy-2ti/out.bibsonomy-2ti"), sep=" ", header=None, index_col=None)
    print(df0.head(10))
    print(df1.head(10))
    df = pd.merge(df0, df1, left_index=True, right_index=True)
    print(df.head(10))
    for i, row in tqdm(df.iterrows()):
        if row[5] not in artistID:  # 1_y artist
            artistID[row[5]] = len(artistID)
        if row[0] not in userID:  # 0_x item
            userID[row[0]] = len(userID)
        if row[1] not in tagID:  # 0_y tag
            tagID[row[1]] = len(tagID)
        # print(row[0], row[1], row[5])
        # if i > 5:
        #     exit(1)
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

    tag_to_artistcount = {}
    for tag in tag_to_artists:
        tag_to_artistcount[tag] = len(set(tag_to_artists[tag]))

    artist_to_tags = {}
    for artist in artist_to_tagcount:
        related_tags = artist_to_tagcount[artist]
        related_tag_to_artistcount = {tag: tag_to_artistcount[tag] for tag in related_tags}

        # Limit the number of tags.
        related_tag_to_artistcount = dict(sorted(related_tag_to_artistcount.items(), key=lambda x: x[1], reverse=True)[:20])
        related_tags = list(related_tag_to_artistcount)
        artist_to_tags[artist] = related_tags

    print(len(tag_to_artists), len(set(rows)), len(set(cols)))  # tagnum204673 user5794 item767447  #
    return spp.csr_matrix((np.asarray(data), (np.asarray(rows), np.asarray(cols)))), artist_to_tags, artistID, userID, tagID

if __name__ == "__main__":
    np.random.seed(12326)
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", dest="input", type=str, help="Path to the input folder.", default=None)
    parser.add_argument("-o", "--output", dest="output", type=str, help="Path to the output folder.", default=None)

    business_num = 5000
    user_num = 1500
    train_num, val_num, test_num = 500, 500, 500

    d = 50
    assert train_num + val_num + test_num == user_num

    args = parser.parse_args()
    if not args.input:
        args.input = "../../source/bibsonomy/"
    if not args.output:
        args.output = f"./un_{user_num}-bn_{business_num}_0616_replay"

    source_data = args.input
    target_prefix = args.output
    os.makedirs(args.output, exist_ok=True)
    print(args.input)
    print(args.output)

    tmp_M_cat_top_itemuserdict_save = f'{target_prefix}_M_cat_top_useritemdict.npy'

    if not os.path.exists(tmp_M_cat_top_itemuserdict_save):
        M, category_list, business_dict, user_dict, tag_dict = load_sparse_matrix(source_data)
        M, top_business = get_review_matrix(user_num, business_num, M)
        M_cat_top = [M, category_list, top_business, business_dict, user_dict, tag_dict]
        np.save(tmp_M_cat_top_itemuserdict_save, M_cat_top)
        print("m_cat_top exist")
    else:
        reduced_matrix, category_list, top_business, business_dict, user_dict, tag_dict = np.load(tmp_M_cat_top_itemuserdict_save)

    print(business_dict)
    print(business_dict.__len__())  # 59047
    print(user_dict.__len__())  # 162541
    print(len(set(business_dict.keys())))

    def save_raw_id_2_matrix_index(business_dict):
        raw_id_2_matrix_index = {}
        for business_dict_each in business_dict:
            raw_id_2_matrix_index[int(business_dict_each)] = business_dict[business_dict_each]
        return raw_id_2_matrix_index
    raw_id_2_matrix_index = save_raw_id_2_matrix_index(business_dict)
    raw_id_2_matrix_index_file = target_prefix + "_raw_id_2_matrix_index.npy"
    # arm_index -> matrix_index -> raw_id
    np.save(raw_id_2_matrix_index_file, raw_id_2_matrix_index)


    # top_business [  159    82     0 ... 15814 17611 17688]
    # 159 to 0, 82 to 1
    print("max top_business", max(top_business))
    matrix_index_2_arm_index = {}
    for i, arm_id in enumerate(top_business):
        matrix_index_2_arm_index[arm_id] = i
    matrix_index_2_arm_index_file = target_prefix + "_matrix_index_2_arm_index.npy"
    np.save(matrix_index_2_arm_index_file, matrix_index_2_arm_index)


    tag_raw_id_2_matrix_index = {}
    for raw_id in tag_dict:
        print("raw_id", raw_id)
        tag_raw_id_2_matrix_index[raw_id] = tag_dict[raw_id]
    tag_raw_id_2_matrix_index_file = target_prefix + "_tag_raw_id_2_matrix_index.npy"
    np.save(tag_raw_id_2_matrix_index_file, tag_raw_id_2_matrix_index)
