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
    print("a", os.path.join(input_folder, "pics_ut/out.pics_ut"))
    print("a", os.path.join(input_folder, "pic_ti/out.pics_ti"))
    artist_to_tagcount = {}
    tag_to_artists = {}
    df0 = pd.read_csv(os.path.join(input_folder, "pics_ut/out.pics_ut"), sep=" ", header=None, index_col=None)
    df1 = pd.read_csv(os.path.join(input_folder, "pics_ti/out.pics_ti"), sep=" ", header=None, index_col=None)
    df = pd.merge(df0, df1, left_index=True, right_index=True)
    for i, row in tqdm(df.iterrows()):
        if i > 9:
            exit(1)
        if row[4] not in artistID:
            artistID[row[4]] = len(artistID)  # item
        if row[0] not in userID:
            userID[row[0]] = len(userID)  # user
        if row[1] not in tagID:
            tagID[row[1]] = len(tagID)  # tag
        print(row[0], row[1], row[4])

        artist = artistID[row[4]]
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
    return spp.csr_matrix((np.asarray(data), (np.asarray(rows), np.asarray(cols)))), artist_to_tags

if __name__ == "__main__":
    np.random.seed(12326)
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", dest="input", type=str, help="Path to the input folder.", default=None)
    parser.add_argument("-o", "--output", dest="output", type=str, help="Path to the output folder.", default=None)
    d = 50



    business_num = 10000
    user_num = 1500
    train_num, val_num, test_num = 500, 500, 500

    assert train_num + val_num + test_num == user_num

    args = parser.parse_args()
    if not args.input:
        args.input = "../../source/visualizeus/"
    if not args.output:
        args.output = f"./un_{user_num}-bn_{business_num}_0616_replay/"
    os.makedirs(args.output, exist_ok=True)
    print(args.input)
    print(args.output)

    # Extract data.
    tmp_file_full_matrix_artist_to_tag = "./tmp_file_full_matrix_artist_to_tag.npy"

    if 1: #not os.path.exists(tmp_file_full_matrix_artist_to_tag):
        full_matrix, artist_to_tags = load_sparse_matrix(args.input)
        to_save = (full_matrix, artist_to_tags)
        np.save(tmp_file_full_matrix_artist_to_tag, to_save)
    else:
        full_matrix, artist_to_tags = np.load(tmp_file_full_matrix_artist_to_tag)

    M, category_list = full_matrix, artist_to_tags
    print("M.shape", M.shape)  # 57944 767447
    # np.save(args.output+"top_business.npy", top_business)
    if not os.path.exists(target_prefix + "_top_business.npy"):
        M, top_business = get_review_matrix(user_num, business_num, M)
        print("M.shape", M.shape)
        np.save(target_prefix + "_top_business.npy", top_business)
    else:
        top_business = np.load(target_prefix + "_top_business.npy")
        print("top_business", top_business)
        print("top_business", top_business.__len__())

    print("max top_business", max(top_business))
    arm_id_2_arm_index = {}
    for i, arm_id in enumerate(top_business):
        arm_id_2_arm_index[arm_id] = i
    arm_id_2_arm_index_file = target_prefix + "_arm_id_2_arm_index.npy"
    np.save(arm_id_2_arm_index_file, arm_id_2_arm_index)
