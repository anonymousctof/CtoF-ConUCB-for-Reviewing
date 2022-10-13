import argparse
import os

import numpy as np
import pandas as pd
import scipy.sparse as spp
from tqdm import tqdm

def load_sparse_matrix(input_folder):
    userID = {}
    artistID = {}
    tagID = {}
    rows = []
    cols = []
    data = []
    artist_to_tagcount = {}
    tag_to_artists = {}
    df0 = pd.read_csv(os.path.join(input_folder, "delicious-ut/out.delicious_delicious-ut_delicious-ut"), sep=" ", header=None, index_col=None, skiprows=2)
    df1 = pd.read_csv(os.path.join(input_folder, "delicious-ti/out.delicious_delicious-ti_delicious-ti"), sep=" ", header=None, index_col=None, skiprows=2)
    df = pd.merge(df0, df1, left_index=True, right_index=True)
    for i, row in tqdm(df.iterrows()):
        if i > 9:
            exit(1)
        # print(row[0], row[1], row[5])

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

    meta_data = {"rating_num": len(data), "user_num": len(rows), "item_num": len(cols),
                 "user_num_of_user_dict": len(userID), "item_num_of_bussiness_dict": len(artistID),
                 "key_term": len(tagID)}
    import json
    json.dump(meta_data, open(f"{input_folder}/meta.json", 'w'))

    return spp.csr_matrix((np.asarray(data), (np.asarray(rows), np.asarray(cols)))), artist_to_tags
if __name__ == "__main__":
    np.random.seed(12326)
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", dest="input", type=str, help="Path to the input folder.", default=None)
    parser.add_argument("-o", "--output", dest="output", type=str, help="Path to the output folder.", default=None)
    d = 50

    args = parser.parse_args()
    if not args.input:
        args.input = "../../source/delicious/"

    full_matrix, artist_to_tags = load_sparse_matrix(args.input)
