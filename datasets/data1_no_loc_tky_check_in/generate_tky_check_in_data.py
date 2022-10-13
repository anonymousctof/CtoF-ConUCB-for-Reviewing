import argparse
import os

import numpy as np
import pandas as pd
import scipy.sparse as spp
from tqdm import tqdm
from ExtractFeatures import extract_features, get_review_matrix
from GenerateInfo import generate_info

def load_sparse_matrix(input_folder):
    """
    1. User ID (anonymized)
    2. Venue ID (Foursquare)
    3. Venue category ID (Foursquare)
    4. Venue category name (Fousquare)
    5. Latitude
    6. Longitude
    7. Timezone offset in minutes (The offset in minutes between when this check-in occurred and the same time in UTC)
    8. UTC time
    470	49bbd6c0f964a520f4531fe3	4bf58dd8d48988d127951735	Arts & Crafts Store	40.719810375488535	-74.00258103213994	-240	Tue Apr 03 18:00:09 +0000 2012
    979	4a43c0aef964a520c6a61fe3	4bf58dd8d48988d1df941735	Bridge	40.60679958140643	-74.04416981025437	-240	Tue Apr 03 18:00:25 +0000 2012
    69	4c5cc7b485a1e21e00d35711	4bf58dd8d48988d103941735	Home (private)	40.716161684843215	-73.88307005845945	-240	Tue Apr 03 18:02:24 +0000 2012
    395	4bc7086715a7ef3bef9878da	4bf58dd8d48988d104941735	Medical Center	40.7451638	-73.982518775	-240	Tue Apr 03 18:02:41 +0000 2012
    87	4cf2c5321d18a143951b5cec	4bf58dd8d48988d1cb941735	Food Truck	40.74010382743943	-73.98965835571289	-240	Tue Apr 03 18:03:00 +0000 2012
    484	4b5b981bf964a520900929e3	4bf58dd8d48988d118951735	Food & Drink Shop	40.69042711809854	-73.95468677509598	-240	Tue Apr 03 18:04:00 +0000 2012
    642	4ab966c3f964a5203c7f20e3	4bf58dd8d48988d1e0931735	Coffee Shop	40.751591431346306	-73.9741214009634	-240	Tue Apr 03 18:04:38 +0000 2012
    :param input_folder:
    :return:
    """
    tag_id_2_tag_name = {}

    userID = {}
    artistID = {}
    tagID = {}
    rows = []
    cols = []
    data = []
    artist_to_tagcount = {}
    tag_to_artists = {}
    df = pd.read_csv(input_folder + '.txt', sep='\t', header=None, engine="python", names=['user_id', "item_id", "tag_id", "tag_name", "latitude", "longitude", "timezone_offset", "utc_time"])

    item_matrix_index_2_tude = {}
    for i, row in df.iterrows():  #  tqdm(df.iterrows()):
        if row['item_id'] not in artistID:
            artistID[row['item_id']] = len(artistID)  # item
        if row['user_id'] not in userID:
            userID[row['user_id']] = len(userID)  # user
        if row['tag_id'] not in tagID:
            tagID[row['tag_id']] = len(tagID)  # tag
        artist = artistID[row['item_id']]
        user = userID[row['user_id']]
        tag = tagID[row['tag_id']]
        latitude = row['latitude']
        longitude = row['longitude']
        if artist not in item_matrix_index_2_tude:
            item_matrix_index_2_tude[artist] = (latitude, longitude)

        if row['tag_id'] not in tag_id_2_tag_name:
            tag_id_2_tag_name[row['tag_id']] = row['tag_name']
        else:
            assert tag_id_2_tag_name[row['tag_id']] == row['tag_name']

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
    meta_data = {"rating_num": len(data), "user_num": len(rows), "item_num": len(cols),
                 "user_num_of_user_dict": len(userID), "item_num_of_bussiness_dict": len(artistID),
                 "key_term": len(tagID)}
    print(meta_data)
    import json
    os.makedirs(f'{input_folder}', exist_ok=True)
    json.dump(meta_data, open(f"{input_folder}/meta.json", 'w'))

    return spp.csr_matrix((np.asarray(data), (np.asarray(rows), np.asarray(cols)))), artist_to_tags, artistID, userID, tagID, tag_id_2_tag_name, item_matrix_index_2_tude

if __name__ == "__main__":
    seed = 12345
    sig = "0623"
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
    business_num_list = [5000]
    assert train_num + val_num + test_num == user_num
    for business_num in business_num_list:
        args = parser.parse_args()
        if not args.input:
            args.input = "../../source/four_square/dataset_tsmc2014/dataset_TSMC2014_TKY"
        if not args.output:
            args.output = f"./{target_prefix_p}_un_{user_num}-bn_{business_num}/"
        os.makedirs(args.output, exist_ok=True)
        print(args.input)
        print(args.output)

        # Extract data.
        tmp_file_matrix_result = "./tmp_file_matrix_result.npy"

        if not os.path.exists(tmp_file_matrix_result):
            full_matrix, artist_to_tags, artistID, userID, tagID, tag_id_2_tag_name, item_matrix_index_2_tude = load_sparse_matrix(args.input)
            to_save = (full_matrix, artist_to_tags, artistID, userID, tagID, tag_id_2_tag_name, item_matrix_index_2_tude)
            np.save(tmp_file_matrix_result, to_save)
        else:
            full_matrix, artist_to_tags, artistID, userID, tagID, tag_id_2_tag_name, item_matrix_index_2_tude = np.load(tmp_file_matrix_result)

        M, category_list = full_matrix, artist_to_tags
        print("M.shape", M.shape)
        M, top_business = get_review_matrix(user_num, business_num, M)
        print("M.shape", M.shape)

        if cut_value is not None:
            M = np.clip(M, -10, cut_value)
            print("cuted")
            input()

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

        # os.makedirs(target_prefix + '/train_item_user_test_rate', exist_ok=True)
        generate_info(U_train, V_train, train_num, top_business, category_list,
                      target_prefix + '/train_item_user_test_rate')
        np.save(target_prefix + '/train_item_user_test_rate/user_item.npy', reduced_matrix_test)
