from sklearn.preprocessing import normalize, minmax_scale
import scipy.sparse as spp
import numpy as np


def extract_rows(top_k, sparse_matrix):
    business_review_count = sparse_matrix.getnnz(axis=1)
    business_count = business_review_count.shape[0]
    print("business_count - 1, business_count - 1 - top_k", business_count - 1, business_count - 1 - top_k)
    if top_k >= business_count:
        top_k_index = np.argsort(business_review_count)
    else:
        top_k_index = np.argsort(business_review_count)[business_count - 1: business_count - 1 - top_k:-1]
    # top_k_index = np.random.choice(business_count, top_k, replace=False)
    print("top_k_index.shape", top_k_index.shape)
    matrix = spp.vstack([sparse_matrix.getrow(i) for i in top_k_index])
    return matrix


def extract_cols(top_k, sparse_matrix):
    user_review_count = sparse_matrix.getnnz(axis=0)
    user_count = user_review_count.shape[0]
    if top_k >= user_count:
        top_k_index = np.argsort(user_review_count)
    else:
        top_k_index = np.argsort(user_review_count)[user_count - 1: user_count - 1 - top_k:-1]
    # top_k_index=np.random.choice(user_count, top_k, replace=False)
    matrix = spp.hstack([sparse_matrix.getcol(i) for i in top_k_index])
    return matrix, top_k_index


def get_review_matrix(user_num, item_num, sparse_matrix):
    print("user_num*3", user_num*3)
    row_reduced_matrix = extract_rows(user_num * 3, sparse_matrix)
    reduced_matrix, top_items = extract_cols(item_num, row_reduced_matrix)
    reduced_matrix = extract_rows(user_num, reduced_matrix)
    reduced_matrix = reduced_matrix.toarray()
    return reduced_matrix, top_items


def extract_features(num_users, num_items, dim, inputs):
    A1 = inputs[:num_users, :num_items]
    u, s, vt = np.linalg.svd(A1)

    u = u[:, :dim - 1]
    u = normalize(u, axis=1, norm='l2')
    v = vt.T[:, :dim - 1]
    v = normalize(v, axis=1, norm='l2')

    U = np.concatenate((u, np.ones((num_users, 1))), axis=1) / np.sqrt(2)
    V = np.concatenate((v, np.ones((num_items, 1))), axis=1) / np.sqrt(2)
    return U, V
