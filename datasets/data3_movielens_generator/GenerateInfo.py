import json


def generate_info(U, V, user_num, top_items, attr_list, target_prefix):
    # user info: user_preference.txt
    f = open(target_prefix + '/user_preference.txt', 'w')
    for i in range(user_num):
        user = dict()
        user['uid'] = i
        user['preference_v'] = U[i].reshape((-1, 1)).tolist()
        f.write(json.dumps(user) + '\n')
    f.close()

    # pair items and attributes
    attr_list = [attr_list[i] for i in top_items]
    item_attr = dict()
    for attr in attr_list:
        for item in attr:
            if item not in item_attr:
                item_attr[item] = len(item_attr)

    # item info: arm_info.txt
    # item-attribute info: arm_suparm_relation.txt
    f = open(target_prefix + '/arm_suparm_relation.txt', 'w')
    g = open(target_prefix + '/arm_info.txt', 'w')
    separate = ','
    for i, attr in enumerate(attr_list):
        arm = {'a_id': i, 'fv': V[i, :].reshape((-1, 1)).tolist()}
        f.write(str(i) + '\t' + separate.join([str(item_attr[item]) for item in attr]) + ',\n')
        g.write(json.dumps(arm) + '\n')
    f.close()
    g.close()
