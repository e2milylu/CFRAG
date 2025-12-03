import sys

sys.path.append(".")
import argparse
import json
import os
import pickle

import numpy as np
import pandas as pd
from tqdm import tqdm

from prompts.pre_process import load_get_corpus_fn

parser = argparse.ArgumentParser()
parser.add_argument("--task", default='LaMP_2_time')
parser.add_argument("--source", default='recency')

parser.add_argument("--cut_his_len", type=int, default=100) #？

if __name__ == "__main__":
    opts = parser.parse_args()
    print("task: {}".format(opts.task))
    get_corpus_fn = load_get_corpus_fn(opts.task) #适用于LaMa2处理profile为固定格式的函数
    use_date = opts.source.endswith('date') #False

    try:
        cut_prof_len = int(opts.source.split('_')[-1])
    except:
        cut_prof_len = None

    opts.input_path = os.path.join("data", opts.task)

    train_questions = json.load(
        open(os.path.join(opts.input_path, 'train/train_questions.json'), 'r'))
    train_outputs = json.load(
        open(os.path.join(opts.input_path, 'train/train_outputs.json'), 'r'))

    dev_questions = json.load(
        open(os.path.join(opts.input_path, 'dev/dev_questions.json'), 'r'))
    dev_outputs = json.load(
        open(os.path.join(opts.input_path, 'dev/dev_outputs.json'), 'r'))

    #获取train和dev中的user_id和profile，理想情况一个user_id和一个profile，user_id如果重复会出现什么情况，相同的user_id的profile一样吗？猜测一样
    user_id_list = [] #每一个user_id
    corpus_list = [] #每一个user的profile
    #len(user_id_list)==len(corpus_list)
    for i in tqdm(range(len(train_questions))):
        user_id_list.append(train_questions[i]['user_id'])
        profile = sorted(
            train_questions[i]['profile'],
            key=lambda x: tuple(map(int,
                                    str(x['date']).split("-"))))
        if cut_prof_len is not None:
            profile = profile[-cut_prof_len:] #取最近时间的historcial document
        corpus = get_corpus_fn(profile, use_date=use_date)
        if opts.cut_his_len is not None:
            corpus = corpus[-opts.cut_his_len:] #字符串列表
        corpus_list.extend(corpus)

    for i in tqdm(range(len(dev_questions))):
        user_id_list.append(dev_questions[i]['user_id'])
        profile = sorted(
            dev_questions[i]['profile'],
            key=lambda x: tuple(map(int,
                                    str(x['date']).split("-"))))
        if cut_prof_len is not None:
            profile = profile[-cut_prof_len:]
        corpus = get_corpus_fn(profile, use_date=use_date)
        if opts.cut_his_len is not None:
            corpus = corpus[-opts.cut_his_len:]
        corpus_list.extend(corpus)

    """
    len(user_df)?=len(corpus_df)
    user_df:
    id user_id
    user_id去重，得到user的数量
    corpus_df:
    id corpus
    train和dev会出现同一个user_id，但是train和dev的profile不一定一样吧？得看数据集怎么设计的
    condition 1: 同一个user_id，profile相同，len(user_df)==len(corpus_df)
    """
    user_df = pd.DataFrame({"user_id": list(set(user_id_list))})
    user_df['id'] = np.arange(len(user_df))
    print("num user: {}".format(len(user_df)))

    corpus_df = pd.DataFrame(
        {"corpus": ['<pad>'] + list(set(corpus_list)) + ['<mask>', '']})
    corpus_df['id'] = np.arange(len(corpus_df))
    print("num corpus: {}".format(len(corpus_df)))

    user_vocab = user_df.set_index('id', drop=False).to_dict('index')
    user2id = {
        user['user_id']: id
        for id, user in user_vocab.items() if user['user_id'] is not None
    }

    corpus_vocab = corpus_df.set_index('id', drop=False).to_dict('index')
    corpus2id = {
        corpus['corpus']: id
        for id, corpus in corpus_vocab.items() if corpus['corpus'] is not None
    }

    for idx in user_vocab.keys():
        user_vocab[idx]['profile'] = []
        user_vocab[idx]['corpus_ids'] = []

    for i in tqdm(range(len(train_questions))):
        user_id = train_questions[i]['user_id']
        profile = sorted(
            train_questions[i]['profile'],
            key=lambda x: tuple(map(int,
                                    str(x['date']).split("-"))))
        if cut_prof_len is not None:
            profile = profile[-cut_prof_len:]
        corpus = get_corpus_fn(profile, use_date=use_date)
        if opts.cut_his_len is not None:
            corpus = corpus[-opts.cut_his_len:]
        corpus_ids = [corpus2id[x] for x in corpus] #每一个historical document中的每一条记录的id

        new_profile = []
        for j in range(len(profile)):
            cur_profile = profile[j]
            cur_profile['user_id'] = user_id
            new_profile.append(cur_profile)

        if len(user_vocab[user2id[user_id]]['corpus_ids']) == 0:
            user_vocab[user2id[user_id]]['profile'] = new_profile #一个user的所有historical
            user_vocab[user2id[user_id]]['corpus_ids'] = corpus_ids #一个user的topk historical
        else:
            prev_corpus_ids = user_vocab[user2id[user_id]]['corpus_ids']
            assert prev_corpus_ids == corpus_ids

    for i in tqdm(range(len(dev_questions))):
        user_id = dev_questions[i]['user_id']
        profile = sorted(
            dev_questions[i]['profile'],
            key=lambda x: tuple(map(int,
                                    str(x['date']).split("-"))))
        if cut_prof_len is not None:
            profile = profile[-cut_prof_len:]
        corpus = get_corpus_fn(profile, use_date=use_date)
        if opts.cut_his_len is not None:
            corpus = corpus[-opts.cut_his_len:]
        corpus_ids = [corpus2id[x] for x in corpus]

        new_profile = []
        for j in range(len(profile)):
            cur_profile = profile[j]
            cur_profile['user_id'] = user_id
            new_profile.append(cur_profile)

        if len(user_vocab[user2id[user_id]]['corpus_ids']) == 0:
            user_vocab[user2id[user_id]]['profile'] = new_profile
            user_vocab[user2id[user_id]]['corpus_ids'] = corpus_ids
        else:
            prev_corpus_ids = user_vocab[user2id[user_id]]['corpus_ids']
            assert prev_corpus_ids == corpus_ids

    user_profile_len = pd.DataFrame({
        "len": [len(user_vocab[i]['profile']) for i in range(len(user_vocab))]
    })
    print(user_profile_len.describe())

    with open(
            os.path.join(opts.input_path, f"dev/{opts.source}/user_vocab.pkl"),
            "wb") as file:
        pickle.dump(user_vocab, file)

    with open(os.path.join(opts.input_path, f"dev/{opts.source}/user2id.pkl"),
              "wb") as file:
        pickle.dump(user2id, file)

    with open(
            os.path.join(opts.input_path,
                         f"dev/{opts.source}/corpus_vocab.pkl"), "wb") as file:
        pickle.dump(corpus_vocab, file)

    with open(
            os.path.join(opts.input_path, f"dev/{opts.source}/corpus2id.pkl"),
            "wb") as file:
        pickle.dump(corpus2id, file)
