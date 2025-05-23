import logging
import time
from scipy.optimize import linear_sum_assignment
import numpy as np
import itertools
import re
import json
import prettytable as pt
import cv2
with open("./data/type2nature_all.json", "r", encoding="utf-8") as f:
    t2n = json.loads(f.read())["mnre"]
    j_t2n = [i.split("/")[-1] for i in t2n.keys()]

n2t = {}
for k, v in t2n.items():
    n2t[v] = k

def calu_res(p, r, c_p, c_r=None):
    if not c_r:
        c_r = c_p
    try:
        precious = c_p / p
    except ZeroDivisionError:
        precious = 0
    try:
        recall = c_r / r
    except ZeroDivisionError:
        recall = 0

    try:
        f1 = 2 * precious * recall / (precious + recall)
    except ZeroDivisionError:
        f1 = 0
    return round(precious, 4), round(recall, 4), round(f1, 4)

from tqdm import tqdm
import copy
def main(root, mode, lang, lr):
    result_list = {}
    result_avg_list = {}
    for res_name in tqdm(os.listdir(f"./{root}")):  
        c_lr = float(res_name.split("-")[4].split(".json")[0])
        if lr != c_lr:
            continue
        seed = res_name.split('-')[3]
        result_list[seed] = {}
        result_avg_list[seed] = {}
        if is_print:
            print(res_name)
        n2t_15 = {
                "PER": "person",
                "LOC": "location",
                "ORG": "organization",
                "OTHER": "other",
                "MISC": "miscellaneous"
            }
        with open(f"./{root}/{res_name}", "r", encoding="utf-8") as f:
            result = [json.loads(line) for line in f.read().split("\n") if line]
            total_rel_p, total_rel_r, total_rel_c = 0, 0, 0
            type_dic, type_dic_prc = {}, {}
            for item in result:
                # print(item)
                pred_result = item["result"]
                doc = item["doc"]
                chain_dic = {}
                words = [word.lower() for word in doc.split(" ")]
                gold_link = item["link"]
                for pred in pred_result.split("\n"):
                    if "chain_dic" in pred and " = " in pred:
                        try:
                            exec(pred.replace('<s> ', "").strip())
                        except:
                            pass
                new_ent = {}
                for link_id, value in chain_dic.items():
                    ent = value[0][0].lower().strip()
                    ent_type = value[-1]
                    min_ = 100000
                    min_index = None
                    for span in range(1, len(words)):
                        for i in range((len(words))-span+1):
                            q = "".join(words[i:i+span]).strip()
                            if not q:
                                continue
                            copy_q = copy.deepcopy(q)
                            k = ent.replace(" ", "")
                            copy_k = copy.deepcopy(k)
                            for s in k:
                                copy_q = copy_q.replace(s, "", 1)
                            for s in q:
                                copy_k = copy_k.replace(s, "", 1)
                            l = max([len(copy_q), len(copy_k)])
                            if l < 1 and l < min_:
                                min_ = l
                                min_index = [i, i+span, ent_type, " ".join(words[i:i+span]), l, copy_q, copy_k]
                        if min_index is not None:
                            new_ent[link_id] = min_index
                # print(chain_dic)
                # print(new_ent)
                # print(gold)
                # print(words)
                # print("="*50)
                pred = []
                for v in new_ent.values():
                    # print(v[1] - v[0])
                    if v[1] - v[0] >= 5:
                        continue
                    pred_ent = " ".join(words[v[0]:v[1]]).lower()
                    # for word in pred_ent.split(" "):
                    pred.append(f"{pred_ent}-{v[2]}")
                gold = []
                for k, v in gold_link.items():
                    # for word in k.lower().split(" "):
                    gold.append(f"{k.lower()}-{n2t_15[v]}")
                # print(pred)
                # print(gold)
                # exit()
                total_rel_p += len(pred)
                total_rel_r += len(gold)
                total_rel_c += len(set(pred) & set(gold))
                # if list(set(pred) - set(gold)):
                #     print(list(set(gold) - set(pred)))
                #     print(list(set(pred) - set(gold)))
                #     print("="*50)
            rel_precious, rel_recall, rel_f1 = calu_res(total_rel_p, total_rel_r, total_rel_c)
            table = pt.PrettyTable([f"{mode}-{lang}", "Precision", "Recall", "F1"])
            table.add_row(["Relation"] + ["{:3.4f}".format(x) for x in [rel_precious, rel_recall, rel_f1]])
            # exit()

            result_list[seed] = [rel_precious, rel_recall, rel_f1]
    return result_list

def res(res_list):
    return round(sum(res_list) / len(res_list) * 100, 2), round(np.std(res_list) * 100, 2)

def get_avg_f1(f1_list):
    r = []
    for f1, res in f1_list:
        r.append(f1)
    return sum(r) / len(r)

if __name__ == "__main__":
    import os
    from openpyxl import Workbook
    with open("../bloom-multi/zh_video_list.json", "r", encoding="utf-8") as f:
        zh_video_list = json.loads(f.read())
    temp_temp_temp = []
    is_print = False
    # res_mode = "all_result_sorted"
    res_mode = 0
    test_len = {"zh":205, "en":207,"mix":412}
    mode_list = [["result_output_code_mner17_ours_code","llama", "en", 0.0002]]
    tabel_list = []
    for root, mode, lang, lr in mode_list:
        # print(root, mode, rank, lang)
        error = []
        rel_prc_list = []
        rand_len = 2
        result_list = main(root, mode, lang, lr)
        result_list = sorted(result_list.items(), key=lambda x:x[-1][-1], reverse=True)
        print([f"{item[0]}-{round(item[-1][-1], 4)}" for item in result_list])
        result_list = result_list[:3]
        if result_list:
            rel_precious_list = [item[-1][0] for item in result_list]
            rel_precious = res(rel_precious_list)
            rel_recall_list = [item[-1][1] for item in result_list]
            rel_recall = res(rel_recall_list)
            rel_f1_list = [item[-1][2] for item in result_list]
            rel_f1 = res(rel_f1_list)
            rel_precious_list = [item[-1][0] for item in result_list]
            rel_precious = res(rel_precious_list)
            rel_recall_list = [item[-1][1] for item in result_list]
            rel_recall = res(rel_recall_list)
            rel_f1_list = [item[-1][2] for item in result_list]
            rel_f1 = res(rel_f1_list)
            table = pt.PrettyTable([f"{mode}-{lang}", "Precision", "Recall", "F1/mAP"])
            table.add_row(["Relation"] + [f"{rel_precious[0]}({rel_precious[-1]})", f"{rel_recall[0]}({rel_recall[-1]})", f"{rel_f1[0]}({rel_f1[-1]})"])
            print(table)
            tabel_list.append(table)