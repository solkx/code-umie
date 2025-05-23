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

def main(root, mode, lang, lr):
    result_list = {}
    result_avg_list = {}
    for res_name in tqdm(os.listdir(f"./{root}")):
        if f"-{mode}-" not in res_name:
            continue
        c_lang = res_name.split("-")[0]
        if c_lang != lang:
            continue   
        c_lr = float(res_name.split("-")[4].split(".json")[0])
        if lr != c_lr:
            continue
        seed = res_name.split('-')[3]
        result_list[seed] = {}
        result_avg_list[seed] = {}
        if is_print:
            print(res_name)
        with open(f"./{root}/{res_name}", "r", encoding="utf-8") as f:
            result = [json.loads(line) for line in f.read().split("\n") if line]
            total_rel_p, total_rel_r, total_rel_c = 0, 0, 0
            type_dic, type_dic_prc = {}, {}
            for item in result:
                pred_result = item["result"]
                rel_gold = item["relation"]
                if rel_gold in t2n or rel_gold in j_t2n:
                    pass
                else:
                    rel_gold = n2t[rel_gold]
                if rel_gold != "None":
                    total_rel_r += 1
                try:
                    relation_dic = {}
                    exec(pred_result.replace('<s> ', "").strip())
                    pred_result = list(relation_dic.keys())[0]
                    if pred_result in t2n or pred_result in j_t2n:
                        pass
                    else:
                        pred_result = n2t[pred_result]
                    if pred_result != "None":
                        if pred_result in t2n or pred_result in j_t2n:
                            total_rel_p += 1
                        else:
                            pred_result = "None"
                except:
                    pred_result = "None"
                if pred_result not in type_dic:
                    type_dic[pred_result] = [0, 0, 0]
                if rel_gold not in type_dic:
                    type_dic[rel_gold] = [0, 0, 0]
                if pred_result == rel_gold:
                    type_dic[rel_gold][-1] += 1
                    if pred_result != "None":
                        total_rel_c += 1
                else:
                    type_dic[pred_result][0] += 1
                    type_dic[rel_gold][1] += 1

            # print(total_rel_p, total_rel_r, total_rel_c)
            fp = sum([v[0] for v in type_dic.values()])
            fn = sum([v[1] for v in type_dic.values()])
            tp = sum([v[2] for v in type_dic.values()])

            # for k, v in type_dic.items():
            #     type_dic_prc[k] = list(calu_res(v[0], v[1], v[2]))
            # mean_values = list(map(lambda x: sum(x) / len(x), zip(*type_dic_prc.values())))
            rel_precious, rel_recall, rel_f1 = calu_res(total_rel_p, total_rel_r, total_rel_c)
            # print(rel_precious, rel_recall, rel_f1)
            total_rel_p, total_rel_r, total_rel_c = tp + fp, tp + fn, tp
            mean_values = calu_res(total_rel_p, total_rel_r, total_rel_c)
            table = pt.PrettyTable([f"{mode}-{lang}", "Precision", "Recall", "F1"])
            table.add_row(["Relation"] + ["{:3.4f}".format(x) for x in [rel_precious, rel_recall, rel_f1]])
            table.add_row(["Relation_avg"] + ["{:3.4f}".format(x) for x in [mean_values[0], mean_values[1], mean_values[2]]])

            result_list[seed] = [rel_precious, rel_recall, rel_f1]
            result_avg_list[seed] = [mean_values[0], mean_values[1], mean_values[2]]
    return result_list, result_avg_list

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
    mode_list = [["result_output_code_mnre_with_entity_type","all-llama", "en", 0.0002]]
    tabel_list = []
    for root, mode, lang, lr in mode_list:
        # print(root, mode, rank, lang)
        error = []
        rel_prc_list = []
        rand_len = 2
        result_list, result_avg_list = main(root, mode, lang, lr)
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
            # print(result_list)
            # print(round(sum(result_list) / len(result_list), 4), round(np.std(result_list), 4))
            table = pt.PrettyTable([f"{mode}-{lang}", "Precision", "Recall", "F1/mAP"])
            table.add_row(["Relation"] + [f"{rel_precious[0]}({rel_precious[-1]})", f"{rel_recall[0]}({rel_recall[-1]})", f"{rel_f1[0]}({rel_f1[-1]})"])
            print(table)
            tabel_list.append(table)
        result_avg_list = sorted(result_avg_list.items(), key=lambda x:x[-1][-1], reverse=True)
        print([f"{item[0]}-{round(item[-1][-1], 4)}" for item in result_avg_list])
        result_avg_list = result_avg_list[:3]
        if result_avg_list:
            rel_precious_list = [item[-1][0] for item in result_avg_list]
            rel_precious = res(rel_precious_list)
            rel_recall_list = [item[-1][1] for item in result_avg_list]
            rel_recall = res(rel_recall_list)
            rel_f1_list = [item[-1][2] for item in result_avg_list]
            rel_f1 = res(rel_f1_list)
            rel_precious_list = [item[-1][0] for item in result_avg_list]
            rel_precious = res(rel_precious_list)
            rel_recall_list = [item[-1][1] for item in result_avg_list]
            rel_recall = res(rel_recall_list)
            rel_f1_list = [item[-1][2] for item in result_avg_list]
            rel_f1 = res(rel_f1_list)
            # print(result_list)
            # print(round(sum(result_list) / len(result_list), 4), round(np.std(result_list), 4))
            table = pt.PrettyTable([f"{mode}-{lang}", "Precision", "Recall", "F1/mAP"])
            table.add_row(["Relation"] + [f"{rel_precious[0]}({rel_precious[-1]})", f"{rel_recall[0]}({rel_recall[-1]})", f"{rel_f1[0]}({rel_f1[-1]})"])
            print(table)
            tabel_list.append(table)