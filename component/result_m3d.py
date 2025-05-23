import logging
import time
from scipy.optimize import linear_sum_assignment
import numpy as np
import itertools
import re
import json
import prettytable as pt
import cv2

with open("../bloom-multi/data/type2nature_all.json", "r", encoding="utf-8") as f:
    t2n = json.loads(f.read())

not_test_entity_list = []
for lang in ["zh", "en"]:
    for name in ["train", "dev"]:
        with open(f'../bloom-multi/data/data/{lang}/{name}_{lang}.json', 'r', encoding='utf-8') as f:
            train_data = json.loads(f.read())
        for item in train_data:
            for linkItem in item["entityLink"].values():
                for ent in linkItem["link"]:
                    if ent["text"] not in not_test_entity_list:
                        not_test_entity_list.append(ent["text"])

per_list = [
  "I", "you", "he", "she", "it", "we", "they",  # 人称代词 - 主格
  "me", "you", "him", "her", "it", "us", "them",  # 人称代词 - 宾格
  "my", "your", "his", "her", "its", "our", "their",  # 描述性物主代词
  "mine", "yours", "his", "hers", "its", "ours", "theirs",  # 独立物主代词
  "myself", "yourself", "himself", "herself", "itself", "ourselves", "yourselves", "themselves",  # 反身代词
  "each other", "one another",  # 相互代词
  "this", "that", "these", "those",  # 指示代词
  "who", "whom", "whose", "which", "what",  # 疑问代词
  "who", "whom", "whose", "which", "that",  # 关系代词
  "all", "any", "both", "each", "either", "few", "many", "more", "most", "much", "neither", "none", "no one", "nobody", "one", "several", "some", "someone", "somebody", "such", "few", "little", "much", "several", "enough"  # 不定代词
]

n2t_ent = {"en":{},"zh":{}}
n2t_rel = {"en":{},"zh":{}}
for ent_t, n_t in t2n["en"]["entity_type_dic"].items():
    n2t_ent["en"][n_t] = ent_t
for rel_t, r_t in t2n["en"]["relation_type_dic"].items():
    n2t_rel["en"][r_t] = rel_t

for ent_t, n_t in t2n["zh"]["entity_type_dic"].items():
    n2t_ent["zh"][n_t] = ent_t
for rel_t, r_t in t2n["zh"]["relation_type_dic"].items():
    n2t_rel["zh"][r_t] = rel_t

def get_logger(config):
    pathname = f"./log/test_{time.strftime('%m-%d_%H-%M-%S')}.txt"
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s",
                                  datefmt='%Y-%m-%d %H:%M:%S')

    file_handler = logging.FileHandler(pathname)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger

def gold_pro(gold, gold_rel):
    gold_link = []
    gold_entity = []
    gold_id2link = {}
    gold_relType2link = {}
    for link_id, entity_item in gold.items():
        entity_type = entity_item["type"]
        entity_list = []
        for entity in entity_item["link"]:
            entity = entity["text"]
            num = 0
            while True:
                if entity not in entity_list:
                    entity_list.append(entity)
                    gold_entity.append(f"{entity}-{entity_type}")
                    break
                else:
                    num += 1
                    entity = entity.split("_")[0] + f"_{num}"
        gold_link.append(entity_list)
        if link_id not in gold_id2link:
            gold_id2link[link_id] = entity_list
    for rel in gold_rel:
        rel_type = rel["type"]
        link1_id = rel["link1"]
        link2_id = rel["link2"]
        if rel_type not in gold_relType2link:
            gold_relType2link[rel_type] = [[gold_id2link[link1_id], gold_id2link[link2_id]]]
        else:
            gold_relType2link[rel_type].append([gold_id2link[link1_id], gold_id2link[link2_id]])

    return gold_entity, gold_link, gold_relType2link


def rel_prc(pre_rel_dic, gold_rel_dic):
    # print(pre_rel_dic)
    # print(gold_rel_dic)
  
    # print("="*100)
    rel_p, rel_r, rel_c = 0, 0, 0
    per_error = []
    for g in gold_rel_dic.values():
        rel_r += len(g)
    for rel_type, rel_pair_list in pre_rel_dic.items():
        is_join = False
        for rel_pair in rel_pair_list:
            rel_p += 1
            if rel_type in gold_rel_dic:
                gold_pair_list = gold_rel_dic[rel_type]
                for gold_pair in gold_pair_list:
                    pre_1 = rel_pair[0]
                    pre_2 = rel_pair[-1]
                    gold_1 = gold_pair[0]
                    gold_2 = gold_pair[-1]
                    if ((set(pre_1) & set(gold_1)) and (set(pre_2) & set(gold_2))) or (
                            (set(pre_2) & set(gold_1)) and (set(pre_1) & set(gold_2))):
                        rel_c += 1
                        gold_pair_list.remove(gold_pair)
                        is_join = True
                if not is_join:
                    per_error.append(f"{pre_1}, {pre_2}")
    rel_prc_list.append([rel_p, rel_r, rel_c])
    error.append(per_error)
    return rel_p, rel_r, rel_c

def error_ent(predList, goldList):
    b_e, t_e, total = 0, 0, 0
    goldDic = {gold.split("-")[0]:gold.split("-")[1] for gold in goldList}
    for pred in predList:
        if pred.split("-")[0] not in goldDic:
            b_e += 1
            total += 1
        else:
            if pred.split("-")[1] != goldDic[pred.split("-")[0]]:
                t_e += 1
                total += 1
    return b_e, t_e, total

def entity_prc(pre_entity, pre_link, gold_entity, gold_link):
    p = len(pre_entity)
    r = len(gold_entity)
    c = len(set(pre_entity) & set(gold_entity))

    ceaf_c, ceaf_p, ceaf_r = ceaf(pre_link, gold_link)
    muc_c, muc_p, muc_r = muc(pre_link, gold_link)
    b3_c_p, b3_c_r, b3_p, b3_r = b3(pre_link, gold_link)
    return p, r, c, ceaf_p, ceaf_r, ceaf_c, muc_p, muc_r, muc_c, b3_c_p, b3_c_r, b3_p, b3_r

def entity_pro(content):
    for p in [".", ",", "?", "!", "(", ")", "[", "]", "{", "}", '"', "'", "@", "#", "$", "%", "^", "…", "&", "*", "-", "_", "+", "=", "/", "\\", "|", ":", ";", "<", ">", "~", "`"]:
        content = content.replace(p, f" {p} ")
    content = content.split(" ")
    while "" in content:
        content.remove("")
    content = " ".join(content)
    return content

def lowercase_pronouns(text):    
    # 使用正则表达式替换代词为小写
    for pronoun in per_list:
        text = re.sub(r'\b' + pronoun + r'\b', pronoun.lower(), text, flags=re.IGNORECASE)
    
    return text

def entity_quchong(matches, doc, is_big_id, lang):
    ent_text2num = {}
    sorted_list = []
    for link_id, entity_list in matches.items():
        try:
            link_id = int(link_id[1:])
        except:
            continue
        if lang == "en":
            entity_list = [entity_pro(ent.strip()) for ent in entity_list]
            entity_list = [ent for ent in entity_list if ent.lower() != ent or ent in not_test_entity_list]
        else:
            entity_list = [ent.strip() for ent in entity_list]
            # entity_list = [ent for ent in entity_list if ent.lower() != ent or ent in not_test_entity_list]
        for entity in entity_list:
            if entity not in ent_text2num:
                ent_text2num[entity] = 1
            else:
                ent_text2num[entity] += 1
        entity_list.append(link_id)
        sorted_list.append(entity_list)
    def custom_sort(item):
        # 首先按照子列表的长度排序
        length = len(item)
        # 如果子列表长度相同，则按照子列表内的最后一项的数字排序
        key = (length, -int(item[-1]))
        return key
    if is_big_id:
        sorted_list = sorted(sorted_list, key=custom_sort)
    else:
        sorted_list = sorted(sorted_list, key=len)
    if lang == "en":
        ent_text2num_list = sorted(ent_text2num.items(), key=lambda x: len(x[0].split(" ")), reverse=True)
    else:
        ent_text2num_list = sorted(ent_text2num.items(), key=lambda x: len(x[0]), reverse=True)
    for ent_text, num in ent_text2num_list:
        if "[" in ent_text or "]" in ent_text:
            continue
        if lang == "en":
            all_real_ent = re.findall(f" {ent_text} ", doc, flags=re.IGNORECASE)
        else:
            all_real_ent = re.findall(ent_text, doc, flags=re.IGNORECASE)
        doc = re.sub(f" {ent_text} ", " ", doc)
        if all_real_ent and ent_text.lower() not in per_list:
            real_ent_num = len(all_real_ent)
        else:
            real_ent_num = 0
        while num > real_ent_num:
            for subList in sorted_list:
                if ent_text in subList:
                    subList.remove(ent_text)
                    num -= 1
                    break
        while num < real_ent_num:
            for subList in sorted_list[::-1]:
                if ent_text in subList:
                    subList.insert(0, ent_text)
                    num += 1
                    break
    return [f'{per[-1]} is {" | ".join(per[:-1])}' for per in sorted_list if len(per) != 1]

def my_strip(my_list):
    return [item.strip() for item in my_list]

def decode_link_rel(result, gold_link, gold_rel, doc, video_id, is_loose, is_big_id, is_drop_id, lang):
    link_list = []
    pre_entity_list = []
    id2link = {}
    temp = []
    pre_relType2link = {}
    gold_entity, gold_link, gold_relType2link = gold_pro(gold_link, gold_rel)
    temp = []
    chain_dic = {}
    relation_dic = {}
    grounding_dic = {}
    for line in result.replace("<s>", "").strip().split("\n"):
        try:
            if "chain_dic" in line and "=" in line:
                exec(line)
            elif "relation_dic" in line and "=" in line:
                exec(line)
            elif "grounding_dic" in line and "=" in line:
                exec(line)
        except:
            # print(line)
            continue
    chain_chain_dic = {}
    for link_id, links in chain_dic.items():
        link_type = str(links[-1])
        if link_type not in chain_chain_dic:
            chain_chain_dic[link_type] = {link_id:links[0]}
        else:
            chain_chain_dic[link_type][link_id] = links[0]
    for line_type, links_dic in chain_chain_dic.items(): 
        matches = entity_quchong(links_dic, doc, is_big_id, lang)
        # print(matches)
        # time.sleep(1)
        # print("="*100)
        if line_type not in n2t_ent[lang]:
            continue
        line_type = n2t_ent[lang][line_type]
        quchong = []
        for pre_link in matches:
            link_id = pre_link.split(" is ")[0].strip()
            entity_list = my_strip(" is ".join(pre_link.split(" is ")[1:]).strip().split("|"))
            if "" in entity_list:
                continue
            new_entity_list = []
            if entity_list not in quchong:
                quchong.append(entity_list)
            else:
                continue

            for entity in entity_list:
                num = 0
                while True:
                    if entity not in new_entity_list:
                        new_entity_list.append(entity)
                        pre_entity_list.append(f"{entity}-{line_type}")
                        break
                    else:
                        num += 1
                        entity = entity.split("_")[0] + f"_{num}"
            link_list.append(new_entity_list)
            if link_id not in id2link:
                id2link[link_id] = [new_entity_list, line_type]
    for rel_type, reltion_list in relation_dic.items():
        if rel_type not in n2t_rel[lang]:
            continue
        rel_type = n2t_rel[lang][rel_type]
        sorted_link_id = sorted(id2link.keys(), key=lambda x: int(x))
        for matche in reltion_list:
            link1_id = matche[0][1:]
            link2_id = matche[-1][1:]
            if not is_drop_id:
                if link1_id == link2_id:
                    if link1_id != sorted_link_id[-1]:
                        link2_id = sorted_link_id[-1]
                    else:
                        link2_id = sorted_link_id[-2]
                if link1_id not in id2link:
                    if link2_id == sorted_link_id[-1]:
                        link1_id = sorted_link_id[-2]
                    else:
                        link1_id = sorted_link_id[-1]
                if link2_id not in id2link:
                    if link1_id == sorted_link_id[-1]:
                        link2_id = sorted_link_id[-2]
                    else:
                        link2_id = sorted_link_id[-1]
            else:
                if link1_id == link2_id or link1_id not in id2link or link2_id not in id2link:
                    continue
            link_1 = id2link[link1_id][0]
            link_2 = id2link[link2_id][0]
            link_1_type = id2link[link1_id][1]
            link_2_type = id2link[link2_id][1]
            if f"{link_1_type}-{link_2_type}" not in rel_type and f"{link_2_type}-{link_1_type}" not in rel_type:
                continue
            if rel_type not in pre_relType2link:
                pre_relType2link[rel_type] = [[link_1, link_2]]
            else:
                pre_relType2link[rel_type].append([link_1, link_2])
    # print(pre_entity_list)
    # print(gold_entity)
    # print("+"*100)
    p, r, c, ceaf_p, ceaf_r, ceaf_c, muc_p, muc_r, muc_c, b3_c_p, b3_c_r, b3_p, b3_r = entity_prc(pre_entity_list, link_list, gold_entity, gold_link)
    rel_p, rel_r, rel_c = rel_prc(pre_relType2link, gold_relType2link)
    ground_p, ground_r, ground_c = grounding_encode_new(grounding_dic, video_id, lang)
    return p, r, c, ceaf_p, ceaf_r, ceaf_c, muc_p, muc_r, muc_c, b3_c_p, b3_c_r, b3_p, b3_r, rel_p, rel_r, rel_c, ground_p, ground_r, ground_c

def calculate_iou(rect1_center_x, rect1_center_y, rect1_width, rect1_height,
                  rect2_center_x, rect2_center_y, rect2_width, rect2_height):
    
    # 计算两个矩形的左上角和右下角坐标
    rect1_left = rect1_center_x - rect1_width / 2
    rect1_right = rect1_center_x + rect1_width / 2
    rect1_top = rect1_center_y - rect1_height / 2
    rect1_bottom = rect1_center_y + rect1_height / 2

    rect2_left = rect2_center_x - rect2_width / 2
    rect2_right = rect2_center_x + rect2_width / 2
    rect2_top = rect2_center_y - rect2_height / 2
    rect2_bottom = rect2_center_y + rect2_height / 2

    # 计算重叠部分的左上角和右下角坐标
    overlap_left = max(rect1_left, rect2_left)
    overlap_right = min(rect1_right, rect2_right)
    overlap_top = max(rect1_top, rect2_top)
    overlap_bottom = min(rect1_bottom, rect2_bottom)

    # 计算重叠部分的宽度和高度
    overlap_width = max(0, overlap_right - overlap_left)
    overlap_height = max(0, overlap_bottom - overlap_top)

    # 计算重叠部分的面积
    overlap_area = overlap_width * overlap_height

    # 计算两个矩形的面积
    rect1_area = rect1_width * rect1_height
    rect2_area = rect2_width * rect2_height

    # 计算并集面积
    union_area = rect1_area + rect2_area - overlap_area

    # 计算IoU
    iou = overlap_area / union_area

    return iou

def average_precision(recalls, precisions):
    """
    计算平均精度（AP）
    """
    recalls = np.concatenate(([0.], recalls, [1.]))
    precisions = np.concatenate(([0.], precisions, [0.]))

    for i in range(precisions.size - 1, 0, -1):
        precisions[i - 1] = np.maximum(precisions[i - 1], precisions[i])

    indices = np.where(recalls[1:] != recalls[:-1])[0]
    ap = np.sum((recalls[indices + 1] - recalls[indices]) * precisions[indices + 1])
    return ap

def grouding_format(groundingDic, real_WH):
    type2id = {"person":0, "location":1, "organization":2}
    for img_id, groundingList in groundingDic.items():
        new_dic = {}
        for item in groundingList:
            groundingType = type2id[item[0]]
            new_groundingList = []
            for j, iii in enumerate(item[1:]):
                if j == 0 or j == 2:
                    iii = iii * real_WH[img_id][1]
                else:
                    iii = iii * real_WH[img_id][0]
                new_groundingList.append(iii)
            if groundingType not in new_dic:
                new_dic[groundingType] = [new_groundingList]
            else:
                new_dic[groundingType].append(new_groundingList)
        groundingDic[img_id] = new_dic
    return [item[-1] for item in sorted(groundingDic.items(), key=lambda x:x[0])]

def calculate_map(true_boxes, pred_boxes, iou_threshold=0.5, num_classes=20):
    """
    计算均值平均精度（mAP）
    true_boxes 和 pred_boxes 的格式为
    {
        class_id: [[x1, y1, x2, y2], ...],
        ...
    }
    """
    average_precisions = []

    for class_id in range(num_classes):
        true_boxes_class = true_boxes.get(class_id, [])
        pred_boxes_class = sorted(pred_boxes.get(class_id, []), key=lambda x: x[1], reverse=True)

        if len(true_boxes_class) == 0:
            continue

        true_positives = np.zeros(len(pred_boxes_class))
        false_positives = np.zeros(len(pred_boxes_class))
        detected_boxes = []

        for i, pred_box in enumerate(pred_boxes_class):
            best_iou = 0
            best_gt_idx = -1

            for j, true_box in enumerate(true_boxes_class):
                if j not in detected_boxes:
                    iou = calculate_iou(true_box[0], true_box[1], true_box[2], true_box[3], pred_box[0], pred_box[1], pred_box[2], pred_box[3])
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = j

            if best_iou > iou_threshold:
                true_positives[i] = 1
                detected_boxes.append(best_gt_idx)
            else:
                false_positives[i] = 1

        tp_cumsum = np.cumsum(true_positives)
        fp_cumsum = np.cumsum(false_positives)

        recalls = tp_cumsum / len(true_boxes_class)
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum)

        ap = average_precision(recalls, precisions)
        average_precisions.append(ap)

    map_score = np.mean(average_precisions)
    return map_score

# def grounding_encode_new_ap(grounding_result, video_id):
#     root = f"../../../mnt/second/liujiang/multiTask/img/{video_id}"
#     gold_img_num = len(os.listdir(root)) // 2
#     ground_pred = {}
#     ground_gold = {}
#     real_WH = {}
#     for grounding in grounding_result:
#         grounding = grounding.strip()
#         pred_id = int(grounding.split(" ")[0])
#         if pred_id >= gold_img_num:
#             continue
#         if " is " in grounding:
#             grounding_item = grounding.split(" is ")[1].strip()
#             if " , " in grounding_item:
#                 for grounding_item_item in grounding_item.split(" , "):
#                     if " [entity] " in grounding_item_item:
#                         grounding_entity = grounding_item_item.split(" [entity] ")[1].strip()
#                         coordinateX_p, coordinateY_p, weight_p, height_p = tuple([float(v) for v in grounding_item_item.split(" [entity] ")[0].strip().split(" and ")])
#                         if pred_id not in ground_pred:
#                             ground_pred[pred_id] = [[grounding_entity, coordinateX_p, coordinateY_p, weight_p, height_p]]
#                         else:
#                             ground_pred[pred_id].append([grounding_entity, coordinateX_p, coordinateY_p, weight_p, height_p])
#                     else:
#                         continue
#             else:
#                 if " [entity] " in grounding_item:
#                     grounding_entity = grounding_item.split(" [entity] ")[1].strip()
#                     coordinateX_p, coordinateY_p, weight_p, height_p = tuple([float(re.findall(r'\d+\.\d+', v)[0]) for v in grounding_item.split(" [entity] ")[0].strip().split(" and ")])
#                     if pred_id not in ground_pred:
#                         ground_pred[pred_id] = [[grounding_entity, coordinateX_p, coordinateY_p, weight_p, height_p]]
#                     else:
#                         ground_pred[pred_id].append([grounding_entity, coordinateX_p, coordinateY_p, weight_p, height_p])
#                 else:
#                     continue
#         else:
#             continue
#         if pred_id not in ground_pred:
#             ground_pred[pred_id] = [[grounding_entity, coordinateX_p, coordinateY_p, weight_p, height_p]]
#         else:
#             ground_pred[pred_id].append([grounding_entity, coordinateX_p, coordinateY_p, weight_p, height_p])
#     img_num = 0
#     for filename in os.listdir(root):
#         if ".txt" not in filename:
#             continue
#         image = cv2.imread(f"{root}/{filename.replace('.txt', '.jpg')}")
#         heigh, width, _ = image.shape
#         real_WH[img_num] = [heigh, width]
#         with open(f"{root}/{filename}", "r", encoding="utf-8") as f:
#             for line_id, line in enumerate(f.read().split("\n")):
#                 coordinateX_g = float(line.split(" ")[1])
#                 coordinateY_g = float(line.split(" ")[2])
#                 weight_g = float(line.split(" ")[3])
#                 height_g = float(line.split(" ")[4])
#                 grounding_entity_g = entity_type_dic[line.split(" ")[0]]
#                 if img_num not in ground_gold:
#                     ground_gold[img_num] = [[grounding_entity_g, float(coordinateX_g), float(coordinateY_g), float(weight_g), float(height_g)]]
#                 else:
#                     ground_gold[img_num].append([grounding_entity_g, float(coordinateX_g), float(coordinateY_g), float(weight_g), float(height_g)])
#         img_num += 1   
#     ground_pred = grouding_format(ground_pred, real_WH)
#     ground_gold = grouding_format(ground_gold, real_WH)
#     return ground_pred, ground_gold
    
    
def grounding_encode_new(grounding_result, video_id, lang):
    root = f"../../../mnt/second/liujiang/multiTask/img/{video_id}"
    ground_p, ground_r, ground_c = 0, 0, 0
    gold_img_num = len(os.listdir(root)) // 2
    if not grounding_result:
        for filename in os.listdir(root):
            if ".txt" not in filename:
                continue
            with open(f"{root}/{filename}", "r", encoding="utf-8") as f:
                ground_r += len(f.read().split("\n"))
    else:
        ground_pred = {}
        ground_gold = {}
        for pred_id, grounding in grounding_result.items():
            # pred_id = int("".join(re.findall(r'\d', pred_id)))
            try:
                pred_id = int(pred_id[3:])
            except:
                continue
            if pred_id >= gold_img_num:
                continue
            for grounding_item_item in grounding:
                grounding_entity = grounding_item_item[0]
                if grounding_entity not in t2n[lang]["entity_type_dic"].values():
                    continue
                coorList = grounding_item_item[1:]
                if len(coorList) >= 4:
                    try:
                        for i in range(len(coorList)//4):
                            coordinateX_p = round(float(re.findall(r'\d+(?:\.\d+)?', coorList[i*4])[0]), 2)
                            coordinateY_p = round(float(re.findall(r'\d+(?:\.\d+)?', coorList[i*4+1])[0]), 2)
                            weight_p = round(float(re.findall(r'\d+(?:\.\d+)?', coorList[i*4+2])[0]), 2)
                            height_p = round(float(re.findall(r'\d+(?:\.\d+)?', coorList[i*4+3])[0]), 2)
                    except:
                        continue
                    
                else:
                    continue
                ground_p += 1
                if pred_id not in ground_pred:
                    ground_pred[pred_id] = [[grounding_entity, coordinateX_p, coordinateY_p, weight_p, height_p]]
                else:
                    ground_pred[pred_id].append([grounding_entity, coordinateX_p, coordinateY_p, weight_p, height_p])

        img_num = 0
        for filename in os.listdir(root):
            if ".txt" not in filename:
                continue
            image = cv2.imread(f"{root}/{filename.replace('.txt', '.jpg')}")
            heigh, width, _ = image.shape
            with open(f"{root}/{filename}", "r", encoding="utf-8") as f:
                for line_id, line in enumerate(f.read().split("\n")):
                    ground_r += 1
                    coordinateX_g = round(float(line.split(" ")[1]), 2) * width
                    coordinateY_g = round(float(line.split(" ")[2]), 2) * heigh
                    weight_g = round(float(line.split(" ")[3]), 2) * width
                    height_g = round(float(line.split(" ")[4]), 2) * heigh
                    grounding_entity_g = t2n[lang]["entity_type_dic"][line.split(" ")[0]]
                    if img_num not in ground_gold:
                        ground_gold[img_num] = [[width, heigh], [grounding_entity_g, float(coordinateX_g), float(coordinateY_g), float(weight_g), float(height_g)]]
                    else:
                        ground_gold[img_num].append([grounding_entity_g, float(coordinateX_g), float(coordinateY_g), float(weight_g), float(height_g)])
            img_num += 1   
        for pred_img_id in ground_pred:
            if pred_img_id in ground_gold:
                gold_item = ground_gold[pred_img_id]
                real_width = gold_item[0][0]
                real_heigh = gold_item[0][1]
                for pred_item in ground_pred[pred_img_id]:
                    for gold_item_item in gold_item[1:]:
                        entity_g = gold_item_item[0]
                        coordinateX_g = gold_item_item[1]
                        coordinateY_g = gold_item_item[2]
                        weight_g = gold_item_item[3]
                        height_g = gold_item_item[4]
                        entity_p = pred_item[0]
                        coordinateX_p = pred_item[1] * real_width
                        coordinateY_p = pred_item[2] * real_heigh
                        weight_p = pred_item[3] * real_width
                        height_p = pred_item[4] * real_heigh
                        if entity_g == entity_p:
                            iou = calculate_iou(coordinateX_g, coordinateY_g, weight_g, height_g, coordinateX_p, coordinateY_p, weight_p, height_p)
                            if iou > 0.5:
                                ground_c += 1
                                gold_item.remove(gold_item_item)
                                break                   
    return ground_p, ground_r, ground_c

def grounding_encode(grounding_result, video_id):
    root = f"../../../mnt/second/liujiang/multiTask/img/{video_id}"
    ground_p, ground_r, ground_c = 0, 0, 0
    gold_img_num = len(os.listdir(root)) // 2
    if not grounding_result:
        for filename in os.listdir(root):
            if ".txt" not in filename:
                continue
            with open(f"{root}/{filename}", "r", encoding="utf-8") as f:
                ground_r += len(f.read().split("\n"))

    else:
        grounding_result = grounding_result[0].split(", ")
        ground_pred = {}
        assert len(grounding_result) % 2 == 0
        img_num = 0
        for i in range(len(grounding_result)//2):
            coordinate = grounding_result[i*2].strip()
            weight = grounding_result[i*2+1].strip()
            pred_id = coordinate.split(" ")[0]
            if int(pred_id[0]) >= gold_img_num:
                continue
            coordinate = re.findall(r'\((.*?)\)', coordinate)[0].split(",")
            wh = weight.split(" is ")[-1].split(" and ")
            ground_pred[pred_id] = [float(coordinate[0]), float(coordinate[1]), float(wh[0]), float(wh[1])]
        ground_p += len(ground_pred)
        for filename in os.listdir(root):
            if ".txt" not in filename:
                continue
            pred = []
            image = cv2.imread(f"{root}/{filename.replace('.txt', '.jpg')}")
            heigh, width, _ = image.shape
            with open(f"{root}/{filename}", "r", encoding="utf-8") as f:
                for line_id, line in enumerate(f.read().split("\n")):
                    ground_r += 1
                    coordinateX = float(line.split(" ")[1]) * width
                    coordinateY = float(line.split(" ")[2]) * heigh
                    weight = float(line.split(" ")[3]) * width
                    height = float(line.split(" ")[4]) * heigh
                    gold_id = f"{img_num}{line_id}"
                    if gold_id in ground_pred:
                        item = ground_pred[gold_id]
                        coordinateX_p = item[0] * width
                        coordinateY_p = item[1] * heigh
                        weight_p = item[2] * width
                        height_p = item[3] * heigh
                        pred.append(f"0 {item[0]} {item[1]} {item[2]} {item[3]}")
                        iou = calculate_iou(coordinateX, coordinateY, weight, height, coordinateX_p, coordinateY_p, weight_p, height_p)
                        if iou > 0.5:
                            ground_c += 1
            img_num += 1
    return ground_p, ground_r, ground_c

def muc(predicted_clusters, gold_clusters):      
    """
        predicted_clusters      list(list)       预测实体簇
        gold_clusters           list(list)       标注实体簇
    """  
    pred_edges = set()
    for cluster in predicted_clusters:
        pred_edges |= set(itertools.combinations(cluster, 2))
    gold_edges = set()
    for cluster in gold_clusters:
        gold_edges |= set(itertools.combinations(cluster, 2))
    correct_edges = gold_edges & pred_edges
    return len(correct_edges), len(pred_edges), len(gold_edges)


def b3(predicted_clusters, gold_clusters):
    """
        predicted_clusters      list(list)       预测实体簇
        gold_clusters           list(list)       标注实体簇
    """  
    mentions = set(sum(predicted_clusters, [])) & set(sum(gold_clusters, []))
    precisions = []
    recalls = []
    for mention in mentions:
        mention2predicted_cluster = [x for x in predicted_clusters if mention in x][0]
        mention2gold_cluster = [x for x in gold_clusters if mention in x][0]
        corrects = set(mention2predicted_cluster) & set(mention2gold_cluster)
        precisions.append(len(corrects) / len(mention2predicted_cluster))
        recalls.append(len(corrects) / len(mention2gold_cluster))
    return sum(precisions), sum(recalls), len(precisions), len(recalls)


def ceaf(predicted_clusters, gold_clusters):
    """
        predicted_clusters      list(list)       预测实体簇
        gold_clusters           list(list)       标注实体簇
    """  
    scores = np.zeros((len(predicted_clusters), len(gold_clusters)))
    for j in range(len(gold_clusters)):
        for i in range(len(predicted_clusters)):
            scores[i, j] = len(set(predicted_clusters[i]) & set(gold_clusters[j]))
    indexs = linear_sum_assignment(scores, maximize=True)
    max_correct_mentions = sum(
        [scores[indexs[0][i], indexs[1][i]] for i in range(indexs[0].shape[0])]
    )
    return max_correct_mentions, len(sum(predicted_clusters, [])), len(sum(gold_clusters, []))

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
        if is_print:
            print(res_name)
        with open(f"./{root}/{res_name}", "r", encoding="utf-8") as f:
            result = [json.loads(line) for line in f.read().split("\n") if line]
        for is_loose in [True]:
            for is_big_id in [True]:
                for is_drop_id in [True]:
                    total_p, total_r, total_c = 0, 0, 0
                    total_rel_p, total_rel_r, total_rel_c = 0, 0, 0
                    total_ceaf_c, total_ceaf_p, total_ceaf_r, total_muc_c, total_muc_p, total_muc_r, total_b3_c_p, total_b3_c_r, total_b3_p, total_b3_r = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
                    total_ground_p_list, total_ground_g_list, total_ground_p, total_ground_r, total_ground_c = [], [], 0, 0, 0
                    total_b_e, total_t_e, total_total_ent_r = 0, 0, 0
                    for item in result:
                        pred_result = item["result"]
                        init_doc = item["doc"]
                        link_gold = item["link"]
                        rel_gold = item["relation"]
                        if mode == "t":
                            video_id = ""
                        else:
                            video_id = item["video_id"]
                        # print(pred_result, link_gold, rel_gold, init_doc, video_id, is_loose, is_big_id, is_drop_id, lang)
                        p, r, c, ceaf_p, ceaf_r, ceaf_c, muc_p, muc_r, muc_c, b3_c_p, b3_c_r, b3_p, b3_r, rel_p, rel_r, rel_c, ground_pred, ground_gold, ground_p, ground_r, ground_c, b_e, t_e, total_ent_r = decode_link_rel(pred_result, link_gold, rel_gold, init_doc, video_id, is_loose, is_big_id, is_drop_id, lang)

                        total_b_e += b_e
                        total_t_e += t_e
                        total_total_ent_r += total_ent_r

                        total_p += p 
                        total_r += r
                        total_c += c

                        total_rel_p += rel_p
                        total_rel_r += rel_r
                        total_rel_c += rel_c

                        total_ceaf_c += ceaf_c
                        total_ceaf_p += ceaf_p
                        total_ceaf_r += ceaf_r
                        total_muc_c += muc_c
                        total_muc_p += muc_p
                        total_muc_r += muc_r
                        total_b3_c_p += b3_c_p
                        total_b3_c_r += b3_c_r
                        total_b3_p += b3_p
                        total_b3_r += b3_r

                        total_ground_p += ground_p
                        total_ground_r += ground_r
                        total_ground_c += ground_c

                        total_ground_p_list.extend(ground_pred)
                        total_ground_g_list.extend(ground_gold)

                    ent_precious, ent_recall, ent_f1 = calu_res(total_p, total_r, total_c)
                        
                    rel_precious, rel_recall, rel_f1 = calu_res(total_rel_p, total_rel_r, total_rel_c)

                    ground_precious, ground_recall, ground_f1 = calu_res(total_ground_p, total_ground_r, total_ground_c)

                    # ground_map = np.mean([calculate_map(true_boxes, pred_boxes, 0.5, 3) for true_boxes, pred_boxes in zip(total_ground_p_list, total_ground_g_list)])
                    ground_map = 0
                        

                    ceaf_precious, ceaf_recall, ceaf_f1 = calu_res(total_ceaf_p, total_ceaf_r, total_ceaf_c)
                    muc_precious, muc_recall, muc_f1 = calu_res(total_muc_p, total_muc_r, total_muc_c)
                    b3_precious, b3_recall, b3_f1 = calu_res(total_b3_p, total_b3_r, total_b3_c_p, total_b3_c_r)

                    if total_total_ent_r:
                        ent_b_e, ent_t_e = total_b_e/total_total_ent_r, total_t_e/total_total_ent_r
                    else:
                        ent_b_e, ent_t_e = 0, 0

                    table = pt.PrettyTable([f"{mode}-{lang}", "Precision", "Recall", "F1"])
                    table.add_row(["Entity"] + ["{:3.4f}".format(x) for x in [ent_precious, ent_recall, ent_f1]])
                    table.add_row(["Link(MUC)"] + ["{:3.4f}".format(x) for x in [muc_precious, muc_recall, muc_f1]])
                    table.add_row(["Link(CEAF)"] + ["{:3.4f}".format(x) for x in [ceaf_precious, ceaf_recall, ceaf_f1]])
                    table.add_row(["Link(B3)"] + ["{:3.4f}".format(x) for x in [b3_precious, b3_recall, b3_f1]])
                    table.add_row(["Link(Avg.)"] + ["{:3.4f}".format(x) for x in [np.mean([muc_precious, ceaf_precious, b3_precious]), np.mean([muc_recall, ceaf_recall, b3_recall]), np.mean([muc_f1, ceaf_f1, b3_f1])]])
                    table.add_row(["Relation"] + ["{:3.4f}".format(x) for x in [rel_precious, rel_recall, rel_f1]])
                    table.add_row(["Grounding"] + ["{:3.4f}".format(x) for x in [ground_precious, ground_recall, ground_f1]])
                    table.add_row(["Avg."] + ["{:3.4f}".format(x) for x in [np.mean([ent_precious, np.mean([muc_precious, ceaf_precious, b3_precious]), rel_precious, ground_precious]), np.mean([ent_recall, np.mean([muc_recall, ceaf_recall, b3_recall]), rel_recall, ground_recall]), np.mean([ent_f1, np.mean([muc_f1, ceaf_f1, b3_f1]), rel_f1, ground_f1])]])

                    if is_print:
                        print("{}".format(table))
                    # if not is_min and np.mean([ent_f1, muc_f1, ceaf_f1, b3_f1, rel_f1]) > result_list[file_id][-5]:
                    result_list[seed] = [ent_precious, ent_recall, ent_f1, muc_precious, muc_recall, muc_f1, ceaf_precious, ceaf_recall, ceaf_f1, b3_precious, b3_recall, b3_f1, np.mean([muc_precious, ceaf_precious, b3_precious]), np.mean([muc_recall, ceaf_recall, b3_recall]), np.mean([muc_f1, ceaf_f1, b3_f1]), rel_precious, rel_recall, rel_f1, ground_precious, ground_recall, ground_f1, ground_map, np.mean([ent_precious, np.mean([muc_precious, ceaf_precious, b3_precious]), rel_precious, ground_precious]), np.mean([ent_recall, np.mean([muc_recall, ceaf_recall, b3_recall]), rel_recall, ground_recall]), np.mean([ent_f1, np.mean([muc_f1, ceaf_f1, b3_f1]), rel_f1, ground_f1]), ent_b_e, ent_t_e, 0, 0]

    return result_list

def res(res_list):
    return round(sum(res_list) / len(res_list) * 100, 2), round(np.std(res_list) * 100, 2)

def get_avg_f1(f1_list):
    r = []
    for f1, res in f1_list:
        r.append(f1)
    return sum(r) / len(r)
