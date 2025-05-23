import json
import torch
from torch.utils.data import Dataset
# from torch.nn.utils.rnn import pad_sequence
import numpy as np
# import prettytable as pt
import os
from tqdm import tqdm
os.environ["TOKENIZERS_PARALLELISM"] = "false"
mnre_vgDic = {"train":{}, "val":{}, "test":{}}
for name in ["train", "val", "test"]:
    for filename in os.listdir(f"../../../mnt/third/liujiang/multiTask/data/img_vg/{name}/crops"):
        index = int(filename.split("_")[0])
        if index not in mnre_vgDic[name]:
            mnre_vgDic[name][index] = [filename]
        else:
            mnre_vgDic[name][index].append(filename)
    
with open("./data/type2nature_all.json", "r", encoding="utf-8") as f:
    t2n = json.loads(f.read())

class MyDataset(Dataset):
    def __init__(self, src_list, src_mask_list, tgt_list, tgt_mask_list, video_atts_list, video_hidden_list, video_query_list, relation, links, test_list, video_id_list, docs):
        self.src_list = src_list
        self.src_mask_list = src_mask_list
        self.tgt_list = tgt_list
        self.tgt_mask_list = tgt_mask_list
        self.video_atts_list = video_atts_list
        self.video_hidden_list = video_hidden_list
        self.video_query_list = video_query_list
        self.relation = relation
        self.links = links
        self.test_list = test_list
        self.video_id_list = video_id_list
        self.docs = docs

    def __getitem__(self, item):
        return torch.LongTensor(self.src_list[item]), \
               torch.LongTensor(self.src_mask_list[item]), \
               torch.LongTensor(self.tgt_list[item]), \
               torch.LongTensor(self.tgt_mask_list[item]), \
               torch.LongTensor(self.video_atts_list[item]), \
               torch.FloatTensor(self.video_hidden_list[item]), \
               torch.FloatTensor(self.video_query_list[item]), \
               self.relation[item], \
               self.links[item], \
               self.test_list[item], \
               self.video_id_list[item], \
               self.docs[item]

    
    def __len__(self):
        return len(self.src_list)
    
class SingleTaskDataset(Dataset):
    def __init__(self, src_list, src_mask_list, tgt_list, tgt_mask_list, list_p1s, list_p2s, list_p3s, list_p4s, weights, tokens, tokens_att, hs, ts, phrase, phrase_att, relation, links, test_list, video_id_list, docs):
        self.src_list = src_list
        self.src_mask_list = src_mask_list
        self.tgt_list = tgt_list
        self.tgt_mask_list = tgt_mask_list
        self.list_p1s = list_p1s
        self.list_p2s = list_p2s
        self.list_p3s = list_p3s
        self.list_p4s = list_p4s
        self.weights = weights
        self.tokens = tokens
        self.tokens_att = tokens_att
        self.hs = hs
        self.ts = ts
        self.phrase = phrase
        self.phrase_att = phrase_att
        self.relation = relation
        self.links = links
        self.test_list = test_list
        self.video_id_list = video_id_list
        self.docs = docs

    def __getitem__(self, item):
        return torch.LongTensor(self.src_list[item]), \
               torch.LongTensor(self.src_mask_list[item]), \
               torch.LongTensor(self.tgt_list[item]), \
               torch.LongTensor(self.tgt_mask_list[item]), \
               torch.FloatTensor(self.list_p1s[item]), \
               torch.FloatTensor(self.list_p2s[item]), \
               torch.FloatTensor(self.list_p3s[item]), \
               torch.FloatTensor(self.list_p4s[item]), \
               torch.FloatTensor(self.weights[item]), \
               torch.LongTensor(self.tokens[item]), \
               torch.LongTensor(self.tokens_att[item]), \
               torch.LongTensor(self.hs[item]), \
               torch.LongTensor(self.ts[item]), \
               torch.LongTensor(self.phrase[item]), \
               torch.LongTensor(self.phrase_att[item]), \
               self.relation[item], \
               self.links[item], \
               self.test_list[item], \
               self.video_id_list[item], \
               self.docs[item]

    
    def __len__(self):
        return len(self.src_list)
    
class MyDataset(Dataset):
    def __init__(self, src_list, src_mask_list, tgt_list, tgt_mask_list, video_atts_list, video_hidden_list, video_query_list, vg_list, relation, links, test_list, video_id_list, docs):
        self.src_list = src_list
        self.src_mask_list = src_mask_list
        self.tgt_list = tgt_list
        self.tgt_mask_list = tgt_mask_list
        self.video_atts_list = video_atts_list
        self.video_hidden_list = video_hidden_list
        self.video_query_list = video_query_list
        self.vg_list = vg_list
        self.relation = relation
        self.links = links
        self.test_list = test_list
        self.video_id_list = video_id_list
        self.docs = docs

    def __getitem__(self, item):
        return torch.LongTensor(self.src_list[item]), \
               torch.LongTensor(self.src_mask_list[item]), \
               torch.LongTensor(self.tgt_list[item]), \
               torch.LongTensor(self.tgt_mask_list[item]), \
               torch.LongTensor(self.video_atts_list[item]), \
               torch.FloatTensor(self.video_hidden_list[item]), \
               torch.FloatTensor(self.video_query_list[item]), \
               self.vg_list[item], \
               self.relation[item], \
               self.links[item], \
               self.test_list[item], \
               self.video_id_list[item], \
               self.docs[item]

    
    def __len__(self):
        return len(self.src_list)

def dataPro_med_umie_zh(rel, chain, v_id, lang):
    output = ""
    # chain_dic = {}
    # rel_list = []
    chain_temp = []
    for chain_id, v in chain.items():
        # s = ", ".join(["\"" + item["text"] + "\"" for item in v["link"]])
        chain_temp.append(f'({chain_id}， {"，".join(list(set([item["text"] for item in v["link"]])))}，{t2n[lang]["entity_type_dic"][v["type"]]})')
    output += "".join(chain_temp)
        # chain_dic[chain_id] = [item["text"] for item in v["link"]]
    rel_temp = []
    for r in rel:
        rel_temp.append(f'({r["link1"]}，{t2n[lang]["relation_type_dic"][r["type"]]}，{r["link2"]})')
    output += ''.join(rel_temp)
        # output += f'relation_dic["{t2n[lang]["relation_type_dic"][k]}"] = {json.dumps(v, ensure_ascii=False)}'
        # output += "\n"
        # rel_list.append([r["link1"], r["link2"], relation_type_dic[r["type"]]])
    img_list = []
    for img in os.listdir(f"../../../../mnt/second/liujiang/multiTask/img/{v_id}"):
        if "txt" not in img:
            img_list.append(int(img.split(".")[0]))
    img_list = sorted(img_list)
    img_temp = []
    for i, img in enumerate(img_list):
        with open(f"../../../../mnt/second/liujiang/multiTask/img/{v_id}/{img}.txt", "r", encoding="utf-8") as f:
            new = []
            for line in f.read().split("\n"):
                new_line = []
                for item in line.split(" "):
                    if item in t2n[lang]["entity_type_dic"]:
                        new_line.append(t2n[lang]["entity_type_dic"][item])
                    else:
                        new_line.append(str(round(float(item), 1)))
            img_temp.append(f"(Img{i}，{'，'.join(new_line[1:])}，{new_line[0]})")
    output += ''.join(img_temp) + "。"
    return output

def dataPro_med_nl_zh(rel, chain, v_id, lang):
    output = ""
    # chain_dic = {}
    # rel_list = []
    chain_temp = []
    for chain_id, v in chain.items():
        # s = ", ".join(["\"" + item["text"] + "\"" for item in v["link"]])
        chain_temp.append(f'({chain_id}， {"，".join(list(set([item["text"] for item in v["link"]])))}，{t2n[lang]["entity_type_dic"][v["type"]]})')
    output += "".join(chain_temp)
        # chain_dic[chain_id] = [item["text"] for item in v["link"]]
    rel_temp = []
    for r in rel:
        rel_temp.append(f'({r["link1"]}，{t2n[lang]["relation_type_dic"][r["type"]]}，{r["link2"]})')
    output += ''.join(rel_temp)
        # output += f'relation_dic["{t2n[lang]["relation_type_dic"][k]}"] = {json.dumps(v, ensure_ascii=False)}'
        # output += "\n"
        # rel_list.append([r["link1"], r["link2"], relation_type_dic[r["type"]]])
    img_list = []
    for img in os.listdir(f"../../../../mnt/second/liujiang/multiTask/img/{v_id}"):
        if "txt" not in img:
            img_list.append(int(img.split(".")[0]))
    img_list = sorted(img_list)
    img_temp = []
    for i, img in enumerate(img_list):
        with open(f"../../../../mnt/second/liujiang/multiTask/img/{v_id}/{img}.txt", "r", encoding="utf-8") as f:
            new = []
            for line in f.read().split("\n"):
                new_line = []
                for item in line.split(" "):
                    if item in t2n[lang]["entity_type_dic"]:
                        new_line.append(t2n[lang]["entity_type_dic"][item])
                    else:
                        new_line.append(str(round(float(item), 1)))
            img_temp.append(f"(Img{i}，{'，'.join(new_line[1:])}，{new_line[0]})")
    output += ''.join(img_temp) + "。"
    return output

def dataPro_m3d_umie_en(rel, chain, v_id, lang):
    output = ""
    # chain_dic = {}
    # rel_list = []
    chain_temp = []
    for chain_id, v in chain.items():
        # s = ", ".join(["\"" + item["text"] + "\"" for item in v["link"]])
        chain_temp.append(f'({chain_id} , {" , ".join(list(set([item["text"] for item in v["link"]])))} , {t2n[lang]["entity_type_dic"][v["type"]]})')
    output += " ".join(chain_temp)
        # chain_dic[chain_id] = [item["text"] for item in v["link"]]
    rel_temp = []
    for r in rel:
        rel_temp.append(f'({r["link1"]} , {t2n[lang]["relation_type_dic"][r["type"]]} , {r["link2"]})')
    output += " " + ' '.join(rel_temp)
        # output += f'relation_dic["{t2n[lang]["relation_type_dic"][k]}"] = {json.dumps(v, ensure_ascii=False)}'
        # output += "\n"
        # rel_list.append([r["link1"], r["link2"], relation_type_dic[r["type"]]])
    img_list = []
    for img in os.listdir(f"../../../../mnt/second/liujiang/multiTask/img/{v_id}"):
        if "txt" not in img:
            img_list.append(int(img.split(".")[0]))
    img_list = sorted(img_list)
    img_temp = []
    for i, img in enumerate(img_list):
        with open(f"../../../../mnt/second/liujiang/multiTask/img/{v_id}/{img}.txt", "r", encoding="utf-8") as f:
            new = []
            for line in f.read().split("\n"):
                new_line = []
                for item in line.split(" "):
                    if item in t2n[lang]["entity_type_dic"]:
                        new_line.append(t2n[lang]["entity_type_dic"][item])
                    else:
                        new_line.append(str(round(float(item), 1)))
            img_temp.append(f"(Img{i} , {' , '.join(new_line[1:])} , {new_line[0]})")
    output += " " + ' '.join(img_temp) + " ."
    return output

def dataPro_m3d_nl_en(rel, chain, v_id, lang):
    output = ""
    # chain_dic = {}
    # rel_list = []
    chain_temp = []
    for chain_id, v in chain.items():
        # s = ", ".join(["\"" + item["text"] + "\"" for item in v["link"]])
        chain_temp.append(f'({chain_id} , {" , ".join(list(set([item["text"] for item in v["link"]])))} , {t2n[lang]["entity_type_dic"][v["type"]]})')
    output += " ".join(chain_temp)
        # chain_dic[chain_id] = [item["text"] for item in v["link"]]
    rel_temp = []
    for r in rel:
        rel_temp.append(f'({r["link1"]} , {t2n[lang]["relation_type_dic"][r["type"]]} , {r["link2"]})')
    output += " " + ' '.join(rel_temp)
        # output += f'relation_dic["{t2n[lang]["relation_type_dic"][k]}"] = {json.dumps(v, ensure_ascii=False)}'
        # output += "\n"
        # rel_list.append([r["link1"], r["link2"], relation_type_dic[r["type"]]])
    img_list = []
    for img in os.listdir(f"../../../../mnt/second/liujiang/multiTask/img/{v_id}"):
        if "txt" not in img:
            img_list.append(int(img.split(".")[0]))
    img_list = sorted(img_list)
    img_temp = []
    for i, img in enumerate(img_list):
        with open(f"../../../../mnt/second/liujiang/multiTask/img/{v_id}/{img}.txt", "r", encoding="utf-8") as f:
            new = []
            for line in f.read().split("\n"):
                new_line = []
                for item in line.split(" "):
                    if item in t2n[lang]["entity_type_dic"]:
                        new_line.append(t2n[lang]["entity_type_dic"][item])
                    else:
                        new_line.append(str(round(float(item), 1)))
            img_temp.append(f"(Img{i} , {' , '.join(new_line[1:])} , {new_line[0]})")
    output += " " + ' '.join(img_temp) + " ."
    return output

def dataPro_m3d_code(rel, chain, v_id, lang):
    output = ""
    # chain_dic = {}
    # rel_list = []
    for chain_id, v in chain.items():
        # s = ", ".join(["\"" + item["text"] + "\"" for item in v["link"]])
        output += f'chain_dic["{chain_id}"] = {json.dumps([list(set([item["text"] for item in v["link"]])), t2n[lang]["entity_type_dic"][v["type"]]], ensure_ascii=False)}'
        output += "\n"
        # chain_dic[chain_id] = [item["text"] for item in v["link"]]
    r_dic = {}
    for r in rel:
        t = r["type"]
        if t not in r_dic:
            r_dic[t] = [[r["link1"], r["link2"]]]
        else:
            r_dic[t].append([r["link1"], r["link2"]])
        # ss = ", ".join(["\"" + r["link1"] + "\"", "\"" + r["link2"] + "\"", "\"" + relation_type_dic[r["type"]] + "\""])
    for k, v in r_dic.items():
        output += f'relation_dic["{t2n[lang]["relation_type_dic"][k]}"] = {json.dumps(v, ensure_ascii=False)}'
        output += "\n"
        # rel_list.append([r["link1"], r["link2"], relation_type_dic[r["type"]]])
    img_list = []
    for img in os.listdir(f"../../../../mnt/second/liujiang/multiTask/img/{v_id}"):
        if "txt" not in img:
            img_list.append(int(img.split(".")[0]))
    img_list = sorted(img_list)
    for i, img in enumerate(img_list):
        with open(f"../../../../mnt/second/liujiang/multiTask/img/{v_id}/{img}.txt", "r", encoding="utf-8") as f:
            new = []
            for line in f.read().split("\n"):
                new_line = []
                for item in line.split(" "):
                    if item in t2n[lang]["entity_type_dic"]:
                        new_line.append(t2n[lang]["entity_type_dic"][item])
                    else:
                        new_line.append(str(round(float(item), 1)))
                new.append(new_line)
            output += f'grounding_dic["Img{i}"] = {json.dumps(new, ensure_ascii=False)}'
            output += "\n"
    return output

def process_bert(data, tokenizer, myconfig, name="train"):
    src_list = []
    src_mask_list = []
    tgt_list = []
    tgt_mask_list = []
    link_list = []
    relation_list = []
    test_list = []
    video_id_list = []
    docs = []
    data_max_len = 0
    video_atts_list = []
    video_hidden_list = []
    video_query_list = []
    vg_list = []
    for item in tqdm(data):
        prompt_1, prompt_2 = "", ""
        vg_path = []
        if myconfig.dataset == "m3d_en":
            with open("./data/graph_m3d.json", "r", encoding="utf-8") as f:
                graph = json.loads(f.read())
            text = item["doc"]
            video_id = item["video_id"]
            relation = item["relation"]
            entityLink = item["entityLink"]
            video_path = "../../../mnt/nlp/liujiang/multiTask/feature/video"
            if myconfig.template == "ours_code":
                prompt_1 = """def information_extraction(input_text, scene_graph, input_image):
                \"\"\"
                    first , Extract entity chains from input_text .
                    second , Extract entity chains relation based on entity chains .
                    three , Inferring the visual area coordinate and type in the image based on the scene graph .
                \"\"\"
                input_text = \"""" + text + """\"
                scene_graph = """ + json.dumps({f"Img{i}": item[-1] for i, item in enumerate(sorted(graph["en"][video_id].items(), key=lambda x: int(x[0].split('.')[0])))} if video_id in graph["en"] else {}, ensure_ascii=False) + """
                input_image = """

                prompt_2 = """\nchain_dic = {}
                relation_dic = {}
                grounding_dic = {}
                # extacted entity chains , relations and visual area"""

                output = dataPro_m3d_code(relation, entityLink, video_id, "en") 
                
            elif myconfig.template == "ours_nl":
                prompt_1 = f"""Please extract all entity chains , relations and visual areas and their types from the given text and image . The output format of entity chain is (entity chain ID , entity1 , entity2 , ... , entity chain type) . The output format of relation is (subject entity chain ID , relation type , object entity chain ID) . The output format of visual area is (image ID , center horizontal coordinate , center vertical coordinate , height , width , visual area type) . \n Candidate entity chain types : {' , '.join(t2n["en"]['entity_type_dic'].values())} . \n Candidate relation types : {' , '.join(t2n["en"]['relation_type_dic'].values())} . \n Candidate visual area types : person , location , organization . \n Input Text and Image : {text} """ + " , ".join([f"image {i} contains the relations {' , '.join([f'{ii[0]} {ii[-1]} {ii[1]}' for ii in item[-1]])}" for i, item in enumerate(sorted(graph["en"][video_id].items(), key=lambda x: int(x[0].split('.')[0])))]) if video_id in graph["en"] else " , ".join([])

                output = dataPro_m3d_nl_en(relation, entityLink, video_id, "en")
            elif myconfig.template == "reamo":
                prompt_1 = f"""Please extract all entity chains , relations and visual areas and their types from the given text and image . The output format of entity chain is (entity chain ID , entity1 , entity2 , ... , entity chain type) . The output format of relation is (subject entity chain ID , relation type , object entity chain ID) . The output format of visual area is (image ID , center horizontal coordinate , center vertical coordinate , height , width , visual area type) . \n Candidate entity chain types : {' , '.join(t2n["en"]['entity_type_dic'].values())} . \n Candidate relation types : {' , '.join(t2n["en"]['relation_type_dic'].values())} . \n Candidate visual area types : person , location , organization . \n Input Text and Image : {text}"""

                output = dataPro_m3d_nl_en(relation, entityLink, video_id, "en")
            elif myconfig.template == "umie":
                print("umie model dose not work in m3d dataset!!!")
                exit()
            elif myconfig.template == "feature":
                prompt_1 = """def information_extraction(input_text, input_image):
                \"\"\"
                    first , Extract entity chains from input_text .
                    second , Extract entity chains relation based on entity chains .
                    three , Inferring the visual area coordinate and type in the image .
                \"\"\"
                input_text = \"""" + text + """\"
                input_image = """

                prompt_2 = """\nchain_dic = {}
                relation_dic = {}
                grounding_dic = {}
                # extacted entity chains , relations and visual area"""

                output = dataPro_m3d_code(relation, entityLink, video_id, "en") 
        elif myconfig.dataset == "m3d_zh":
            with open("./data/graph_m3d.json", "r", encoding="utf-8") as f:
                graph = json.loads(f.read())
            text = item["doc"]
            video_id = item["video_id"]
            relation = item["relation"]
            entityLink = item["entityLink"]
            video_path = "../../../mnt/nlp/liujiang/multiTask/feature/video"
            if myconfig.template == "ours_code":
                prompt_1 = """def information_extraction(input_text, scene_graph, input_image):
                \"\"\"
                    第一，从inint_text中抽取实体链。
                    第二，基于实体链抽取实体链关系。
                    第三，基于场景图从图像中推理视觉区域坐标和类型。
                \"\"\"
                input_text = \"""" + text + """\"
                scene_graph = """ + json.dumps({f"Img{i}": item[-1] for i, item in enumerate(sorted(graph["zh"][video_id].items(), key=lambda x: int(x[0].split('.')[0])))} if video_id in graph["zh"] else {}, ensure_ascii=False) + """ 
                input_image = """ 

                prompt_2 = """\nchain_dic = {}
                relation_dic = {}
                grounding_dic = {}
                # 抽取实体链，关系和视觉区域"""

                output = dataPro_m3d_code(relation, entityLink, video_id, "zh") 
                
            elif myconfig.template == "ours_nl":
                prompt_1 = f"""请从给定的文本和图像中抽取所有实体链，关系和视觉区域以及它们的类型。实体链的输出格式为(实体链ID，实体1，实体2，...，实体链类型)。关系输出格式为(主体实体链ID，关系类型，客体实体链ID)。视觉区域输出格式为(图像ID，中心横坐标，中心纵坐标，宽度，高度，视觉区域类型)。\n候选实体链类型：{', '.join(t2n["zh"]['entity_type_dic'].values())}。\n候选关系类型：{', '.join(t2n["zh"]['relation_type_dic'].values())}。\n候选视觉区域类型：人物，地点，组织。\n输入文本和图像：{text} """ + "，".join([f"图像{i}的关系有{'，'.join([f'{ii[0]}{ii[-1]}{ii[1]}' for ii in item[-1]])}" for i, item in enumerate(sorted(graph["zh"][video_id].items(), key=lambda x: int(x[0].split('.')[0])))]) if video_id in graph["zh"] else "，".join([])

                output = dataPro_med_nl_zh(relation, entityLink, video_id, "zh")
            elif myconfig.template == "reamo":
                prompt_1 = f"""请从给定的文本和图像中抽取所有实体链，关系和视觉区域以及它们的类型。实体链的输出格式为(实体链ID，实体1，实体2，...，实体链类型)。关系输出格式为(主体实体链ID，关系类型，客体实体链ID)。视觉区域输出格式为(图像ID，中心横坐标，中心纵坐标，宽度，高度，视觉区域类型)。\n候选实体链类型：{', '.join(t2n["zh"]['entity_type_dic'].values())}。\n候选关系类型：{', '.join(t2n["zh"]['relation_type_dic'].values())}。\n候选视觉区域类型：人物，地点，组织。\n输入文本和图像：{text}"""

                output = dataPro_med_nl_zh(relation, entityLink, video_id, "zh")
            elif myconfig.template == "umie":
                print("umie model dose not work in m3d dataset!!!")
                exit()
            elif myconfig.template == "feature":
                prompt_1 = """def information_extraction(input_text, input_image):
                \"\"\"
                    第一，从inint_text中抽取实体链。
                    第二，基于实体链抽取实体链关系。
                    第三，从图像中推理视觉区域坐标和类型。
                \"\"\"
                input_text = \"""" + text + """\"
                input_image = """ 

                prompt_2 = """\nchain_dic = {}
                relation_dic = {}
                grounding_dic = {}
                # 抽取实体链，关系和视觉区域"""

                output = dataPro_m3d_code(relation, entityLink, video_id, "zh") 
        elif myconfig.dataset == "mnre":
            with open("./data/graph_mnre.json", "r", encoding="utf-8") as f:
                graph = json.loads(f.read())
            with open("./data/entity2type.json", "r", encoding="utf-8") as f:
                ent2type = json.loads(f.read())
            text = " ".join(item["token"])
            video_id = ".".join(item["img_id"].split(".")[:-1])
            h, t = item["h"]["name"], item["t"]["name"]
            relation = t2n["mnre_code"][item["relation"]]
            vg_path = item["vg"]
            entityLink = []
            video_path = f"../../../mnt/nlp/liujiang/multiTask/mnre_feature/train"
            if myconfig.template == "ours_code":

                prompt_1 = """def information_extraction(input_text, scene_graph, input_image):
                \"\"\"
                    Extract entity relation based on entity .
                \"\"\"
                input_text = \"""" + text + """\"
                scene_graph = """ + json.dumps({f"Img{i}": item[-1] for i, item in enumerate(sorted({'1':graph[name][item['img_id']]}.items(), key=lambda x: int(x[0].split('.')[0])))} if item["img_id"] in graph[name] else {}, ensure_ascii=False) + """
                input_image = """

                prompt_2 = """\nchain_dic = {}
                relation_dic = {}
                # extacted relations\n""" + f"""chain_dic[\"L0\"] = [\"{h}\", \"{ent2type[h]}\"]\nchain_dic[\"L1\"] = [\"{t}\", \"{ent2type[t]}\"]"""

                output = f"relation_dic[\"{relation}\"] = [\"L0\", \"L1\"]" 
                
            elif myconfig.template == "ours_nl":
                prompt_1 = f"""Please extract all relations from the given text and image . The output format of relation is ({h} , {ent2type[h]} , relation type , {t} , {ent2type[t]}) . \n Candidate relation types : {' , '.join(t2n["mnre_code"].values())} . \n Input Text and Image : {text} """ + (" , ".join([f"image {i} contains the relations {' , '.join([f'{ii[0]} {ii[-1]} {ii[1]}' for ii in item[-1]])}" for i, item in enumerate(sorted({'1':graph[name][item['img_id']]}.items(), key=lambda x: int(x[0].split('.')[0])))]) if item['img_id'] in graph[name] else " , ".join([]))

                output = f"({h} , {ent2type[h]} , {relation} , {t} , {ent2type[t]})"
            elif myconfig.template == "reamo":
                prompt_1 = f"""Please extract all relations from the given text and image . The output format of relation is ({h} , {ent2type[h]} , relation type , {t} , {ent2type[t]}) . \n Candidate relation types : {' , '.join(t2n["mnre_code"].values())} . \n Input Text and Image : {text} """

                output = f"({h} , {ent2type[h]} , {relation} , {t} , {ent2type[t]})"
            elif myconfig.template == "umie":
                prompt_1 = f"""Please extract the following relation between {ent2type[h]} entity {h} and {ent2type[t]} entity {t} : {' , '.join(t2n["mnre_code"].values())} . {text}"""

                output = f"{h} , {relation} , {t}"
            elif myconfig.template == "feature":
                prompt_1 = """def information_extraction(input_text, input_image):
                \"\"\"
                    Extract entity relation based on entity .
                \"\"\"
                input_text = \"""" + text + """\"
                input_image = """

                prompt_2 = """\nchain_dic = {}
                relation_dic = {}
                # extacted relations\n""" + f"""chain_dic[\"L0\"] = [\"{h}\", \"{ent2type[h]}\"]\nchain_dic[\"L1\"] = [\"{t}\", \"{ent2type[t]}\"]"""

                output = f"relation_dic[\"{relation}\"] = [\"L0\", \"L1\"]"
        elif myconfig.dataset == "mner15" or myconfig.dataset == "mner17":
            with open(f"./data/graph_{myconfig.dataset[4:]}.json", "r", encoding="utf-8") as f:
                graph = json.loads(f.read())
            new_words = []
            for word in item["tokens"]:
                if "http" not in word:
                    new_words.append(word)
            text = " ".join(new_words)
            video_id = item["img"]
            relation = {}
            output = []
            entityLink = item["entity"]
            video_path = "../../../mnt/nlp/liujiang/multiTask/mner_feature"

            if myconfig.template == "ours_code":
                prompt_1 = """def information_extraction(input_text, scene_graph, input_image):
                \"\"\"
                    Extract entity from input_text .
                \"\"\"
                input_text = \"""" + text + """\"
                scene_graph = """ + json.dumps({f"Img{i}": item[-1] for i, item in enumerate(sorted({"1":graph[f'{video_id}.jpg'][""]}.items(), key=lambda x: int(x[0].split('.')[0])))} if f'{video_id}.jpg' in graph else {}, ensure_ascii=False) + """
                input_image = """

                prompt_2 = """\nrelation_dic = {}
                # extacted entities"""
                
                num = 0
                for ent_text, ent_type in entityLink.items():
                    if ent_type == "OTHER":
                        ent_type = "MISC"
                    ent_type = t2n["mner"][ent_type]
                    output.append(f"chain_dic[\"L{num}\"] = [[\"{ent_text}\"], \"{ent_type}\"]")
                    num += 1
                output = "\n".join(output)
                
            elif myconfig.template == "ours_nl":
                prompt_1 = f"""Please extract all entities and their types from the given text and image . The output format of entity is (entity , ... , entity type) . \n Candidate entity chain types : {' , '.join(t2n["mner"].values())} . \n Input Text and Image : {text} """ + (" , ".join([f"image {i} contains the relations {' , '.join([f'{ii[0]} {ii[-1]} {ii[1]}' for ii in item[-1]])}" for i, item in enumerate(sorted({"1":graph[f'{video_id}.jpg'][""]}.items(), key=lambda x: int(x[0].split('.')[0])))]) if f'{video_id}.jpg' in graph else " , ".join([]))

                num = 0
                for ent_text, ent_type in entityLink.items():
                    if ent_type == "OTHER":
                        ent_type = "MISC"
                    ent_type = t2n["mner"][ent_type]
                    output.append(f"({ent_text} , {ent_type})")
                    num += 1
                output = " ".join(output)

            elif myconfig.template == "reamo":
                prompt_1 = f"""Please extract all entities , and their types from the given text and image . The output format of entity is (entity , ... , entity type) . \n Candidate entity chain types : {' , '.join(t2n["mner"].values())} . \n Input Text and Image : {text} """

                num = 0
                for ent_text, ent_type in entityLink.items():
                    if ent_type == "OTHER":
                        ent_type = "MISC"
                    ent_type = t2n["mner"][ent_type]
                    output.append(f"({ent_text} , {ent_type})")
                    num += 1
                output = " ".join(output)

            elif myconfig.template == "umie":
                prompt_1 = f"Please extract the following entity type : {' , '.join(t2n['mner'].values())} . {text}"

                num = 0
                for ent_text, ent_type in entityLink.items():
                    if ent_type == "OTHER":
                        ent_type = "MISC"
                    ent_type = t2n["mner"][ent_type]
                    output.append(f"({ent_type} , {ent_text})")
                    num += 1
                output = " <spot> ".join(output)
            elif myconfig.template == "feature":
                prompt_1 = """def information_extraction(input_text, input_image):
                \"\"\"
                    Extract entity from input_text .
                \"\"\"
                input_text = \"""" + text + """\"
                input_image = """

                prompt_2 = """\nrelation_dic = {}
                # extacted entities"""
                
                num = 0
                for ent_text, ent_type in entityLink.items():
                    if ent_type == "OTHER":
                        ent_type = "MISC"
                    ent_type = t2n["mner"][ent_type]
                    output.append(f"chain_dic[\"L{num}\"] = [[\"{ent_text}\"], \"{ent_type}\"]")
                    num += 1
                output = "\n".join(output)
        if name == "train":
            src = prompt_1.strip()
            result = prompt_2 + "\n" + output.strip() + " </s>"
        else:
            src = prompt_1.strip()
            result = prompt_2 + "\n"
        if video_id != "none": 
            video_atts = torch.load(f"{video_path}/{video_id}_frame_atts.pth").to('cpu').long()
            video_hidden = torch.load(f"{video_path}/{video_id}_frame_hidden_state.pth").to('cpu').float()
            video_query = torch.load(f"{video_path}/{video_id}_video_query_tokens.pth").to('cpu').float()
        else:
            video_atts = torch.zeros((1, 1)).long()
            video_hidden = torch.zeros((1, 1, 768)).float()
            video_query = torch.zeros((1, 1, 768)).float()
        video_atts_list.append(video_atts)
        video_hidden_list.append(video_hidden)
        video_query_list.append(video_query)
        test_list.append(src)
        src_item = tokenizer(src)
        tgt_item = tokenizer(result)
        src_ids = src_item.input_ids
        tgt_ids = tgt_item.input_ids
        src_mask = src_item.attention_mask
        tgt_mask = tgt_item.attention_mask
        src_len = src_ids.__len__()
        tgt_len = tgt_ids.__len__()
        if src_len + tgt_len > data_max_len:
            data_max_len = src_len + tgt_len
        src_list.append(src_ids)
        src_mask_list.append(src_mask)
        tgt_list.append(tgt_ids)
        tgt_mask_list.append(tgt_mask)
        link_list.append(entityLink)
        relation_list.append(relation)
        video_id_list.append(video_id)
        docs.append(text)
        vg_list.append(vg_path)
    print(data_max_len)
    return src_list, src_mask_list, tgt_list, tgt_mask_list, video_atts_list, video_hidden_list, video_query_list, vg_list, relation_list, link_list, test_list, video_id_list, docs

import re
def txt2json(lines, mode):
    res = []
    img_id = []
    for ind, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        pattern = r"'([^']+)'(?=[,|\]])"
        matches = re.findall(pattern, line)
        for m in matches:
            if "," in m:
                continue
            if "\"" in m:
                newM = m.replace("\"", "\\\"")
                line = line.replace(f"'{m}'", f"'{newM}'")
        pattern = r'"([^"]+)"(?=[,|\]])'
        matches = re.findall(pattern, line)
        for m in matches:
            if "," in m:
                continue
            newM = m.replace("'", "\\\\'")
            line = line.replace(f"\"{m}\"", f"\'{newM}\'")
        line = line.replace("'", '"').replace("\\\\\"", "'")
        line = json.loads(line)
        img = line["img_id"]
        if img not in img_id:
            img_id.append(img)
        line["vg"] = mnre_vgDic[mode][len(img_id)-1]
        res.append(line)
    return res

def txt2json_mner(data):
    findata = []
    temp = []
    sent = []
    ents = {}
    img = ""
    for line in data:
        line = line.strip()
        if not line:
            if temp:
                ents[" ".join(temp[1:])] = temp[0]
                temp = []
            findata.append({
                "tokens": sent,
                "entity": ents,
                "img": img
            })
            sent = []
            ents = {}
            continue
        if "IMGID" in line:
            img = line.split(":")[-1]
            continue
        word = line.split("	")[0]
        label = line.split("	")[-1]
        if "-" in label:
            ent_t = label.split("-")[-1]
            label = label.split("-")[0]
            if ent_t == "OTHER":
                ent_t = "MISC"
        else:
            ent_t = ""
        sent.append(word)
        if label == "B":
            if temp:
                ents[" ".join(temp[1:])] = temp[0]
                temp = []
            temp.extend([ent_t, word])
        elif label == "I":
            if temp:
                temp.append(word)
        else:
            if temp:
                ents[" ".join(temp[1:])] = temp[0]
                temp = []
    if temp:
        ents[" ".join(temp[1:])] = temp[0]
        temp = []
    findata.append({
        "tokens": sent,
        "entity": ents,
        "img": img
    })
    return findata

def load_data_bert(tokenizer, myconfig, model):
    if myconfig.dataset == "m3d_en" or myconfig.dataset == "m3d_zh":
        lang = myconfig.dataset.split("_")[-1]
        with open(f'./data/m3d/{lang}/train_{lang}.json', 'r', encoding='utf-8') as f:
            train_data = json.loads(f.read())
        with open(f'./data/m3d/{lang}/test_{lang}.json', 'r', encoding='utf-8') as f:
            test_data = json.loads(f.read())
    elif myconfig.dataset == "mnre":
        with open(f'./data/mnre_txt/mnre_train.txt', 'r', encoding='utf-8') as f:
            train_data = txt2json(f.read().split("\n"), "train")
        with open(f'./data/mnre_txt/mnre_test.txt', 'r', encoding='utf-8') as f:
            test_data = txt2json(f.read().split("\n"), "test")
    # elif myconfig.dataset == "mnre_v1":
    #     with open(f'./data/mnre_v1/ours_train.txt', 'r', encoding='utf-8') as f:
    #         train_data = [txt2json(line) for line in f.read().split("\n") if line.strip()]
    #     with open(f'./data/mnre_v1/ours_test.txt', 'r', encoding='utf-8') as f:
    #         test_data = [txt2json(line) for line in f.read().split("\n") if line.strip()]
    elif myconfig.dataset == "mner15" or myconfig.dataset == "mner17":
        with open(f'./data/twitter20{myconfig.dataset[4:]}/train.txt', 'r', encoding='utf-8') as f:
            train_data = txt2json_mner(f.read().split("\n"))
        with open(f'./data/twitter20{myconfig.dataset[4:]}/test.txt', 'r', encoding='utf-8') as f:
            test_data = txt2json_mner(f.read().split("\n"))
    train_dataset = MyDataset(*process_bert(train_data, tokenizer, myconfig, name="train"))
    test_dataset = MyDataset(*process_bert(test_data, tokenizer, myconfig, name="test"))
    return train_dataset, test_dataset