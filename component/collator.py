from typing import Any, Dict, List
import torch
from torch.nn.utils.rnn import pad_sequence
import numpy as np

class MNERCollator(object):
    def __init__(self, tokenizer, max_seq_length):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.pad_token_id = tokenizer.pad_token_id

    def padding(self, data, ml, pad_vlaue):
        return torch.stack([torch.cat([item, torch.LongTensor([pad_vlaue]*(ml - item.shape[0]))], dim=0) for item in data], dim=0)
    
    def __call__(self, batch: List[Any]) -> Dict[str, Any]:
        src_list = []
        src_mask_list = []
        tgt_list = []
        tgt_mask_list = []
        relation_list = []
        link_list = []
        input_texts = []
        video_id_list = []
        video_atts_list = []
        video_hidden_list = []
        video_query_list = []
        docs = []
        vg_list = []
        for x in batch:
            src_ids, src_mask, tgt_ids, tgt_mask, video_atts, video_hidden, video_query, vg_path, relation, links, input_text, video_id, doc = x
            src_list.append(src_ids)
            src_mask_list.append(src_mask)
            tgt_list.append(tgt_ids)
            tgt_mask_list.append(tgt_mask)
            relation_list.append(relation)
            link_list.append(links)
            input_texts.append(input_text)
            video_id_list.append(video_id)
            docs.append(doc)
            video_atts_list.append(video_atts)
            video_hidden_list.append(video_hidden)
            video_query_list.append(video_query)
            vg_list.append(vg_path)
        inputs = {
            'input_ids': src_list,
            'attention_mask': src_mask_list,
            'labels': tgt_list,
            'labels_mask': tgt_mask_list,
            "vg": vg_list,
            'relation': relation_list,
            'link': link_list,
            'input_texts': input_texts,
            'video_id_list': video_id_list,
            'docs': docs,
            "video_atts_list": video_atts_list,
            "video_hidden_list": video_hidden_list,
            "video_query_list": video_query_list,
        }
        return inputs
    
class PretrainCollator(object):
    def __init__(self, tokenizer, max_seq_length):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.pad_token_id = tokenizer.pad_token_id

    def padding(self, data, ml, pad_vlaue):
        return torch.stack([torch.cat([item, torch.LongTensor([pad_vlaue]*(ml - item.shape[0]))], dim=0) for item in data], dim=0)
    
    def __call__(self, batch: List[Any]) -> Dict[str, Any]:
        src_list = []
        src_mask_list = []
        tgt_list = []
        tgt_mask_list = []
        relation_list = []
        link_list = []
        input_texts = []
        video_id_list = []
        docs = []
        video_atts_list = []
        video_hidden_list = []
        video_query_list = []
        vg_list = []
        for x in batch:
            src_ids, src_mask, tgt_ids, tgt_mask, video_atts, video_hidden, video_query, vgs, relation, links, input_text, video_id, doc = x
            src_list.append(src_ids)
            src_mask_list.append(src_mask)
            tgt_list.append(tgt_ids)
            tgt_mask_list.append(tgt_mask)
            relation_list.append(relation)
            link_list.append(links)
            input_texts.append(input_text)
            video_id_list.append(video_id)
            docs.append(doc)
            video_atts_list.append(video_atts)
            video_hidden_list.append(video_hidden)
            video_query_list.append(video_query)
            vg_list.append(vgs)

        inputs = {
            'input_ids': src_list,
            'attention_mask': src_mask_list,
            'labels': tgt_list,
            'labels_mask': tgt_mask_list,
            'relation': relation_list,
            'link': link_list,
            'input_texts': input_texts,
            'video_id_list': video_id_list,
            "vgs": vg_list,
            'docs': docs,
            "video_atts_list": video_atts_list,
            "video_hidden_list": video_hidden_list,
            "video_query_list": video_query_list,
        }
        return inputs


class MNRECollator(object):
    def __init__(self, tokenizer, max_seq_length):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.pad_token_id = tokenizer.pad_token_id

    def padding(self, data, ml, pad_vlaue):
        return torch.stack([torch.cat([item, torch.LongTensor([pad_vlaue]*(ml - item.shape[0]))], dim=0) for item in data], dim=0)
    
    def __call__(self, batch: List[Any]) -> Dict[str, Any]:
        src_list = []
        src_mask_list = []
        tgt_list = []
        tgt_mask_list = []
        relation_list = []
        link_list = []
        input_texts = []
        video_id_list = []
        docs = []
        tokens, tokens_att, list_p1s, list_p2s, list_p3s, list_p4s, weights, hs, ts, phrases, phrase_atts = [], [], [], [], [], [], [], [], [], [], []
        for x in batch:
            src_ids, src_mask, tgt_ids, tgt_mask, list_p1, list_p2, list_p3, list_p4, weight, token, token_att, h, t, phrase, phrase_att, relation, links, input_text, video_id, doc = x
            src_list.append(src_ids)
            src_mask_list.append(src_mask)
            tgt_list.append(tgt_ids)
            tgt_mask_list.append(tgt_mask)
            relation_list.append(relation)
            link_list.append(links)
            input_texts.append(input_text)
            video_id_list.append(video_id)
            docs.append(doc)
            list_p1s.append(list_p1)
            list_p2s.append(list_p2)
            list_p3s.append(list_p3)
            list_p4s.append(list_p4)
            weights.append(weight)
            tokens.append(token)
            tokens_att.append(token_att)
            hs.append(h)
            ts.append(t)
            phrases.append(phrase)
            phrase_atts.append(phrase_att)

        inputs = {
            'input_ids': src_list,
            'attention_mask': src_mask_list,
            'labels': tgt_list,
            'labels_mask': tgt_mask_list,
            'pic_dif': list_p1,
            'pic_ori': list_p2,
            'pic_dif_objects': list_p3,
            'pic_ori_objects': list_p4,
            'weight': weight,
            "hs": hs,
            "tokens": tokens,
            "tokens_att": tokens_att,
            "ts": ts,
            "phrases": phrases,
            'phrase_atts': phrase_atts,
            'relation': relation_list,
            'link': link_list,
            'input_texts': input_texts,
            'video_id_list': video_id_list,
            'docs': docs
        }
        return inputs