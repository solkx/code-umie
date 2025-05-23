import torch
import torch.nn as nn
from transformers import BloomModel
import json

class Loss(object):
    """
    所有loss的类父类
    """
    def __call__(self, model, inputs, training_args, return_outputs=False):
        """
        todo label smoothing
        用于计算loss。
        看源码发现，return_outputs=True为train时调用，return_outputs=False为eval和predict调用
        :param model: 模型
        :param inputs: 模型输入，dict
        :param training_args: 训练配置参数
        :param return_outputs:是否返回模型的输出
        :return:
        """
        raise NotImplemented


class CausalLMLoss(Loss):
    """
    预训练损失
    """
    def __init__(self, ignore_index=-100):
        super().__init__()
        self.ignore_index = ignore_index
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def __call__(self, model, inputs, args, return_outputs=False):
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        labels = inputs['labels']
        labels_mask = inputs["labels_mask"]
        video_ids = inputs["video_id_list"]
        video_atts_list = inputs["video_atts_list"]
        video_hidden_list = inputs["video_hidden_list"]
        video_query_list = inputs["video_query_list"]
        vg = inputs["vgs"]
        # 模型前馈预测

        outputs, labels = model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True, args=args, labels=labels, labels_mask=labels_mask, video_atts_list=video_atts_list, video_hidden_list=video_hidden_list, video_query_list=video_query_list, vg=vg)
        logits = outputs["logits"] if isinstance(outputs, dict) else outputs[0]
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss = self.loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        return (loss, outputs) if return_outputs else loss
