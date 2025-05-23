# import transformers
# from transformers import (
#     PreTrainedModel,
#     TrainingArguments,
#     DataCollator,
#     PreTrainedTokenizerBase,
#     EvalPrediction,
#     TrainerCallback,
# )
# from typing import Callable, Dict, List, Optional, Tuple, Union, Any
# from torch import nn
# from torch.utils.data import Dataset
# from transformers.utils import (
#     logging,
# )
# from typing import Optional
# import os
# import torch
# from os.path import join
# from transformers.modeling_utils import unwrap_model
# from component.result_m3d import decode_link_rel, calu_res
# import json
# from tqdm import tqdm
# logger = logging.get_logger(__name__)

# TRAINING_ARGS_NAME = "training_args.bin"


# class Trainer(transformers.Trainer):
#     """
#     主要修改逻辑：通过传入compute_loss，支持自定义loss计算方式
#     """
#     def __init__(
#             self,
#             model: Union[PreTrainedModel, nn.Module] = None,
#             args: TrainingArguments = None,
#             data_collator: Optional[DataCollator] = None,
#             train_dataset: Optional[Dataset] = None,
#             eval_dataset: Optional[Dataset] = None,
#             tokenizer: Optional[PreTrainedTokenizerBase] = None,
#             model_init: Callable[[], PreTrainedModel] = None,
#             compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
#             callbacks: Optional[List[TrainerCallback]] = None,
#             optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
#             preprocess_logits_for_metrics: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = None,
#             compute_loss=None,
#     ):  
#         self.config = args[-1]
#         self.eval_dataset = eval_dataset
#         self.tokenizer = tokenizer
#         super(Trainer, self).__init__(
#             model=model,
#             args=args[0],
#             data_collator=data_collator,
#             train_dataset=train_dataset,
#             eval_dataset=eval_dataset,
#             tokenizer=tokenizer,
#             model_init=model_init,
#             compute_metrics=compute_metrics,
#             callbacks=callbacks,
#             optimizers=optimizers,
#             preprocess_logits_for_metrics=preprocess_logits_for_metrics,
#         )
#         self.loss_func = compute_loss
#         with open("./data/type2nature_all.json", "r", encoding="utf-8") as f:
#             self.t2n = json.loads(f.read())
#         # self.loss_func = self.loss_func 

#     def compute_loss(self, model, inputs, return_outputs=False):
#         """
#         重写loss的计算方式
#         How the loss is computed by Trainer. By default, all models return the loss in the first element.

#         Subclass and override for custom behavior.
#         """
#         if self.loss_func is None:
#             loss = super().compute_loss(model, inputs, return_outputs)
#         else:
#             loss = self.loss_func(model, inputs, self.config, return_outputs)
#         return loss


# class LoRATrainer(Trainer):
#     """
#     修改checkkpoint的保存逻辑，只保存lora
#     如果训练embedding，则保存embedding和lm_head的权重
#     """
#     def _save(self, output_dir: Optional[str] = None, state_dict=None):
#         # 如果需要扩词表，则保存word_tokens和lm_head的权重
#         if self.args.train_embedding:
#             if self.config.model_type == "llama" or self.config.model_type == "baichuan":
#                 # Only save the model itself if we are using distributed training
#                 output_dir = join(self.args.output_dir, f"checkpoint-{self.state.global_step}")
#                 best_output_dir = f'{self.args.output_dir}_best_model'
#                 if not os.path.exists(output_dir):
#                     os.makedirs(output_dir)
#                 if os.path.exists(f"{self.args.output_dir}/dev_result.json"):
#                     with open(f"{self.args.output_dir}/dev_result.json", "r", encoding="utf-8") as f:
#                         try:
#                             val_res = json.loads(f.read())
#                             best_dev_f1 = val_res["F1"]
#                         except:
#                             best_dev_f1 = 0
#                 else:
#                     best_dev_f1 = 0
#                 model = unwrap_model(self.model)
#                 total_p, total_r, total_c = 0, 0, 0
#                 total_rel_p, total_rel_r, total_rel_c = 0, 0, 0
#                 total_ceaf_c, total_ceaf_p, total_ceaf_r, total_muc_c, total_muc_p, total_muc_r, total_b3_c_p, total_b3_c_r, total_b3_p, total_b3_r = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
#                 total_ground_p, total_ground_r, total_ground_c = 0, 0, 0
#                 if self.state.global_step >= self.config.th_step:
#                     for i, data_batch in enumerate(tqdm(self.eval_dataset)):
#                         relation, link, src, video_id, doc = data_batch[-5:]
#                         lang = "en"
#                         data_batch = [data.cuda() for data in data_batch[:-5]]
#                         embed_tokens = model.model.model.model.embed_tokens
#                         inputs_embeds = None
#                         input_ids, src_mask_list, tgt_list, tgt_mask_list, atts, hiddens, qureys = data_batch
#                         video_Qformer = model.model.video_Qformer
#                         llama_proj = model.model.llama_proj
#                         video_query_output = video_Qformer.bert(
#                                 query_embeds=qureys.float(),
#                                 encoder_hidden_states=hiddens.float(),
#                                 encoder_attention_mask=atts.float(),
#                                 return_dict=True,
#                                 )
#                         inputs_llama_video = video_query_output.last_hidden_state

#                         inputs_llama_video = llama_proj(inputs_llama_video)
#                         src_embeds = embed_tokens(input_ids)
#                         result_embeds = embed_tokens(tgt_list)   
#                         inputs_embeds = torch.cat([src_embeds.unsqueeze(0), inputs_llama_video, result_embeds.unsqueeze(0)], dim=1)

#                         with torch.no_grad():
#                             if self.config.model_type == "llama":
#                                 lang = "en"
#                                 outputs = model.model.model.generate(
#                                     inputs_embeds=inputs_embeds,
#                                     max_length=self.config.max_len, 
#                                     do_sample=True,
#                                     top_p=0.9, 
#                                     temperature=0.35, 
#                                     repetition_penalty=1.0,
#                                     eos_token_id=self.tokenizer.eos_token_id
#                                 )
#                             else:
#                                 lang = "zh"
#                                 outputs = model.model.model.generate(
#                                     inputs_embeds=inputs_embeds,
#                                     max_new_tokens=self.config.max_len, 
#                                     do_sample=True,
#                                     top_p=0.9, 
#                                     temperature=0.35, 
#                                     repetition_penalty=1.0,
#                                     eos_token_id=self.tokenizer.eos_token_id
#                                 )
#                             outputs = outputs.tolist()[0]
#                             response = self.tokenizer.decode(outputs)
#                             response = response.strip().replace(self.tokenizer.eos_token, "").strip()
#                             # relation, link, src, video_id, doc
#                             if self.config.dataset == "en" or self.config.dataset == "zh":
#                                 p, r, c, ceaf_p, ceaf_r, ceaf_c, muc_p, muc_r, muc_c, b3_c_p, b3_c_r, b3_p, b3_r, rel_p, rel_r, rel_c, ground_p, ground_r, ground_c = decode_link_rel(response, link, relation, doc, video_id, True, True, True, lang)

#                                 total_p += p 
#                                 total_r += r
#                                 total_c += c

#                                 total_rel_p += rel_p
#                                 total_rel_r += rel_r
#                                 total_rel_c += rel_c

#                                 total_ceaf_c += ceaf_c
#                                 total_ceaf_p += ceaf_p
#                                 total_ceaf_r += ceaf_r
#                                 total_muc_c += muc_c
#                                 total_muc_p += muc_p
#                                 total_muc_r += muc_r
#                                 total_b3_c_p += b3_c_p
#                                 total_b3_c_r += b3_c_r
#                                 total_b3_p += b3_p
#                                 total_b3_r += b3_r

#                                 total_ground_p += ground_p
#                                 total_ground_r += ground_r
#                                 total_ground_c += ground_c
#                             if self.config.dataset == "mnre":
#                                 try:
#                                     pred = []
#                                     for v in self.t2n["mnre"].values():
#                                         if v in response:
#                                             pred.append(v)
#                                 except:
#                                     pred = ["no relation"]
#                                 if not pred:
#                                     pred.append("no relation")
#                                 pred = pred[0]
#                                 if "no relation" != pred:
#                                     total_p += 1
#                                 if relation != "no relation":
#                                     total_r += 1
#                                     if relation == pred:
#                                         total_c += 1

#                     ent_precious, ent_recall, ent_f1 = calu_res(total_p, total_r, total_c)
                            
#                     rel_precious, rel_recall, rel_f1 = calu_res(total_rel_p, total_rel_r, total_rel_c)

#                     ground_precious, ground_recall, ground_f1 = calu_res(total_ground_p, total_ground_r, total_ground_c)                  

#                     ceaf_precious, ceaf_recall, ceaf_f1 = calu_res(total_ceaf_p, total_ceaf_r, total_ceaf_c)
#                     muc_precious, muc_recall, muc_f1 = calu_res(total_muc_p, total_muc_r, total_muc_c)
#                     b3_precious, b3_recall, b3_f1 = calu_res(total_b3_p, total_b3_r, total_b3_c_p, total_b3_c_r)

#                     if self.config.dataset == "en" or self.config.dataset == "zh":
#                         avg_p = np.mean([ent_precious, np.mean([muc_precious, ceaf_precious, b3_precious]), rel_precious, ground_precious])
#                         avg_r = np.mean([ent_recall, np.mean([muc_recall, ceaf_recall, b3_recall]), rel_recall, ground_recall])
#                         avg_f1 = np.mean([ent_f1, np.mean([muc_f1, ceaf_f1, b3_f1]), rel_f1, ground_f1])
#                     elif self.config.dataset == "mnre":
#                         avg_p = ent_precious
#                         avg_r = ent_recall
#                         avg_f1 = ent_f1 
#                     if best_dev_f1 < avg_f1:
#                         with open(f"{self.args.output_dir}/dev_result.json", "w", encoding="utf-8") as f:
#                             f.write(json.dumps({
#                                 "P": avg_p,
#                                 "R": avg_r,
#                                 "F1": avg_f1
#                             }, ensure_ascii=False))
#                         os.system(f'rm -rf {best_output_dir}')
#                         if not os.path.exists(best_output_dir):
#                             os.makedirs(best_output_dir)
#                         super(LoRATrainer, self)._save(output_dir, state_dict)
#                         logger.info(f'Saving embed_tokens and lm_head to {output_dir}')
#                         torch.save(model.model.model.model.embed_tokens.state_dict(), join(output_dir, 'embed_tokens.bin'))
#                         torch.save(model.model.model.lm_head.state_dict(), join(output_dir, 'lm_head.bin'))
#                         try:
#                             torch.save(model.model.video_Qformer.state_dict(), join(output_dir, 'video_Qformer.bin'))
#                             torch.save(model.model.llama_proj.state_dict(), join(output_dir, 'llama_proj.bin'))
#                         except:
#                             pass
#                         os.system(f'cp -r {self.args.output_dir} {best_output_dir}')
#                     logger.info(f'best dev F1 / current dev F1: {best_dev_f1} / {avg_f1}')

import transformers
from transformers import (
    PreTrainedModel,
    TrainingArguments,
    DataCollator,
    PreTrainedTokenizerBase,
    EvalPrediction,
    TrainerCallback,
)
from typing import Callable, Dict, List, Optional, Tuple, Union, Any
from torch import nn
from torch.utils.data import Dataset
from transformers.utils import (
    logging,
)
from typing import Optional
import os
import torch
from os.path import join
from transformers.modeling_utils import unwrap_model


logger = logging.get_logger(__name__)

TRAINING_ARGS_NAME = "training_args.bin"


class Trainer(transformers.Trainer):
    """
    主要修改逻辑：通过传入compute_loss，支持自定义loss计算方式
    """
    def __init__(
            self,
            model: Union[PreTrainedModel, nn.Module] = None,
            args: TrainingArguments = None,
            data_collator: Optional[DataCollator] = None,
            train_dataset: Optional[Dataset] = None,
            eval_dataset: Optional[Dataset] = None,
            tokenizer: Optional[PreTrainedTokenizerBase] = None,
            model_init: Callable[[], PreTrainedModel] = None,
            compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
            callbacks: Optional[List[TrainerCallback]] = None,
            optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
            preprocess_logits_for_metrics: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = None,
            compute_loss=None,
    ):  
        self.config = args[-1]
        super(Trainer, self).__init__(
            model=model,
            args=args[0],
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )
        self.loss_func = compute_loss
        # self.loss_func = self.loss_func 

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        重写loss的计算方式
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        if self.loss_func is None:
            loss = super().compute_loss(model, inputs, return_outputs)
        else:
            loss = self.loss_func(model, inputs, self.config, return_outputs)
        return loss


class LoRATrainer(Trainer):
    """
    修改checkkpoint的保存逻辑，只保存lora
    如果训练embedding，则保存embedding和lm_head的权重
    """
    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        super(LoRATrainer, self)._save(output_dir, state_dict)
        model = unwrap_model(self.model)

        output_dir = join(self.args.output_dir, f"checkpoint-{self.state.global_step}")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        logger.info(f'Saving embed_tokens and lm_head to {output_dir}')
        torch.save(model.model.model.model.embed_tokens.state_dict(), join(output_dir, 'embed_tokens.bin'))
        torch.save(model.model.model.lm_head.state_dict(), join(output_dir, 'lm_head.bin'))
        torch.save(model.model.video_Qformer.state_dict(), join(output_dir, 'video_Qformer.bin'))
        torch.save(model.model.llama_proj.state_dict(), join(output_dir, 'llama_proj.bin'))
        
