from transformers import (
    set_seed,
    HfArgumentParser,
    TrainingArguments,
    AutoConfig,
)
from decord import VideoReader, cpu
import torch.distributed as dist
import torch.nn as nn
from torch.utils.data import DataLoader
# from deepspeed import DeepSpeedEngine
from video_llama.common.config import Config
from video_llama.common.registry import registry
import argparse
from loguru import logger
import os
from os.path import join
import yaml
import torch
from video_llama.processors.video_processor import ToTHWC,ToUint8,load_video
import bitsandbytes as bnb
from collections import defaultdict
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoTokenizer
from video_llama.models.modeling_llama import LlamaForCausalLM, MutliModel
from component.collator import PretrainCollator, MNRECollator, MNERCollator
from component.dataset import PretrainDataProcessor, IterableDataset
from component.argument import QLoRAArguments
from component.trainer import LoRATrainer
from component.loss import CausalLMLoss
from tqdm import tqdm
import json
from torch.nn.parallel import DistributedDataParallel as DDP
from PIL import Image

DEFAULT_VIDEO_TOKEN = "<video>"
DEFAULT_VIDEO_PATCH_TOKEN = "<vid_patch>"
DEFAULT_VID_START_TOKEN = "<vid_start>"
DEFAULT_VID_END_TOKEN = "<vid_end>"
def verify_model_dtype(model):
    """
    查看模型种各种类型的参数的情况
    """
    dtype2param_num = defaultdict(int)  # 每种数据类型的参数量
    dtype2param_name = defaultdict(list)  # 每种数据类型的参数名称
    dtype2trainable_param_num = defaultdict(int)  # 每种数据类型参与训练的参数量
    dtype2trainable_param_name = defaultdict(list)  # 每种数据类型参与训练的参数名称
    for name, p in model.named_parameters():
        dtype = p.dtype
        dtype2param_num[dtype] += p.numel()
        dtype2param_name[dtype].append(name)
        if p.requires_grad:
            dtype2trainable_param_num[dtype] += p.numel()
            dtype2trainable_param_name[dtype].append(name)
    # 统计全部参数中，各种类型参数分布
    total = 0
    print('verify all params of the model')
    for k, v in dtype2param_num.items():
        total += v
    for k, v in dtype2param_num.items():
        print(k, v, v / total)
    for k, v in dtype2trainable_param_name.items():
        print(k, v)

    print()
    # 统计可训练参数中，各种类型参数分布
    print('verify trainable params the model')
    total_trainable = 0
    for k, v in dtype2trainable_param_num.items():
        total_trainable += v
    for k, v in dtype2trainable_param_num.items():
        print(k, v, v / total_trainable)
    for k, v in dtype2trainable_param_num.items():
        print(k, v)

    # 查看参与训练的参数情况
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Total model params: %.2fM" % (total / 1e6))
    logger.info(
        f'trainable params: {trainable} || all params: {total} || trainable%: {round(trainable / total, 4)}')


def find_all_linear_names(model):
    """
    找出所有全连接层，为所有全连接添加adapter
    """
    cls = bnb.nn.Linear4bit
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names:  # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    
def setup_everything(args):


    train_args_file = args.train_args_file
    # 读取训练的参数配置
    parser = HfArgumentParser((QLoRAArguments, TrainingArguments))
    # 解析得到自定义参数，以及自带参数
    args, training_args = parser.parse_yaml_file(yaml_file=train_args_file)
    seed_torch(int(training_args.seed))

    # 创建输出目录
    if not os.path.exists(training_args.output_dir):
        os.makedirs(training_args.output_dir)
    logger.add(join(training_args.output_dir, 'train.log'))
    logger.info("train_args:{}".format(training_args))
    # 加载训练配置文件
    with open(train_args_file, "r") as f:
        train_args = yaml.safe_load(f)
    # 保存训练参数到输出目录
    with open(join(training_args.output_dir, 'train_args.yaml'), "w") as f:
        yaml.dump(train_args, f)
    # 设置随机种子
    set_seed(int(training_args.seed))
    training_args.train_embedding = args.train_embedding
    return args, training_args


def load_tokenizer(args, myConfig):
    # 扩充词表的时候，model_name_or_path与tokenizer_name_or_path不一致，其他情况是一致的
    if args.tokenizer_name_or_path is None:
        tokenizer_name_or_path = args.model_name_or_path
    else:
        tokenizer_name_or_path = args.tokenizer_name_or_path
    # model配置，用于加载tokenizer
    # try:
    config = AutoConfig.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    # except:
    #     config = LlavaConfig.from_pretrained(args.model_name_or_path, trust_remote_code=True)

    logger.info(f'Loading tokenizer from {tokenizer_name_or_path}')
    # 加载tokenzier
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name_or_path,
        trust_remote_code=True,
        # llama不支持fast
        use_fast=False
    )
    try:
        tokenizer.pad_token_id = 0
    except:
        pass
    # assert tokenizer.pad_token_id is not None, "pad_token_id should not be None"
    logger.info(f'vocab_size of tokenizer: {tokenizer.vocab_size}')
    return tokenizer


def load_model(args, training_args, tokenizer, cfg, my_config):
    config = AutoConfig.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    logger.info(f'vocab_size of original model: {config.vocab_size}')

    # 如果扩词表，但却不训练词表，不合法
    if config.vocab_size < tokenizer.vocab_size and args.train_embedding is False:
        raise Exception('When model.vocab_size < tokenizer.vocab_size, train_embedding should be True')


    # 加载模型
    logger.info(f'Loading model from base model: {args.model_name_or_path}')
    # local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    # device_map = {'': local_rank}

    # model = MutliModel.from_pretrained(
    #         config._name_or_path,
    #         device_map=device_map,
    #     )
    # model = model.model
    model_config = cfg.model_cfg
    model_config.device_8bit = 0
    model_config.model_type = my_config.model_type
    model_cls = registry.get_model_class(model_config.arch)
    model = model_cls.from_config(model_config).to('cuda:{}'.format(0))
    # model = model.model
    # casts all the non int8 modules to full precision (fp32) for stability
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)
    print(f'memory footprint of model: {model.get_memory_footprint() / (1024 * 1024 * 1024)} GB')
    return model


def insert_adapter(args, model, Myconfig):
    # 找到所有需要插入adapter的全连接层，排除embed_tokens与lm_head
    target_modules = find_all_linear_names(model.model)
    if not target_modules:
        target_modules = ["lm_head"]
    # exit()
    config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        modules_to_save=None)
    llama_model = get_peft_model(model.model, config)
    model = get_peft_model(model, config)
    model.base_model.model.model = llama_model.base_model.model

    model.print_trainable_parameters()
    model.model.model.config.torch_dtype = torch.float32

    # 词表参与训练
    if args.train_embedding:
        for n, p in model.named_parameters():
            if "embed_tokens" in n or "lm_head" in n:
                try:
                    p.requires_grad = True
                except:
                    pass

    # 查看模型种各种类型的参数的情况
    verify_model_dtype(model)
    return model


def init_components(args, training_args, config):
    """
    初始化各个组件
    """
    print(config)
    cfg = Config(config)

    logger.info('Initializing components...')
    # 务必设为False，否则多卡训练会报错
    training_args.ddp_find_unused_parameters = False

    # 加载tokenizer
    tokenizer = load_tokenizer(args, config)
    # 加载模型
    model = load_model(args, training_args, tokenizer, cfg, config)

    model = insert_adapter(args, model, config)
    # print(model)
    # 初始化损失函数
    loss_func = CausalLMLoss(ignore_index=-100)
    # 加载训练集和验证集
    data_processor = PretrainDataProcessor(
        args.data_path,
        tokenizer,
        args.max_seq_length,
        args.min_seq_length,
        args.window_step_size,
        args.eval_size
    )

    vis_processor_cfg = cfg.datasets_cfg.webvid.vis_processor.train
    vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
    all_dataset = data_processor.load_multiData(config, model)

    # all_dataset = data_processor.load_dataset()
    # if config.dataset == "m3d":
    data_collator = PretrainCollator(tokenizer, args.max_seq_length)
    # elif config.dataset == "mnre":
    #     data_collator = MNRECollator(tokenizer, args.max_seq_length)
    # elif config.dataset == "mner15" or config.dataset == "mner17":
    #     data_collator = MNERCollator(tokenizer, args.max_seq_length)

    print(config.epochs)
    training_args.num_train_epochs = config.epochs
    training_args.per_device_train_batch_size = 1
    training_args.per_device_eval_batch_size = 1

    # training_args.auto_find_batch_size = True
    # print(training_args)
    # print("=====================")
    # print(args)
    # exit()
    trainer = LoRATrainer(
        model=model,
        args=[training_args, config],
        train_dataset=all_dataset[0],
        eval_dataset=all_dataset[1],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_loss=loss_func
    )
    return trainer, model, all_dataset[-1], tokenizer

def get_seq_frames(total_num_frames, desired_num_frames):
    """
    Calculate the indices of frames to extract from a video.

    Parameters:
    total_num_frames (int): Total number of frames in the video.
    desired_num_frames (int): Desired number of frames to extract.

    Returns:
    list: List of indices of frames to extract.
    """

    # Calculate the size of each segment from which a frame will be extracted
    seg_size = float(total_num_frames - 1) / desired_num_frames

    seq = []
    for i in range(desired_num_frames):
        # Calculate the start and end indices of each segment
        start = int(np.round(seg_size * i))
        end = int(np.round(seg_size * (i + 1)))

        # Append the middle index of the segment to the list
        seq.append((start + end) // 2)

    return seq

def load_video_gpt(vis_path, n_clips=1, num_frm=100):
    """
    Load video frames from a video file.

    Parameters:
    vis_path (str): Path to the video file.
    n_clips (int): Number of clips to extract from the video. Defaults to 1.
    num_frm (int): Number of frames to extract from each clip. Defaults to 100.

    Returns:
    list: List of PIL.Image.Image objects representing video frames.
    """

    # Load video with VideoReader
    vr = VideoReader(vis_path, ctx=cpu(0))
    total_frame_num = len(vr)

    # Currently, this function supports only 1 clip
    assert n_clips == 1

    # Calculate total number of frames to extract
    total_num_frm = min(total_frame_num, num_frm)
    # Get indices of frames to extract
    frame_idx = get_seq_frames(total_frame_num, total_num_frm)
    # Extract frames as numpy array
    img_array = vr.get_batch(frame_idx).numpy()
    # Set target image height and width
    target_h, target_w = 224, 224
    # If image shape is not as target, resize it
    if img_array.shape[-3] != target_h or img_array.shape[-2] != target_w:
        img_array = torch.from_numpy(img_array).permute(0, 3, 1, 2).float()
        img_array = torch.nn.functional.interpolate(img_array, size=(target_h, target_w))
        img_array = img_array.permute(0, 2, 3, 1).to(torch.uint8).numpy()

    # Reshape array to match number of clips and frames
    img_array = img_array.reshape(
        (n_clips, total_num_frm, img_array.shape[-3], img_array.shape[-2], img_array.shape[-1]))
    # Convert numpy arrays to PIL Image objects
    clip_imgs = [Image.fromarray(img_array[0, j]) for j in range(total_num_frm)]

    return clip_imgs

def get_spatio_temporal_features_torch(features):
    """
    Computes spatio-temporal features from given features.

    Parameters:
    features (torch.Tensor): Input features to process.

    Returns:
    torch.Tensor: Spatio-temporal features.
    """

    # Extract the dimensions of the features
    t, s, c = features.shape

    # Compute temporal tokens as the mean along the time axis
    temporal_tokens = torch.mean(features, dim=1)

    # Padding size calculation
    padding_size = 100 - t

    # Pad temporal tokens if necessary
    if padding_size > 0:
        padding = torch.zeros(padding_size, c, device=features.device)
        temporal_tokens = torch.cat((temporal_tokens, padding), dim=0)

    # Compute spatial tokens as the mean along the spatial axis
    spatial_tokens = torch.mean(features, dim=0)

    # Concatenate temporal and spatial tokens and cast to half precision
    concat_tokens = torch.cat([temporal_tokens, spatial_tokens], dim=0).half()

    return concat_tokens

def pred(model, training_args, test_dataset, tokenizer, args):
    best_step = 0
    best_filename = ""
    best_model_path = f"{training_args.output_dir}"
    for filename in os.listdir(best_model_path):
        if "checkpoint" in filename:
            try:
                step = int(filename.split("-")[1])
            except:
                continue
            if step > best_step:
                best_step = step
                best_filename = filename
    print(f"{best_model_path}/{best_filename}")
    if os.path.exists(f"./{best_model_path}/{best_filename}/adapter_model.bin"):
        fintuneModel_path = f"./{best_model_path}/{best_filename}/adapter_model.bin"
    else:
        fintuneModel_path = f"./{best_model_path}/adapter_model.bin"
    fintuneModel = torch.load(fintuneModel_path)
    embedding = torch.load(f"./{best_model_path}/{best_filename}/embed_tokens.bin")
    lm_head = torch.load(f"./{best_model_path}/{best_filename}/lm_head.bin")
    model_dict = model.state_dict()
    model_dict.update({f"{'.'.join(k.split('.')[:-1] + ['default'] + ['weight'])}":v for k,v in fintuneModel.items()})
    model_dict.update({"base_model.model.model.model.embed_tokens.weight":embedding["weight"]})
    model_dict.update({"base_model.model.model.lm_head.weight":lm_head["weight"]})

    video_Qformer_dict = torch.load(f"./{best_model_path}/{best_filename}/video_Qformer.bin")
    llama_proj_dict = torch.load(f"./{best_model_path}/{best_filename}/llama_proj.bin")
    model_dict.update({f"base_model.model.video_Qformer.{k}":v for k,v in video_Qformer_dict.items()})
    model_dict.update({f"base_model.model.llama_proj.{k}":v for k,v in llama_proj_dict.items()})

    model.load_state_dict(model_dict)
    model = model.eval()
    llama_model = model.base_model.model.model
    load_in_4bit = False
    # 生成超参配置
    max_new_tokens = 4096
    top_p_list = [0.9]
    temperature_list = [0.35]
    repetition_penalty_list = [1.0]
    device = 'cuda'
    for top_p in top_p_list:
        for temperature in temperature_list:
            for repetition_penalty in repetition_penalty_list:
                for j in range(0, 1):
                    result = []
                    outputRoot = training_args.output_dir.split("/")[-2].split("-")[0]
                    if not os.path.exists(f'./result_{outputRoot}'):
                        os.makedirs(f'./result_{outputRoot}')
                    with open(f"./result_{outputRoot}/{config.dataset}-{config.template}-{config.model_type}-{training_args.seed}-{training_args.learning_rate}.json", "a+", encoding="utf-8") as f:
                        for i, data_batch in enumerate(tqdm(test_dataset)):
                            vg, relation, link, src, video_id, doc = data_batch[-6:]
                            data_batch = [data.to(device) for data in data_batch[:-6]]
                            embed_tokens = llama_model.model.embed_tokens
                            inputs_embeds = None

                            input_ids, src_mask_list, tgt_list, tgt_mask_list, atts, hiddens, qureys = data_batch
                            video_Qformer = model.base_model.model.video_Qformer
                            llama_proj = model.base_model.model.llama_proj
                            if args.dataset == "mnre":
                                vg_feature = []
                                for vg_path in vg:
                                    vg_path = ".".join(vg_path.split(".")[:-1])
                                    video_path = f"../../../mnt/third/liujiang/multiTask/data/img_vg/test/feature"
                                    vg_atts = torch.load(f"{video_path}/{vg_path}_frame_atts.pth").cuda().long()
                                    vg_hidden = torch.load(f"{video_path}/{vg_path}_frame_hidden_state.pth").cuda().float()
                                    vg_query = torch.load(f"{video_path}/{vg_path}_video_query_tokens.pth").cuda().float()
                                    vg_output = video_Qformer.bert(
                                        query_embeds=vg_query.float(),
                                        encoder_hidden_states=vg_hidden.float(),
                                        encoder_attention_mask=vg_atts.float(),
                                        return_dict=True,
                                        )
                                    vg_output = vg_output.last_hidden_state
                                    vg_output = llama_proj(vg_output)
                                    vg_feature.append(vg_output)
                                vg_feature = torch.cat(vg_feature, dim=1)
                            video_query_output = video_Qformer.bert(
                                    query_embeds=qureys.float(),
                                    encoder_hidden_states=hiddens.float(),
                                    encoder_attention_mask=atts.float(),
                                    return_dict=True,
                                    )
                            inputs_llama_video = video_query_output.last_hidden_state

                            inputs_llama_video = llama_proj(inputs_llama_video)
                            if args.dataset == "mnre":
                                inputs_llama_video = torch.cat([inputs_llama_video, vg_feature], dim=1)
                            src_embeds = embed_tokens(input_ids)
                            result_embeds = embed_tokens(tgt_list)   
                            inputs_embeds = torch.cat([src_embeds.unsqueeze(0), inputs_llama_video, result_embeds.unsqueeze(0)], dim=1)

                            with torch.no_grad():
                                if args.model_type == "llama":
                                    outputs = llama_model.generate(
                                        inputs_embeds=inputs_embeds,
                                        max_length=4096, 
                                        do_sample=True,
                                        top_p=top_p, 
                                        temperature=temperature, 
                                        repetition_penalty=repetition_penalty,
                                        eos_token_id=tokenizer.eos_token_id
                                    )
                                else:
                                    outputs = llama_model.generate(
                                        inputs_embeds=inputs_embeds,
                                        max_new_tokens=4096, 
                                        do_sample=True,
                                        top_p=top_p, 
                                        temperature=temperature, 
                                        repetition_penalty=repetition_penalty,
                                        eos_token_id=tokenizer.eos_token_id
                                    )
                                outputs = outputs.tolist()[0]
                                response = tokenizer.decode(outputs)
                                response = response.strip().replace(tokenizer.eos_token, "").strip()
                            # print(response)
                            f.write(json.dumps({
                                    "link": link,
                                    "relation": relation,
                                    "video_id": video_id,
                                    "doc": doc,
                                    "result": response
                                }, ensure_ascii=False))
                            f.write("\n")

def main(config):
    # 进行一些配置和检查
    args, training_args = setup_everything(config)
    # 加载各种组件
    trainer, model, test_dataset, tokenizer = init_components(args, training_args, config)
    # 开始训练
    if config.is_train:
        logger.info("*** starting training ***")
        train_result = trainer.train()
        # 保存最后的checkpoint
        trainer.save_model(training_args.output_dir)  # Save the tokenizer too
        # 保存训练指标
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
    # todo merge lora权重
    pred(model, training_args, test_dataset, tokenizer, config)

import random
import numpy as np
def seed_torch(seed=3306):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_args_file", type=str, default='./train_args/code-llm.yaml', help="")
    parser.add_argument("--cfg-path", default='eval_configs_llama/video_llama_eval_withaudio.yaml', help="path to configuration file.")
    parser.add_argument("--model_type", type=str, default="llama", choices=["llama", "baichuan"])
    parser.add_argument("--is_train", type=str2bool, default="True")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--dataset", type=str, default="m3d_en",choices=["m3d_en", "m3d_zh", "mnre", "mnre_v1", "mner15", "mner17"])
    parser.add_argument("--template", type=str, default="ours_code", choices=["ours_code", "ours_nl", "reamo", "umie", "feature"])
    parser.add_argument('--mask_entity', action='store_true',
                    help='Mask entity mentions')
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    config = parser.parse_args()
    logger.info(config)
    # os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_ids
    main(config)
