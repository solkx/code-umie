from transformers import (
    set_seed,
    HfArgumentParser,
    TrainingArguments,
)
import torch
from video_llama.common.config import Config
from video_llama.common.registry import registry
import argparse
from loguru import logger
import os
from os.path import join
import yaml
from video_llama.processors.video_processor import load_video, load_video_base_img
from component.argument import QLoRAArguments
from video_llama.models.ImageBind.data import load_and_transform_audio_data
from tqdm import tqdm
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    
def setup_everything():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_args_file", type=str, default='./train_args/llama2-13b-ext.yaml', help="")
    args = parser.parse_args()

    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg-path", default='eval_configs_llama/video_llama_eval_withaudio copy.yaml', help="path to configuration file.")
    parser.add_argument("--lang", type=str, default="en", choices=["en", "zh", "mix"])
    parser.add_argument("--use_video", type=str2bool, default="False")
    parser.add_argument("--use_audio", type=str2bool, default="False")
    parser.add_argument("--is_train", type=str2bool, default="True")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    config = parser.parse_args()
    train_args_file = args.train_args_file
    # 读取训练的参数配置
    parser = HfArgumentParser((QLoRAArguments, TrainingArguments))
    # 解析得到自定义参数，以及自带参数
    args, training_args = parser.parse_yaml_file(yaml_file=train_args_file)
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
    set_seed(3306)
    training_args.train_embedding = args.train_embedding
    return args, training_args, config



def load_model(args, cfg):

    # 加载模型
    logger.info(f'Loading model from base model: {args.model_name_or_path}')
    model_config = cfg.model_cfg
    model_cls = registry.get_model_class(model_config.arch)
    model = model_cls.from_config(model_config)
    return model

def gen_feature_base_img(model, vis_processor):
    model = model.cuda()
    for fileName in tqdm(os.listdir("../../../mnt/nlp/liujiang/multiTask/video")):
        # print(f"../../../mnt/nlp/liujiang/multiTask/twitter17_data/twitter2017_images/{fileName}")
        # video = load_video_base_img(
        #     video_path=f"../../../mnt/third/liujiang/multiTask/data/img_vg/val/crops/{fileName}",
        #     n_frms=30,
        #     height=224,
        #     width=224,
        #     sampling="uniform", return_msg=True
        #     )
        print(fileName)
        video = load_video(
            video_path=f"../../../mnt/nlp/liujiang/multiTask/video/{fileName}",
            n_frms=30,
            height=224,
            width=224,
            sampling="uniform", return_msg=True
            )[0]
        print(video.shape)
        if video is not None:
            video = vis_processor.transform(video).to(torch.float16)
            video = video.unsqueeze(0).cuda()
            video_query_tokens, frame_hidden_state, frame_atts = model.encode_videoQformer_visual(video)
            print(video_query_tokens.shape, frame_hidden_state.shape, frame_atts.shape)
            exit()
            torch.save(video_query_tokens, f"../../../mnt/third/liujiang/multiTask/data/img_vg/val/feature/{fileName.split('.')[0]}_video_query_tokens.pth")
            torch.save(frame_hidden_state, f"../../../mnt/third/liujiang/multiTask/data/img_vg/val/feature/{fileName.split('.')[0]}_frame_hidden_state.pth")
            torch.save(frame_atts, f"../../../mnt/third/liujiang/multiTask/data/img_vg/val/feature/{fileName.split('.')[0]}_frame_atts.pth")

import torch
from PIL import Image
from torchvision import transforms
import os

def load_images_as_tensor(image_paths, image_size=(224, 224)):
    """
    将一系列图像加载并转换为形状为 (C, T, H, W) 的张量。
    
    参数:
    - image_paths: 包含图像路径的列表。
    - image_size: 目标图像大小，默认为 (224, 224)。
    
    返回:
    - tensor: 形状为 (C, T, H, W) 的张量。
    """
    # 定义图像预处理步骤
    preprocess = transforms.Compose([
        transforms.Resize(image_size),  # 调整图像大小
        transforms.ToTensor(),          # 转换为张量，并归一化到 [0, 1]
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化
    ])
    
    # 初始化一个空列表来存储图像张量
    image_tensors = []
    
    for image_path in image_paths:
        # 加载图像
        image = Image.open(image_path).convert('RGB')
        
        # 预处理图像
        tensor = preprocess(image)
        
        # 添加到列表中
        image_tensors.append(tensor)
    
    # 将列表转换为形状为 (T, C, H, W) 的张量
    image_tensors = torch.stack(image_tensors)
    
    # 如果需要调整顺序为 (C, T, H, W)，可以进行维度交换
    image_tensors = image_tensors.permute(1, 0, 2, 3)
    
    return image_tensors

def gen_feature(model, vis_processor):
    model = model.cuda()
    for fileName in tqdm(os.listdir("../../../mnt/nlp/liujiang/multiTask/img_org/train")):
        if os.path.exists(f"../../../mnt/nlp/liujiang/multiTask/mnre_feature/train/{fileName.split('.')[0]}_frame_atts.pth"):
            continue
        video = load_images_as_tensor(f"../../../mnt/nlp/liujiang/multiTask/img_org/train/{fileName}")
        video = vis_processor.transform(video).to(torch.float16)
        video = video.unsqueeze(0).cuda()
        video_query_tokens, frame_hidden_state, frame_atts = model.encode_videoQformer_visual(video)
        torch.save(video_query_tokens, f"../../../mnt/nlp/liujiang/multiTask/mnre_feature/train/{fileName.split('.')[0]}_video_query_tokens.pth")
        torch.save(frame_hidden_state, f"../../../mnt/nlp/liujiang/multiTask/mnre_feature/train/{fileName.split('.')[0]}_frame_hidden_state.pth")
        torch.save(frame_atts, f"../../../mnt/nlp/liujiang/multiTask/mnre_feature/train/{fileName.split('.')[0]}_frame_atts.pth")

def init_components(args, training_args, config):
    """
    初始化各个组件
    """
    print(config)
    cfg = Config(config)

    logger.info('Initializing components...')
    # 务必设为False，否则多卡训练会报错
    training_args.ddp_find_unused_parameters = False

    model = load_model(args, cfg)

    vis_processor_cfg = cfg.datasets_cfg.webvid.vis_processor.train
    vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)

    # 是否生成特征
    # gen_feature(model, vis_processor)
    gen_feature_base_img(model, vis_processor)


    
if __name__ == "__main__":
    args, training_args, config = setup_everything()
    # 加载各种组件
    init_components(args, training_args, config)
