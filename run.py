import os

 
lrDic = {
    "m3d_en": 0.0002,
    "m3d_zh": 0.0008,
    "mnre": 0.0002,
    "mner15": 0.0002,
    "mner17": 0.0002}

epochDic = {
    "m3d_en": 5,
    "m3d_zh": 5,
    "mnre": 3,
    "mner15": 5,
    "mner17": 5}
datasetList = ["m3d_en"]
templateList = ["ours_code"]
is_train = True
seedList = [42]

for dataset in datasetList:
    epochs = epochDic[dataset]
    if dataset == "m3d_zh":
        model_path = "../../../mnt/third/liujiang/multiTask/pretrain/baichuan"
        model_type = "baichuan"
    else:
        # model_path = "../../../mnt/third/liujiang/multiTask/pretrain/llama-2-7b-chat-hf"
        model_path = "../../../mnt/third/liujiang/multiTask/pretrain/code-llama"
        model_type = "llama"
    for template in templateList:
        outputRoot = f"output_code_{dataset}_{template}_{lrDic[dataset]}_code"
        for seed in seedList:
            for i in range(0, 1):
                with open("./train_args/code-llm-defualt.yaml", "r", encoding="utf-8") as f:
                    args = f.read()
                
                new_args = []
                for line in args.split("\n"):
                    if "output_dir" in line:
                        new_args.append(f'output_dir: ../../../mnt/third/liujiang/{outputRoot}/{seed}')
                    elif "model_name_or_path" in line:
                        new_args.append(f'model_name_or_path: {model_path}')
                    elif "tokenizer_name_or_path" in line:
                        new_args.append(f'tokenizer_name_or_path: {model_path}')
                    elif "learning_rate" in line:
                        new_args.append(f'learning_rate: {lrDic[dataset]}')
                    elif "seed" in line:
                        new_args.append(f'seed: {seed}')
                    else:
                        new_args.append(line)
                with open("./train_args/code-llm.yaml", "w", encoding="utf-8") as f:
                    f.write("\n".join(new_args))

                with open("./eval_configs_llama/video_llama_eval_withaudio_defualt.yaml", "r", encoding="utf-8") as f:
                    args = f.read()
                
                new_args = []
                for line in args.split("\n"):
                    if "llama_model" in line:
                        new_args.append(f'  llama_model: {model_path}')
                    else:
                        new_args.append(line)
                with open("./eval_configs_llama/video_llama_eval_withaudio.yaml", "w", encoding="utf-8") as f:
                    f.write("\n".join(new_args))
                print(f"{dataset}-{template}-{model_type}-{seed}")
                os.system(f"python train.py --epochs {epochs} --is_train {is_train} --model_type {model_type} --dataset {dataset} --template {template}")