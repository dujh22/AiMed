# AiMed: Artificial Intelligence large language model for chinese Medicine  面向中文医学的人工智能大语言模型
[![License Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![python_version](https://img.shields.io/badge/Python-3.8%2B-green.svg)](requirements.txt)
## 🔬 介绍

**AiMed** 面向中文医学的人工智能大语言模型由**清华大学OpenDE团队**和**中国医学科学院医学信息研究所**（下称“医信所”）联合发布。 

**AiMed** 期望实现有效处理医学知识问答、医学论文阅读、医学文献检索等任务和在医学科研中的应用。

## 📆 工作流

[2023/09/01] 清华大学完成第一阶段PT代码开发和模型训练，并接入相关测试，开始进行第一阶段实际运行

[2023/09/01] 医信所发布论文全文数据集1-预计20万篇

[2023/08/30] 医信所发布医学名词解释问答数据集和医学论文摘要问答数据集

[2023/08/29] 清华大学发布《医疗文献LLM训练方案》

[2023/08/29] 医信所发布医学数据数据集-共计108本，612个文件

[2023/08/28] 清华大学发布《关于基座模型选择的相关问题说明》和《关于大模型训练的成本核算》

[2023/08/23] 医信所发布论文摘要数据集2-共计312万篇

[2023/08/22] 清华大学与医信所联合发布《中文医学大语言模型合作方案》

[2023/08/21] 医信所发布论文摘要数据集1-共计276万篇

[2023/08/16] 针对中文医学领域构造大语言模型的初步设想与讨论


## ⏩ 构建流程

**AiMed** 整个构建流程包括：
- PT增量预训练
- SFT有监督微调
- RLHF(奖励建模、强化学习训练)
- DPO(直接偏好优化)

具体，AiMed参照ChatGPT等大语言模型的一般构造思路，实现了医学领域大模型的训练：


- 第一阶段：PT(Continue PreTraining)增量预训练
  - 在海量领域文档数据上二次预训练通用基座大语言模型，以注入领域知识
- 第二阶段：SFT(Supervised Fine-tuning)有监督微调
  - 构造指令微调数据集
  - 在预训练模型基础上做指令精调，以对齐指令意图
- 第三阶段 
  - RLHF(Reinforcement Learning from Human Feedback)基于人类反馈对语言模型进行强化学习，分两步：
    - RM(Reward Model)奖励模型建模
      - 构造人类偏好排序数据集
      - 训练奖励模型，用来建模人类偏好
        - 主要是"HHH"原则，具体是"helpful, honest, harmless"
    - RL(Reinforcement Learning)强化学习
      - 用奖励模型来训练SFT模型
      - 生成模型使用奖励或惩罚来更新其策略
      - 以便生成更高质量、更符合人类偏好的文本
- 第四阶段
  - DPO(Direct Preference Optimization) 直接偏好优化方法
    - DPO通过直接优化语言模型来实现对其行为的精确控制
    - 无需使用复杂的强化学习，也可以有效学习到人类偏好
    - DPO相较于RLHF更容易实现且易于训练，效果更好

## 🌏 模型基座

| 模型名                                                   | 模型大小                     | 开源参数                                                                                    |
| ------------------------------------------------------- | --------------------------- |-----------------------------------------------------------------------------------------|
| [LLaMA-2](https://huggingface.co/meta-llama)            | 7B/13B/70B                  | [ziqingyang/chinese-alpaca-2-7b](https://huggingface.co/ziqingyang/chinese-alpaca-2-7b)                                                                                        |
| [Baichuan](https://github.com/baichuan-inc/baichuan-13B) | 7B/13B                      | [baichuan-inc/Baichuan-13B-Chat](https://huggingface.co/baichuan-inc/Baichuan-13B-Chat) |
| [ChatGLM2](https://github.com/THUDM/ChatGLM2-6B)        | 6B                          | [THUDM/chatglm2-6b](https://huggingface.co/THUDM/chatglm2-6b)                           |

## 😜 推理和部署

推理所需的模型权重、源码、配置目前暂未公开，只保存于本地。下面以 Baichuan-13B-Chat 为例示范多种推理方式。

推理前请安装依赖：
```shell
pip install -r requirements.txt
```

### 1.Python代码方式
```shell
python test.py
```
> 在代码中，模型加载指定 device_map='auto'，会使用所有可用显卡。如需指定使用的设备，可以使用类似 export CUDA_VISIBLE_DEVICES=0,1（使用了0、1号显卡）的方式控制。

### 2.命令行工具方式
```shell
python cli_demo.py
```


## ▶️ Demo


### 1. 工业使用Demo

依靠streamlit运行以下命令，会在本地启动一个 web 服务，把控制台给出的地址放入浏览器即可访问。

```shell
streamlit run web_demo.py
```
### 2. 研究使用Demo

提供一个简洁的基于gradio的交互式web界面，启动服务后，可通过浏览器访问，输入问题，模型会返回答案。

启动服务，命令如下：
```shell
CUDA_VISIBLE_DEVICES=0 python gradio_demo.py --model_type base_model_type --base_model path_to_llama_hf_dir --lora_model path_to_lora_dir
```

比如：
```shell
CUDA_VISIBLE_DEVICES=0 python gradio_demo.py --model_type baichuan --base_model ./AiMed-13B-Chat-V1
```

参数说明：

- `--model_type {base_model_type}`：预训练模型类型，如llama、bloom、chatglm等
- `--base_model {base_model}`：存放HF格式的LLaMA模型权重和配置文件的目录，也可使用HF Model Hub模型调用名称
- `--lora_model {lora_model}`：LoRA文件所在目录，也可使用HF Model Hub模型调用名称。若lora权重已经合并到预训练模型，则删除--lora_model参数
- `--tokenizer_path {tokenizer_path}`：存放对应tokenizer的目录。若不提供此参数，则其默认值与--base_model相同
- `--template_name`：模板名称，如`vicuna`、`alpaca`等。若不提供此参数，则其默认值是vicuna
- `--only_cpu`: 仅使用CPU进行推理
- `--gpus {gpu_ids}`: 指定使用的GPU设备编号，默认为0。如使用多张GPU，以逗号分隔，如0,1,2
- `--resize_emb`：是否调整embedding大小，若不调整，则使用预训练模型的embedding大小，默认不调整


## ⚠️ 局限性

授权协议为 [The Apache License 2.0](/LICENSE)，可免费用做商业用途。
