# AiMed: Artificial Intelligence large language model for chinese Medicine  面向中文医学的人工智能大语言模型
[![License Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![python_version](https://img.shields.io/badge/Python-3.8%2B-green.svg)](requirements.txt)
## 🔬 介绍

**AiMed** 面向中文医学的人工智能大语言模型由**清华大学OpenDE团队**和**中国医学科学院医学信息研究所**联合研发。 

**AiMed** 期望实现有效处理医学知识问答、医学论文阅读、医学文献检索等任务和在医学科研中的应用。

## ⏩ 构建流程

**AiMed** 整个构建流程包括：
- PT增量预训练
- SFT有监督微调
- RLHF(奖励建模、强化学习训练)
- DPO(直接偏好优化)

## 🌏 模型基座

| 模型名                                                   | 模型大小                     | 开源参数                                              |
| ------------------------------------------------------- | --------------------------- |----------------------------------------------------------------------------|          
| [Baichuan](https://github.com/baichuan-inc/baichuan-13B) | 13B                      | [baichuan-inc/Baichuan-13B-Chat](https://huggingface.co/baichuan-inc/Baichuan-13B-Chat)|

## 😜 推理和部署

推理所需的模型权重、配置目前公开于

| 模型名                                                   | 模型用途                     | 开源参数                                              |
| ------------------------------------------------------- | --------------------------- |----------------------------------------------------------------------------|          
| AiMed | 基础医学百科大模型                      |[ https://huggingface.co/DuJinHua/AiMed](https://huggingface.co/DuJinHua/AiMed)|
| AiMed_Paper | 医学文献摘要生成大模型                      | [https://huggingface.co/DuJinHua/AiMed](https://huggingface.co/DuJinHua/AiMed_PaperAbs)|

下面示范多种推理方式。

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

授权协议为 [The Apache License 2.0](/LICENSE)，目前仅支持学术研究，不支持商业用途。
