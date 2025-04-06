## SwarmCloneEmotionDetection

### 介绍

SwarmCloneEmotionDetection 使用 BiLSTM，仅对最后一个时刻的潜变量增加一个 MLP 做分类。

测试数据集 1 [情感对话生成数据集](https://www.biendata.xyz/ccf_tcci2018/datasets/ecg/)  
测试数据集 2 [SMP2020微博情绪分类评测](https://smp2020ewect.github.io/)

| 数据集 | train acc | test acc |
| --- | --- | --- |
| 情感对话生成数据集 | 0.9649 | 0.9117 |
| SMP2020微博情绪分类 | 0.8214 | 0.6771 |

- 对于 情感对话生成数据集，我们将问题与回答分开作为一条数据，随机分割 0.1 作为测试集。  
- 对于 SMP2020微博情绪分类数据集，我们使用 测试数据集/真实测试数据/usual_test_labeled.txt 作为测试集。  

### 获取最新的模型权重请前往

[HuggingFace](https://huggingface.co/YamadaMano/SWCBiLSTM) | [ModelScope](https://modelscope.cn/models/MomoiaMoia/SWCBiLSTM/files)

### 使用方法

关于 Tokenizer 请参考 [Kyv001/MiniLM2](https://github.com/SwarmClone/MiniLM2)

```py
tokenizer = AutoTokenizer.from_pretrained(
    "/home/momoia/codes/MiniLM2/models/tokenizers/tokenizer64k",
    trust_remote_code=True,
)
model = AutoModelForSequenceClassification.from_pretrained(
    "YamadaMano/SWCBiLSTM", 
    trust_remote_code=True
)

input_ids = tokenizer("你好", return_tensors="pt", padding=True)["input_ids"]

emotion = ["中性", "喜爱", "悲伤", "厌恶", "愤怒", "高兴"]
print(model.forward(input_ids))
```