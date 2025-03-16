## SwarmCloneEmotionDetection

SwarmCloneEmotionDetection 使用 BiLSTM，仅对最后一个时刻的潜变量增加一个 MLP 做分类。力求在模型体量和精度间找到最佳平衡。

测试数据集 1 [情感对话生成数据集](https://www.biendata.xyz/ccf_tcci2018/datasets/ecg/)  
测试数据集 2 [SMP2020微博情绪分类评测](https://smp2020ewect.github.io/)

| 数据集 | train acc | test acc |
| --- | --- | --- |
| 情感对话生成数据集 | 0.9465 | 0.9004 |
| SMP2020微博情绪分类 | 0.8214 | 0.6771 |

- 对于 情感对话生成数据集，我们将问题与回答分开作为一条数据，随机分割 0.1 作为测试集。  
- 对于 SMP2020微博情绪分类数据集，我们使用 测试数据集/真实测试数据/usual_test_labeled.txt 作为测试集。  