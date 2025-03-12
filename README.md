## SwarmCloneEmotionDetection

SwarmCloneEmotionDetection 使用 BiLSTM，仅对最后一个时刻的潜变量增加一个 MLP 做分类。力求在模型体量和精度间找到最佳平衡。

测试数据集 1 [情感对话生成数据集](https://www.biendata.xyz/ccf_tcci2018/datasets/ecg/)  
测试数据集 2 [SMP2020微博情绪分类评测](https://smp2020ewect.github.io/)

| 数据集 | train acc | test acc |
| --- | --- | --- |
| 情感对话生成数据集 | 0.9465 | 0.9004 |
| SMP2020微博情绪分类 | \ | \ |
