import torch

from omegaconf import OmegaConf
from transformers import AutoTokenizer
from models.bilstm import BiLSTM


class BiLSTMModelScope(BiLSTM):
    def __init__(
        self,
        vocab_size,
        embedding_dim,
        hidden_dim,
        num_layers,
        num_classes,
        use_extra_mlp=False,
        mlp_ratio=0.5,
        mlp_dropout=0.2,
        recurrent_dropout=0.2,
        embedding_dropout=0.2,
    ):
        super().__init__(
            vocab_size,
            embedding_dim,
            hidden_dim,
            num_layers,
            num_classes,
            use_extra_mlp,
            mlp_ratio,
            mlp_dropout,
            recurrent_dropout,
            embedding_dropout,
        )
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
        self.eval()

    def from_pretrained(self, path):
        state_dict_pretrained = torch.load(path)["state_dict"]
        state_dict = {}
        for k, v in state_dict_pretrained.items():
            state_dict[k.replace("bilstm.", "")] = v
        del state_dict_pretrained
        self.load_state_dict(state_dict)

    def inference(self, text):
        sentence = self.tokenizer(text, return_tensors="pt")["input_ids"]
        _, out = self(sentence)
        return out


if __name__ == "__main__":
    config = OmegaConf.load("logs/ed/version_0/hparams.yaml")
    model_config = config.model.params
    model_config.pop("lr")
    model_config.pop("weight_decay")
    model = BiLSTMModelScope(**model_config)
    model.from_pretrained("logs/ed/version_0/epoch=29-val_acc=0.8834.ckpt")

    emotion = ["中性", "喜爱", "悲伤", "厌恶", "愤怒", "高兴"]
    # emotion = ["neutral", "happy", "angry", "sad", "fear", "surprise"]
    
    from load_data import ECGDataset

    # data = TextDataset("/mnt/d/codes/Swc_Data/ecg_data/ecg_train_data.json")
    # for idx, i in enumerate(data):
    #     out = model.inference(i["text"])
    #     print(i["text"], "\t", emotion[out.argmax(dim=-1).item()], out.argmax(dim=-1).item(), i["label"].item())
    #     if idx > 100:
    #         break

    while True:
        text = input("输出测试文本(输入y退出): ")
        if text == "y":
            break
        out = model.inference(text)
        print(
            text,
            "\t" * 2,
            emotion[out.argmax(dim=-1).item()],
            "\t" * 2,
            f"置信度: {out.max().item()} \n",
        )
