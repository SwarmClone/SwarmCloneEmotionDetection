import torch
import pytorch_lightning as pl

from torch import nn
from transformers import PretrainedConfig, PreTrainedModel


class SWCBiLSTMConfig(PretrainedConfig):
    model_type = "swc_bilstm"
    is_composition = True

    def __init__(
        self,
        vocab_size=65536,
        embedding_dim=300,
        hidden_dim=256,
        num_layers=2,
        num_classes=6,
        use_extra_mlp=False,
        mlp_ratio=[4, 2],
        mlp_dropout=[0.5, 0.2],
        recurrent_dropout=0.2,
        embedding_dropout=0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.use_extra_mlp = use_extra_mlp
        self.mlp_ratio = mlp_ratio
        self.mlp_dropout = mlp_dropout
        self.recurrent_dropout = recurrent_dropout
        self.embedding_dropout = embedding_dropout


class SWCBiLSTMModel(PreTrainedModel, pl.LightningModule):
    config_class = SWCBiLSTMConfig
    base_model_prefix = "swc_bilstm"
    supports_gradient_checkpointing = True

    def __init__(self, config: SWCBiLSTMConfig):
        super().__init__(config)
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.embedding_dim)
        if config.embedding_dropout > 0:
            self.embedding_dropout = nn.Dropout(p=config.embedding_dropout)
        else:
            self.embedding_dropout = nn.Identity()
        self.lstm = nn.LSTM(
            config.embedding_dim,
            config.hidden_dim,
            config.num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=config.recurrent_dropout,
        )
        self.use_extra_mlp = config.use_extra_mlp
        if self.use_extra_mlp:
            # print(" * Using extra mlp with ratio: ", config.mlp_ratio)
            self.mlp = nn.Sequential(
                nn.Linear(
                    config.hidden_dim * 2, config.hidden_dim * config.mlp_ratio[0]
                ),
                nn.ReLU(),
                nn.Dropout(p=config.mlp_dropout[0]),
                nn.Linear(
                    config.hidden_dim * config.mlp_ratio[0],
                    config.hidden_dim * config.mlp_ratio[1],
                ),
                nn.ReLU(),
                nn.Dropout(p=config.mlp_dropout[1]),
                nn.Linear(config.hidden_dim * config.mlp_ratio[1], config.num_classes),
            )
        else:
            self.mlp = nn.Linear(config.hidden_dim * 2, config.num_classes)
        self.softmax = nn.Softmax(dim=1)
        self._init_weights()

    def forward(self, input_ids, labels=None):
        input_ids = self.embedding(input_ids)
        input_ids = self.embedding_dropout(input_ids)

        out, _ = self.lstm(input_ids)
        logits = self.mlp(out[:, -1, :])

        from transformers.modeling_outputs import SequenceClassifierOutput

        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits.view(-1, self.config.num_classes), labels.view(-1))

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None,
        )

    def _init_weights(self):
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)

        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param.data)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param.data)
            elif "bias" in name:
                param.data.fill_(0)
                n = param.size(0)
                param.data[(n // 4) : (n // 2)].fill_(1.0)
        if self.use_extra_mlp:
            for layer in self.mlp:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, 0)
        else:
            nn.init.xavier_uniform_(self.mlp.weight)
            nn.init.constant_(self.mlp.bias, 0)


if __name__ == "__main__":
    # from transformers import AutoTokenizer
    # from transformers import AutoModel
    # from transformers.models.auto.configuration_auto import CONFIG_MAPPING
    # from transformers.models.auto.modeling_auto import MODEL_MAPPING

    # tokenizer = AutoTokenizer.from_pretrained(
    #     "/home/momoia/codes/MiniLM2/models/tokenizers/tokenizer64k",
    #     trust_remote_code=True,
    # )

    # CONFIG_MAPPING.register("swc_bilstm", SWCBiLSTMConfig)
    # MODEL_MAPPING.register(SWCBiLSTMConfig, SWCBiLSTMModel)

    # from omegaconf import OmegaConf

    # config = OmegaConf.load("configs/bilstm.yaml")
    # config = OmegaConf.to_container(config, resolve=True)

    # config = SWCBiLSTMConfig(**config["model"]["params"])
    # model = SWCBiLSTMModel(config)

    # state_dict = torch.load("logs/ed/version_0/epoch=8-val_acc=0.9060.ckpt")[
    #     "state_dict"
    # ]
    # new_state_dict = {}
    # for k, v in state_dict.items():
    #     new_state_dict[k.replace("bilstm.", "")] = v
    # model.load_state_dict(new_state_dict)

    # model.save_pretrained("transformers")  # 保存配置和权重
    # loaded_model = AutoModel.from_pretrained("transformers")

    # from huggingface_hub import create_repo
    # repo_url = create_repo(
    #     repo_id="momoia/swc_bilstm",
    #     repo_type="model",
    #     exist_ok=True,
    # )

    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    tokenizer = AutoTokenizer.from_pretrained(
        "/home/momoia/codes/MiniLM2/models/tokenizers/tokenizer64k",
        trust_remote_code=True,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        "YamadaMano/SWCBiLSTM", trust_remote_code=True
    )
    input_ids = tokenizer("你好", return_tensors="pt", padding=True)["input_ids"]
    print(model.forward(input_ids))
