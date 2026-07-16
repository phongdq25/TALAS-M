from transformers import AutoModel, AutoTokenizer, PreTrainedModel
import torch
from torch import nn


def build_model(model_name, tokenizer_name, device) -> tuple[PreTrainedModel, AutoTokenizer]:
    model = AutoModel.from_pretrained(model_name, 
                                      device_map=device, 
                                      torch_dtype=torch.float32,
                                      output_hidden_states=True)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    return model, tokenizer


def build_projection_layer(num_layer, input_dim, output_dim, device) -> nn.ModuleList:
    proj_hidden_layers = nn.ModuleList()
    for _ in range(num_layer):
        proj_hidden_layers.append(nn.Linear(input_dim, output_dim, bias=False))
        with torch.no_grad():
            proj_hidden_layers[-1].weight.normal_(mean=0.0, std=1e-3)
    proj_hidden_layers = proj_hidden_layers.to(device)
    return proj_hidden_layers