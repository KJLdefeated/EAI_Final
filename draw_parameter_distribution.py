# import torch
# import matplotlib.pyplot as plt
# import transformers
# from transformers import AutoModelForCausalLM

# # model = torch.load('./0.9099_deit3_small_patch16_224.pth', map_location='cpu', weights_only=False)
# model_name = "meta-llama/Llama-3.2-1B-Instruct"   
# model = AutoModelForCausalLM.from_pretrained(
#     model_name,
#     torch_dtype=torch.float16,
#     device_map="cpu",
# )
# model.eval() 
# print(model)
# def plot_num_parameters_distribution(model):
#     num_parameters = dict()
#     for name, param in model.named_parameters():
#         if param.dim() > 1:
#             num_parameters[name] = param.numel()
#     fig = plt.figure(figsize=(20, 12))
#     plt.grid(axis='y')
#     plt.bar(list(num_parameters.keys()), list(num_parameters.values()))
#     plt.title('#Parameter Distribution')
#     plt.ylabel('Number of Parameters')
#     plt.xticks(rotation=60)
#     plt.tight_layout()
#     # plt.show()
#     plt.savefig('num_parameters_distribution_llama.png')

# plot_num_parameters_distribution(model)

import torch
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM

# 載入模型
model_name = "meta-llama/Llama-3.2-3B-Instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="cpu",
)
model.eval()

# model = torch.load('./0.9099_deit3_small_patch16_224.pth', map_location='cpu', weights_only=False)
# model.eval()

# 繪圖函式
def plot_num_parameters_distribution(model):
    num_parameters = {
        name: param.numel() for name, param in model.named_parameters()
        if param.dim() > 1
    }

    # 排序讓柱狀圖更易讀
    # sorted_params = sorted(num_parameters.items(), key=lambda x: x[1], reverse=True)
    sorted_params = num_parameters.items()
    names, values = zip(*sorted_params)

    plt.figure(figsize=(30, 12))
    plt.bar(range(len(values)), values)
    plt.yscale('log')  # 對數尺度讓差異層級更清楚
    plt.title('#Parameter Distribution ', fontsize=18)
    plt.ylabel('Number of Parameters ', fontsize=14)
    plt.xticks(ticks=range(len(names)), labels=names, rotation=90, fontsize=8)
    plt.grid(axis='y', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig('num_parameters_distribution_llama.png', dpi=300)

plot_num_parameters_distribution(model)
