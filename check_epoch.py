import torch

device = torch.device("cuda:0")
e = 29
checkpoint1 = torch.load(f'./results/2/epoch{e}.pt', map_location=device)
checkpoint2 = torch.load(f'./results/5/epoch{e}.pt', map_location=device)


# 检查模型参数是否相同
param_diff = False
for key in checkpoint1:
    if key in checkpoint2:
        if not torch.all(torch.eq(checkpoint1[key], checkpoint2[key])):
            print(f"Parameter {key} is different.")
            param_diff = True
    else:
        print(f"Parameter {key} is missing in checkpoint2.")

for key in checkpoint2:
    if key not in checkpoint1:
        print(f"Parameter {key} is missing in checkpoint1.")

if not param_diff:
    print("Both checkpoints have the same parameters.")

# for key1, key2 in zip(checkpoint1, checkpoint2):
#     print(key1, key2)