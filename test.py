import torch
import models_dwt.resnet

FILE = "Out/resnet50_dwt_bior2.2_32/resnet50_dwt_bior2.2_32_2023-03-07_17-22-52/resnet50_dwt_bior2.2_32_best.pth.tar"
model_checkpoint = torch.load(FILE)
epochs = model_checkpoint["epoch"]
model = models_dwt.resnet.resnet50(wavename="bior2.2")
model_dict = model.state_dict()

optimizer = torch.optim.SGD(model.parameters(), lr=0,
                                momentum = 0.9,
                                weight_decay = 1e-4)
best_acc1= model_checkpoint["best_acc1"]
arch = model_checkpoint["arch"]
# print(model.state_dict())

pretrained_dict = [(k, v) for k, v in model_checkpoint['state_dict'].items() if k in model_dict]
model_dict.update(pretrained_dict)
model.load_state_dict(model_dict)
optimizer.load_state_dict(model_checkpoint["optimizer"])
print(f'{best_acc1} {arch}')