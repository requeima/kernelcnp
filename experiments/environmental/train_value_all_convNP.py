from models import *
from utils import *
from loaders import *
from trainer import *
import torch.nn as nn


print("Training convNP VALUE all...")

torch.manual_seed(0)
np.random.seed(0)

device = torch.device('cuda')

model = convNP(
    ls = 0.02,
    n_samples=24,
    n_latent_vars=32)

model.to(device)
model = nn.DataParallel(model)


train_ds = ValueExperimentDatasetAll(data_dir="./data/", device=device, train=True)
val_ds = ValueExperimentDatasetAll(data_dir="./data/", device=device, train=False)

train_loader = DataLoader(train_ds, batch_size = 16, shuffle = True)
val_loader = DataLoader(val_ds, batch_size = 16)

trainer = TrainerNP(model,
                  train_loader,
                  val_loader,
                  "./final_convNP/convNP_value_all/",
                  1e-4,
                  np=True)

trainer.train(n_epochs=500)

