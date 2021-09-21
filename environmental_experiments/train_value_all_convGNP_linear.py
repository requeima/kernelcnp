from models import *
from utils import *
from loaders import *
from trainer import *

device = torch.device('cuda')

model = convGNPLinear()

model.to(device)

train_ds = ValueExperimentDatasetAll(train=True)
val_ds = ValueExperimentDatasetAll(train=False)

train_loader = DataLoader(
    train_ds, batch_size = 128, shuffle = True)
val_loader = DataLoader(val_ds, batch_size = 128)

trainer = Trainer(model,
                  train_loader,
                  val_loader,
                  model.loss,
                  "./exps/convGNP_linear_value_all/",
                  1e-4)

trainer.train(n_epochs=500)

