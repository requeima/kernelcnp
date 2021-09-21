from models import *
from utils import *
from loaders import *
from trainer import *

device = torch.device('cuda')

model = convGNPLinear()

model.to(device)
#model = nn.DataParallel(model)

train_ds = ValueExperimentDataset86(train=True)
val_ds = ValueExperimentDataset86(train=False)

train_loader = DataLoader(
    train_ds, batch_size = 128, shuffle = True)
val_loader = DataLoader(val_ds, batch_size = 128)

trainer = Trainer(model,
                  train_loader,
                  val_loader,
                  model.loss,
                  "./exps/convGNP_kvv_value_86/",
                  1e-4)

trainer.train(n_epochs=500)

