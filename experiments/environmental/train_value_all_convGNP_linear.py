from models import *
from utils import *
from loaders import *
from trainer import *

print("Training convGNP linear VALUE all...")

torch.manual_seed(0)
np.random.seed(0)

device = torch.device('cuda')

model = convGNPLinear().cuda()

train_ds = ValueExperimentDatasetAll(data_dir="./data/", device=device, train=True)
val_ds = ValueExperimentDatasetAll(data_dir="./data/", device=device, train=False)

train_loader = DataLoader(
    train_ds, batch_size = 128, shuffle = True)
val_loader = DataLoader(val_ds, batch_size = 128)

trainer = Trainer(model,
                  train_loader,
                  val_loader,
                  model.loss,
                  "./exps_no_distance_scaling/convGNP_linear_value_all/",
                  1e-4)

trainer.train(n_epochs=500)
