from models import *
from utils import *
from loaders import *
from trainer import *

torch.manual_seed(0)
np.random.seed(0)

print("Training Germany convGNP KVV...")


device = torch.device('cuda')

model = convGNPKvv()

model.to(device)
#model = nn.DataParallel(model)

test_inds = np.load("data/test_inds.npy")
train_inds = np.load("data/train_inds.npy")

train_ds = GermanyRandomExperimentDataset(100, test_inds, train_inds, data_dir="./data/", device=device, train=True)
val_ds = GermanyRandomExperimentDataset(100, test_inds, train_inds, data_dir="./data/", device=device, train=False)

train_loader = DataLoader(
    train_ds, batch_size = 128, shuffle = True)
val_loader = DataLoader(val_ds, batch_size = 128)

trainer = Trainer(model,
                  train_loader,
                  val_loader,
                  model.loss,
                  "./exps_no_distance_scaling/convGNP_kvv_germany/",
                  1e-4)

trainer.train(n_epochs=500)

