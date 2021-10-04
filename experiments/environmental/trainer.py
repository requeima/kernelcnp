from matplotlib import pyplot as plt
import torch 
import scipy
import numpy as np 

class Trainer():
    """
    Training class for the neural process models
    """
    def __init__(self,
                 model,
                 train_loader,
                 val_loader,
                 loss_function,
                 save_path,
                 learning_rate,
                 np=False):
      
        # Model and data
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.save_path = save_path
        self.np = np

        # Training parameters
        self.opt = torch.optim.Adam(model.parameters(), lr=learning_rate) 
        self.loss_function = model.loss

        # Losses
        self.losses = []

    def plot_losses(self):
        """
        Plot losses for the trained model
        """
        plt.plot(np.array(self.losses))
        plt.xlabel("epoch")
        plt.ylabel("NLL")
        plt.show()

    def _unravel_to_numpy(self, x):
        return x.view(-1).detach().cpu().numpy()

    def get_val_stats_bs(self, predictions, targets):
        
        predictions = predictions.detach().cpu().numpy()
        targets = targets.detach().cpu().numpy()

        maes = np.zeros(predictions.shape[1])
        spearmans = np.zeros(predictions.shape[1])
        pearsons = np.zeros(predictions.shape[1])

        for st in range(predictions.shape[1]):
            # Get station
            true_mean = targets[:, st]
            pred_mean = predictions[:, st,0]
            # Remove nans
            pred_mean = pred_mean[~np.isnan(true_mean)]
            true_mean = true_mean[~np.isnan(true_mean)]

            try:
                maes[st] = np.mean(np.abs(true_mean - pred_mean))
                pearsons[st] = scipy.stats.pearsonr(pred_mean, true_mean)[0]
                spearmans[st] = scipy.stats.spearmanr(pred_mean, true_mean).correlation
            except:
                maes[st] = np.nan
                pearsons[st] = np.nan
                spearmans[st] = np.nan
                continue
        print("Mean absolute error: {}".format(np.nanmedian(maes)))
        print("Pearson correlation: {}".format(np.nanmedian(pearsons)))
        print("Spearman correlation: {}".format(np.nanmedian(spearmans)))

            
    def eval_epoch(self):

        self.model.eval()

        preds = []
        targets = []
        lf = []

        with torch.no_grad():
            for task in self.val_loader:
                task["dists"] = task["dists"][0,...]
                out = self.model(
                    task)
                if self.np:
                    preds.append(out[0,...])
                else:
                    preds.append(out)
                
                targets.append(task["y_target"])
                lf.append(-self.loss_function(task["y_target"], out).item())

        #concat full arrays
        preds = torch.cat(preds, dim = 0)
        targets = torch.cat(targets, dim = 0)

        # Get loss function
        log_loss = np.mean(np.array(lf))
        print("- Log loss: {}".format(log_loss))

        # Get stats
        self.get_val_stats_bs(preds, targets) 

        return log_loss

    def train(self, n_epochs = 100):

        # Init progress bar
        best_loss = 1000
        for epoch in range(n_epochs):

            print("Training epoch {}".format(epoch))
            self.model.train()
    
            for task in self.train_loader:
                task["dists"] = task["dists"][0,...]
                task, n = self.model.preprocess_function(task)

                out = self.model(task)
                loss = -self.loss_function(task["y_target"], out)/n

                loss.backward()
                self.opt.step()
                self.opt.zero_grad()
            epoch_loss = self.eval_epoch()
            if epoch_loss <= best_loss:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.opt.state_dict(),
                    'loss': epoch_loss
                    }, self.save_path+"epoch_{}".format(epoch))
                best_loss = epoch_loss

            self.losses.append(epoch_loss)
        np.save(self.save_path+"losses.npy", np.array(self.losses))
        print("Training complete!")





class TrainerNP():
    """
    Training class for the neural process models
    """
    def __init__(self,
                 model,
                 train_loader,
                 val_loader,
                 save_path,
                 learning_rate,
                 np=False):
      
        # Model and data
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.save_path = save_path
        self.np = np

        # Training parameters
        self.opt = torch.optim.Adam(model.parameters(), lr=learning_rate) 
        self.loss_function = self.loss

        # Losses
        self.losses = []

    def plot_losses(self):
        """
        Plot losses for the trained model
        """
        plt.plot(np.array(self.losses))
        plt.xlabel("epoch")
        plt.ylabel("NLL")
        plt.show()

    def loss(self, targets, out):
        na = targets.sum(dim=0).isnan()

        out = out[:,:,~na,:]
        targets = targets[:,~na]
        
        n_samples = out.shape[0]
        targets = targets.unsqueeze(0).repeat(n_samples, 1, 1)
        dist = Normal(out[...,0], out[...,1])

        lps = dist.log_prob(targets)
        # First sum over target points (should be zeros where masked so 
        #won't contribute)
        lps = torch.sum(lps, axis = 2) # shape (samples, batch)

        # Do logsumexp of the sums
        x = torch.logsumexp(lps, dim = 0) - np.log(out.shape[0]) # shape (batch)
        
        # Return negative mean over bastch dimension
        return torch.mean(x)

    def _unravel_to_numpy(self, x):
        return x.view(-1).detach().cpu().numpy()

    def get_val_stats_bs(self, predictions, targets):
        
        predictions = predictions.detach().cpu().numpy()
        targets = targets.detach().cpu().numpy()

        maes = np.zeros(predictions.shape[1])
        spearmans = np.zeros(predictions.shape[1])
        pearsons = np.zeros(predictions.shape[1])

        for st in range(predictions.shape[1]):
            # Get station
            true_mean = targets[:, st]
            pred_mean = predictions[:, st,0]
            # Remove nans
            pred_mean = pred_mean[~np.isnan(true_mean)]
            true_mean = true_mean[~np.isnan(true_mean)]

            try:
                maes[st] = np.mean(np.abs(true_mean - pred_mean))
                pearsons[st] = scipy.stats.pearsonr(pred_mean, true_mean)[0]
                spearmans[st] = scipy.stats.spearmanr(pred_mean, true_mean).correlation
            except:
                maes[st] = np.nan
                pearsons[st] = np.nan
                spearmans[st] = np.nan
                continue
        print("Mean absolute error: {}".format(np.nanmedian(maes)))
        print("Pearson correlation: {}".format(np.nanmedian(pearsons)))
        print("Spearman correlation: {}".format(np.nanmedian(spearmans)))

            
    def eval_epoch(self):

        self.model.eval()

        preds = []
        targets = []
        lf = []

        with torch.no_grad():
            for task in self.val_loader:
                task["dists"] = task["dists"][0,...]
                out = self.model(
                    task)
                if self.np:
                    preds.append(out[0,...])
                else:
                    preds.append(out)
                
                targets.append(task["y_target"])
                lf.append(-self.loss_function(task["y_target"], out).item())

        #concat full arrays
        preds = torch.cat(preds, dim = 0)
        targets = torch.cat(targets, dim = 0)

        # Get loss function
        log_loss = np.mean(np.array(lf))
        print("- Log loss: {}".format(log_loss))

        # Get stats
        self.get_val_stats_bs(preds, targets) 

        return log_loss

    def train(self, n_epochs = 100):

        # Init progress bar
        best_loss = 1000
        for epoch in range(n_epochs):

            print("Training epoch {}".format(epoch))
            self.model.train()
    
            for task in self.train_loader:
                task["dists"] = task["dists"][0,...]
                print("*********")
                print(task["dists"].shape)
                task, n = self.model.module.preprocess_function(task)
                print(task["dists"].shape)

                out = self.model(task)
                loss = -self.loss_function(task["y_target"], out)/n

                loss.backward()
                self.opt.step()
                self.opt.zero_grad()
            epoch_loss = self.eval_epoch()
            if epoch_loss <= best_loss:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.opt.state_dict(),
                    'loss': epoch_loss
                    }, self.save_path+"epoch_{}".format(epoch))
                best_loss = epoch_loss

            self.losses.append(epoch_loss)
        np.save(self.save_path+"losses.npy", np.array(self.losses))
        print("Training complete!")