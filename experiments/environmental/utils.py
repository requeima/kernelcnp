
import torch 
import numpy as np 

def channels_to_2nd_dim(x):
    return x.permute(*([0,x.dim()-1]+list(range(1,x.dim()- 1))))

def channels_to_last_dim(x):
    return x.permute(*([0]+list(range(2,x.dim()))+[1]))

def collapse_sample_dim(x):
    return x.view(-1, *x.shape[2:])

def cat_elev_dims(x, elev):
    batch = x.shape[0]
    elev = elev.unsqueeze(0).repeat(batch, 1, 1)
    return torch.cat([x,elev], dim=-1)

def log_exp(x):
    """
    Fix overflow
    """
    lt = torch.where(torch.exp(x)<1000)
    if lt[0].shape[0] > 0:
        x[lt] = torch.log(1+torch.exp(x[lt]))
    return x

def force_positive(x):
    return 0.01+ (1-0.1)*log_exp(x)

def get_dists(target_x, grid_x, grid_y):
    """
    Get the distances between the grid points and true points
    """
    # TODO: Fix this, don't need a loop

    # dimensions
    x_dim, y_dim = grid_x.shape

    # Init large scale grid
    total_grid = np.zeros((target_x.shape[0], x_dim, y_dim))
    count = 0

    for point in target_x:
        # Calculate distance from point to each gridpoint
        dists = (grid_x - point[0])**2+(grid_y - point[1])**2
        total_grid[count, :, :] = dists

        count += 1

    return total_grid

def cdist(x):
    quad = (x[..., :, None, :] - x[..., None, :, :]) ** 2
    quad = torch.sum(quad, axis=-1)
    return quad

def rbf_kernel(x, scales):
    # Pariwise squared Euclidean distances between embedding vectors
    dists = cdist(x)

    # Compute the RBF kernel, broadcasting appropriately
    return torch.exp(-0.5 * dists / scales ** 2)
    