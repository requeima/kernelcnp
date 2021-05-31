import abc
import multiprocessing
import threadpoolctl
from netCDF4 import Dataset

import numpy as np
import stheno
import torch
import lab as B
from matrix import Diagonal
from netCDF4 import Dataset
import time

import random
from datetime import datetime

__all__ = ['GPGenerator', 'SawtoothGenerator']


def x_sample_uniform(ranges, num_points, num_dim):
    
    assert len(ranges) == num_dim
    
    lower, upper = list(zip(*ranges))
    lower = np.array(lower)[None, :]
    upper = np.array(upper)[None, :]
    
    uniform = np.random.uniform(size=(num_points, num_dim))
    
    return lower + uniform * (upper - lower)


def _rand(val_range, *shape):
    lower, upper = val_range
    return lower + np.random.rand(*shape) * (upper - lower)


def _uprank(a):
    if len(a.shape) == 1:
        return a[:, None, None]
    elif len(a.shape) == 2:
        return a[:, :, None]
    elif len(a.shape) == 3:
        return a
    else:
        return ValueError(f'Incorrect rank {len(a.shape)}.')


class LambdaIterator:
    """Iterator that repeatedly generates elements from a lambda.

    Args:
        generator (function): Function that generates an element.
        
        num_elements (int): Number of elements to generate.
    """

    def __init__(self, generator, num_elements):
        self.generator = generator
        self.num_elements = num_elements
        self.index = 0

    def __next__(self):
        self.index += 1
        if self.index <= self.num_elements:
            return self.generator()
        else:
            raise StopIteration()

    def __iter__(self):
        return self


class DataGenerator(metaclass=abc.ABCMeta):
    """Data generator for GP samples.

    Args:
        
        iterations_per_epoch int: Number of iterations per epoch.
            One iteration corresponds to sampling one batch of tasks.
            
        batch_size int: Number of tasks in each batch sampled.
        
        x_range tuple[float]: Range of the inputs.
        
        max_num_context int: Number of training points. Must be at least 3.
        
        max_num_target int: Number of testing points. Must be at least 3.
    """

    def __init__(self,
                 iterations_per_epoch,
                 batch_size,
                 x_context_ranges,
                 max_num_context,
                 min_num_target,
                 max_num_target,
                 device,
                 x_target_ranges=None):
        
        assert 3 <= min_num_target <= max_num_target and max_num_context >= 3
        assert (x_target_ranges is None) or \
               (len(x_context_ranges) == len(x_target_ranges))
        
        self.iterations_per_epoch = iterations_per_epoch
        self.batch_size = batch_size
        self.max_num_context = max_num_context
        self.min_num_target = min_num_target
        self.max_num_target = max_num_target
        self.device = device
        
        self.num_dim = len(x_context_ranges)
        self.x_context_ranges = x_context_ranges
        self.x_target_ranges = x_context_ranges if x_target_ranges is None else \
                               x_target_ranges

        
    @abc.abstractmethod
    def sample(self, x):
        """Sample at inputs `x`.

        Args:
            x (vector): Inputs to sample at.

        Returns:
            vector: Sample at inputs `x`.
        """

        
    def generate_task(self, to_torch=True):
        """Generate a task.

        Returns:
            dict: A task, which is a dictionary with keys `x`, `y`, `x_context`,
                `y_context`, `x_target`, and `y_target.
        """
        batch = {'x'         : [],
                 'y'         : [],
                 'x_context' : [],
                 'y_context' : [],
                 'x_target'  : [],
                 'y_target'  : []}

        # Determine number of test and train points.
        num_context_points = np.random.randint(3, self.max_num_context+1)
        
        num_target_points = np.random.randint(self.min_num_target,
                                              self.max_num_target+1)

        for i in range(self.batch_size):
            
            # Sample context and target inputs and concatenate them
            x_context = x_sample_uniform(self.x_context_ranges,
                                         num_context_points,
                                         self.num_dim)
            
            x_target = x_sample_uniform(self.x_context_ranges,
                                         num_target_points,
                                         self.num_dim)
            
            x = np.concatenate([x_context, x_target], axis=0)
            
            # Sample context and target outputs together, then split them
            y = self.sample(x)
            
            y_context = y[:num_context_points]
            y_target = y[num_context_points:]

            # Put data in batch dictionary
            batch['x'].append(x)
            batch['y'].append(y)
            
            batch['x_context'].append(x_context)
            batch['y_context'].append(y_context)
            
            batch['x_target'].append(x_target)
            batch['y_target'].append(y_target)
            
        # Stack batch and convert to PyTorch
        batch = {k: _uprank(np.stack(v, axis=0)) for k, v in batch.items()}

        if to_torch:
            batch = {k: torch.tensor(v, dtype=torch.float32).to(self.device)
                     for k, v in batch.items()}

        return batch

    def __iter__(self):
        return LambdaIterator(lambda: self.generate_task(), self.iterations_per_epoch)

    def pregen_epoch(self):

        num_batches = self.iterations_per_epoch

        # Distribute the batches over the CPUs.
        num_cpus = multiprocessing.cpu_count()
        num_batches_per_cpu = [num_batches // num_cpus] * (num_cpus - 1)
        num_batches_per_cpu.append(num_batches - sum(num_batches_per_cpu))

        seeds = np.random.randint(0, int(1e9), size=(len(num_batches_per_cpu),))
        random_state = np.random.get_state()

        # Perform the pregeneration.
        with multiprocessing.Pool(processes=num_cpus) as pool:
            args = [(self, num, seeds[i]) \
                    for i, num in enumerate(num_batches_per_cpu)]
            batches = sum(pool.starmap(_generate_batches, args), [])

        batches = [
            {k: torch.tensor(v, dtype=torch.float32).to(self.device)
            for k, v in batch.items()} for batch in batches
        ]

        return batches


def _generate_batches(self, num_batches, seed):
    with threadpoolctl.threadpool_limits(limits=1, user_api="blas"):
        return [self.generate_task(to_torch=False) for _ in range(num_batches)]


class GPGenerator(DataGenerator):
    """Generate samples from a GP with a given kernel.

    Further takes in keyword arguments for :class:`.data.DataGenerator`.

    Args:
        kernel (:class:`stheno.Kernel`, optional): Kernel to sample from.
            Defaults to an EQ kernel.
    """

    def __init__(self, kernel, std_noise, **kw_args):
        
        kernel = kernel + std_noise ** 2 * stheno.Delta()
        
        self.kernel = kernel
        self.std_noise = std_noise
        
        DataGenerator.__init__(self, **kw_args)
        

    def sample(self, x):
        gp = stheno.GP(self.kernel)
        return np.squeeze(gp(x).sample())
    

    def log_like(self, x_context, y_context, x_target, y_target, diagonal=False):
        
        to_numpy = lambda x: x.squeeze().cpu().numpy().astype(np.float64)
        
        # Convert torch tensors to numpy arrays
        x_context = to_numpy(x_context)
        y_context = to_numpy(y_context)
        
        x_target = to_numpy(x_target)
        y_target = to_numpy(y_target)
        
        
        prior = stheno.Measure()
        f = stheno.GP(self.kernel, measure=prior)
        noise_1 = stheno.GP(self.std_noise ** 2 * stheno.Delta(), measure=prior)
        noise_2 = stheno.GP(self.std_noise ** 2 * stheno.Delta(), measure=prior)
        
        post = prior | ((f + noise_1)(x_context), y_context)
        ll = post(f + noise_2)(x_target).logpdf(y_target)
        
        if diagonal:
            fdd = post(f + noise_2)(x_target)
            diagonalised_fdd = stheno.Normal(fdd.mean, Diagonal(B.diag(fdd.var)))
            diag_ll = diagonalised_fdd.logpdf(y_target) 
        
            return ll, diag_ll
        else:
            return ll


class SawtoothGenerator(DataGenerator):
    """Generate samples from a random sawtooth.

    Further takes in keyword arguments for :class:`.data.DataGenerator`. The
    default numbers for `max_num_context` and `max_num_target` are 100.

    Args:
        freq_range (tuple[float], optional): Lower and upper bound for the
            random frequency.
        shift_range (tuple[float], optional): Lower and upper bound for the
            random shift.
        trunc_range (tuple[float], optional): Lower and upper bound for the
            random truncation.
    """

    def __init__(self,
                 iterations_per_epoch,
                 freq_range,
                 shift_range,
                 trunc_range,
                 max_num_context,
                 max_num_target,
                 **kw_args):
        
        self.freq_range = freq_range
        self.shift_range = shift_range
        self.trunc_range = trunc_range
        
        DataGenerator.__init__(self,
                               iterations_per_epoch=iterations_per_epoch,
                               max_num_context=max_num_context,
                               max_num_target=max_num_target,
                               **kw_args)
    
    
    def sample(self, x):
        
        # Sample parameters of sawtooth
        amp = 1
        freq = _rand(self.freq_range)
        shift = _rand(self.shift_range)
        trunc = np.random.randint(self.trunc_range[0], self.trunc_range[1] + 1)

        # Construct expansion.
        x = x[:, None, :] + shift
        k = np.arange(1, trunc+1)[None, :, None]
        
        y = 0.5 * amp - amp / np.pi * \
            np.sum((-1) ** k * np.sin(2 * np.pi * k * freq * x) / k, axis=1)
        
        return y





# =============================================================================
# Environmental dataloader
# =============================================================================

class EnvironmentalDataloader:

    def __init__(self,
                 path_to_dataset,
                 iterations_per_epoch,
                 min_num_context,
                 max_num_context,
                 min_num_target,
                 max_num_target,
                 num_datasets,
                 time_subsampler,
                 scale_inputs_by,
                 normalise_by):

        """
        Args:
            path_to_dataset (str)      : path to netcdf file (.nc) with the data
            iterations_per_epoch (int) : # iterations per epoch
            min_num_context (int)      : minimum number of context points
            max_num_context (int)      : maximum number of context points
            min_num_target (int)       : minimum number of target points
            max_num_target (int)       : maximum number of target points
            time_subsampler (lambda)   : which returns bool for each time point
            scale_inputs_by ((float,)  : tuple of floats for normalising inputs
            normalise_by    ((float,)  : tuple of floats for normalising outputs

        Note:
            The scale_inputs_by argument should be set to None for the training
            dataloader. The inputs will then be scaled to the range [-2., 2.],
            both for latitudes and logitudes, and the scaling parameters (for
            translation and stretching) will be stored in self.scale_by. Then,
            in the validation dataloaders you should set the argument
            self.scale_inputs_by=train_dataloader.scale_by. Similarly for output
            normalisation using normalise_by.
        """

        self.dataset = Dataset(path_to_dataset, 'r', format='NETCDF4')
        self.iterations_per_epoch = iterations_per_epoch

        self.min_num_context = min_num_context
        self.max_num_context = max_num_context
        self.min_num_context = min_num_context
        self.max_num_target = max_num_target
        self.num_datasets = num_datasets

        # Preprocessing for latitudes and logitudes
        lat = np.array(self.dataset.variables['latitude'])
        lon = np.array(self.dataset.variables['longitude'])

        lat, lon, scale_by = self.lat_lon_scale(lat=lat,
                                                lon=lon,
                                                scale_by=scale_inputs_by)
        self.lat = lat
        self.lon = lon
        self.scale_by = scale_by
        self.latlon_idx = np.meshgrid(np.arange(lat.shape[0]),
                                      np.arange(lon.shape[0]))
        self.latlon_idx = np.reshape(np.stack(self.latlon_idx, axis=-1),
                                     (-1, 2))

        # Preprocessing for times
        self.time_idx = np.arange(self.dataset.variables['time'].shape[0])
        self.time_idx = list(filter(time_subsampler, self.time_idx))

        # Only output variable is total percipitation for now
        variables, normalise_by = self.normalise(variables=["tp"],
                                                 normalise_by=normalise_by)
        self.variables = variables
        self.normalise_by = normalise_by


    def normalise(self, variables, normalise_by=None):
        
        variables = {variable : np.stack([self.dataset.variables[variable][t] \
                                          for t in self.time_idx], axis=0)    \
                     for variable in variables}

        if normalise_by is None:

            normalise_by = [(np.mean(variable), np.var(variable)**0.5) \
                            for name, variable in variables.items()]

        zipped = zip(variables.items(), normalise_by)
        variables = {name : (variable - mean) / std \
                     for (name, variable), (mean, std) in zipped}

        return variables, normalise_by
        

    def lat_lon_scale(self, lat, lon, scale_by=None):
        """
        Computes the translation and scaling amounts which convert the given
        longitude and latitude arrays to both be in the range lat_lon_range.
        """

        if scale_by is None:
        
            lat_min = lat.min()
            lat_max = lat.max()

            lon_min = lon.min()
            lon_max = lon.max()

            lat_trans = (lat_max + lat_min) / 2
            lat_scale = 0.5 * (lat_max - lat_min)

            lon_trans = (lon_max + lat_min) / 2
            lon_scale = 0.5 * (lon_max - lon_min)

            self.scale_by = (lat_trans, lat_scale, lon_trans, lon_scale)

        else:
            lat_trans, lat_scale, lon_trans, lon_scale = scale_by

        lat = (lat - lat_trans) / lat_scale
        lon = (lon - lon_trans) / lon_scale

        return lat, lon, (lat_trans, lat_scale, lon_trans, lon_scale)


    def __iter__(self):
        return LambdaIterator(lambda : self.generate_task(),
                              self.iterations_per_epoch)


    def generate_task(self):

        # Latitude and logitude resolutions
        num_lat = self.lat.shape[0]
        num_lon = self.lon.shape[0]

        # Dict to store sampled batch
        batch = {
            'x'         : [],
            'y'         : [],
            'x_context' : [],
            'y_context' : [],
            'x_target'  : [],
            'y_target'  : []
        }

        # Sample number of context and target points
        num_context = np.random.randint(1, self.max_num_context+1)
        num_target = np.random.randint(1, self.max_num_target+1)
        num_data = num_context + num_target

        total_num_points = self.latlon_idx.shape[0]
        total_num_times = len(self.time_idx)

        for i in range(self.num_datasets):
            
            # Sample indices for current batch (C + T, 2)
            _idx = np.random.choice(np.arange(total_num_points),
                                    size=(num_data,),
                                    replace=False)
            _idx = self.latlon_idx[_idx]

            # Slice out latitude and longitude values, stack to (C + T, 2)
            # These latitude and longitude values are already rescaled
            x = np.stack([self.lat[_idx[:, 0]], self.lon[_idx[:, 1]]], axis=-1)

            # Slice out output values to be predicted
            t = np.random.choice(np.arange(total_num_times))
            y = [variable[t][_idx[:, 0], _idx[:, 1]] \
                 for key, variable in self.variables.items()]
            y = np.stack(y, axis=-1)

            # Append results to lists in batch dict
            batch['x'].append(x)
            batch['y'].append(y)

            batch['x_context'].append(x[:num_context])
            batch['y_context'].append(y[:num_context])

            batch['x_target'].append(x[num_context:])
            batch['y_target'].append(y[num_context:])

        # Stack arrays and convert to tensors
        batch = {name : torch.tensor(np.stack(tensors, axis=0)).float() \
                 for name, tensors in batch.items()}

        return batch


