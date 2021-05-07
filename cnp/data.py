import abc

import numpy as np
import stheno
import torch

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

        
    def generate_task(self):
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
        batch = {k: torch.tensor(_uprank(np.stack(v, axis=0)),
                                 dtype=torch.float32).to(self.device)
                 for k, v in batch.items()}

        return batch

    def __iter__(self):
        return LambdaIterator(lambda: self.generate_task(), self.iterations_per_epoch)


class GPGenerator(DataGenerator):
    """Generate samples from a GP with a given kernel.

    Further takes in keyword arguments for :class:`.data.DataGenerator`.

    Args:
        kernel (:class:`stheno.Kernel`, optional): Kernel to sample from.
            Defaults to an EQ kernel.
    """

    def __init__(self, kernel, std_noise, **kw_args):
        
        kernel = kernel + std_noise ** 2 * stheno.Delta()
        
        self.gp = stheno.GP(kernel)
        self.std_noise = std_noise
        
        DataGenerator.__init__(self, **kw_args)
        

    def sample(self, x):
        return np.squeeze(self.gp(x).sample())
    

    def log_like(self, x_context, y_context, x_target, y_target):
        
        to_numpy = lambda x: x.squeeze().cpu().numpy().astype(np.float64)
        
        # Convert torch tensors to numpy arrays
        x_context = to_numpy(x_context)
        y_context = to_numpy(y_context)
        
        x_target = to_numpy(x_target)
        y_target = to_numpy(y_target)
        
        
        prior = stheno.Measure()
        f = stheno.GP(self.gp.kernel, measure=prior)
        noise_1 = stheno.GP(self.std_noise ** 2 * stheno.Delta(), measure=prior)
        noise_2 = stheno.GP(self.std_noise ** 2 * stheno.Delta(), measure=prior)
        
        post = prior | ((f + noise_1)(x_context), y_context)
        
        return post(f + noise_2)(x_target).logpdf(y_target)


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