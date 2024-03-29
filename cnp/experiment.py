import os
import shutil
import time

import slugify
import torch

__all__ = ['generate_root',
           'save_checkpoint',
           'WorkingDirectory',
           'RunningAverage']


def generate_root(name):
    """Generate a root path.

    Args:
        name (str): Name of the experiment.

    Returns:

    """
    now = time.strftime('%Y-%m-%d_%H-%M-%S')
    return os.path.join('_experiments', f'{now}_{slugify.slugify(name)}')


def save_checkpoint(wd, state, is_best, epoch=None):
    """Save a checkpoint.

    Args:
        wd (:class:`.experiment.WorkingDirectory`): Working directory.
        state (dict): State to save.
        is_best (bool): This model is the best so far.
    """
    fn = wd.file('checkpoint.pth.tar')
    torch.save(state, fn)
    if is_best:
        fn_best = wd.file('model_best.pth.tar')
        shutil.copyfile(fn, fn_best)

    if epoch is not None:
        epoch_file = wd.file('last_epoch.txt')
        with open(epoch_file, 'w') as epoch_file_write:
            epoch_file_write.write(str(epoch + 1))




class WorkingDirectory:
    """Working directory.

    Args:
        root (str): Root of working directory.
        override (bool, optional): Delete working directory if it already
            exists. Defaults to `False`.
    """

    def __init__(self, root, override=False, print_root=True):
        self.root = root

        # Delete if the root already exists.
        if os.path.exists(self.root) and override:
            print('Experiment directory already exists. Overwriting.')
            shutil.rmtree(self.root)

        if print_root:
            print('Root:', self.root)

        # Create root directory.
        os.makedirs(self.root, exist_ok=True)

    def file(self, *name, exists=False):
        """Get the path of a file.

        Args:
            *name (str): Path to file, relative to the root directory. Use
                different arguments for directories.
            exists (bool): Assert that the file already exists. Defaults to
                `False`.

        Returns:
            str: Path to file.
        """
        path = os.path.join(self.root, *name)

        # Ensure that path exists.
        if exists and not os.path.exists(path):
            raise AssertionError('File "{}" does not exist.'.format(path))
        elif not exists:
            path_dir = os.path.join(self.root, *name[:-1])
            os.makedirs(path_dir, exist_ok=True)

        return path


class RunningAverage:
    """Maintain a running average."""

    def __init__(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def reset(self):
        """Reset the running average."""
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        """Update the running average.

        Args:
            val (float): Value to update with.
            n (int): Number elements used to compute `val`.
        """
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt

    
    
def log_args(wd, args):
    args_file = wd.file('args_file.txt')

    args_str = ""

    for arg in vars(args):
        args_str += f"{arg}: {getattr(args, arg)}"

        args_str += "\n"

    with open(args_file, 'w') as args_file_file_write:
        args_file_file_write.write(args_str)