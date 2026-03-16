import torch.utils.data
import multiprocessing

# Force PyTorch to think workers are always 0
orig_init = torch.utils.data.DataLoader.__init__
def patched_init(self, *args, **kwargs):
    kwargs['num_workers'] = 0
    orig_init(self, *args, **kwargs)
torch.utils.data.DataLoader.__init__ = patched_init

# Force multiprocessing to use 'spawn' instead of 'fork' (safer for memory)
try:
    multiprocessing.set_start_method('spawn', force=True)
except:
    pass

print("🚀 GLOBAL LOCKDOWN: All DataLoaders forced to 0 workers.")
