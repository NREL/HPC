import numpy
import torch.distributed as dist
import os

# Example setup for distributed training (adjust for your specific environment)
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '29500'
os.environ['RANK'] = '0'
os.environ['WORLD_SIZE'] = '1'

try:
 dist.init_process_group(backend='nccl')
 print("Successfully initialized process group with NCCL backend.")
except RuntimeError as e:
 print(f"Failed to initialize process group with NCCL backend: {e}")
 print("This might indicate NCCL is not available or there's an issue with your CUDA setup.")
finally:
 if dist.is_initialized():
  dist.destroy_process_group()
