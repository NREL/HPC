import numpy
import torch
import torch.distributed as dist

def main():
 # Initialize the process group with MPI backend
 # rank and world_size are automatically inferred by MPI
 dist.init_process_group(backend='mpi')

 rank = dist.get_rank()
 world_size = dist.get_world_size()

 print(f"Hello from process {rank} (out of {world_size})!")

 # Your distributed training code goes here

 dist.destroy_process_group()

if __name__ == '__main__':
 main()
