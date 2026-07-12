import torch
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import os
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12355'
dist.init_process_group('gloo', rank=0, world_size=1)
m = torch.nn.Linear(10, 10)
m_ddp = DDP(m)
print('m.state_dict():', list(m.state_dict().keys()))
print('m_ddp.state_dict():', list(m_ddp.state_dict().keys()))
