import torch
# from torch.nn.utils.rnn import PackedSequence
from typing import List, Tuple, Dict, Callable, Any

SocDatasetItem = Tuple[torch.Tensor, torch.Tensor]
SocBatch = Tuple[torch.Tensor, torch.Tensor]
SocSeqBatch = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
# SocSeqBatch = Tuple[PackedSequence, PackedSequence]
SocSeqList = List[SocDatasetItem]
SocCollateFn = Callable[[SocSeqList], SocBatch]

SocConfig = Dict[str, Any]
