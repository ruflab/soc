import torch
# from torch.nn.utils.rnn import PackedSequence
from typing import List, Tuple, Dict, Callable, Union

_TensorOrTensors = Union[torch.Tensor, List[torch.Tensor]]

SocDatasetItem = Tuple[torch.Tensor, torch.Tensor]
SocPolicyDatasetItem = Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
SocBatch = Tuple[torch.Tensor, torch.Tensor]
SocBatchMultipleOut = Tuple[torch.Tensor, List[torch.Tensor]]
SocSeqBatch = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
SocSeqPolicyBatch = Tuple[torch.Tensor,
                          Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                          Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
# SocSeqBatch = Tuple[PackedSequence, PackedSequence]
SocSeqList = List[SocDatasetItem]
SocSeqPolicyList = List[SocPolicyDatasetItem]
SocCollateFn = Callable[[SocSeqList], SocBatch]
SocDataMetadata = Dict[str, List[int]]
