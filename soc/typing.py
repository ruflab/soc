import torch
import numpy as np
from typing import List, Tuple, Dict, Any

SOCSeq = Tuple[np.ndarray, np.ndarray]
SOCSeqList = List[SOCSeq]
SOCSeqTorch = Tuple[torch.Tensor, torch.Tensor]

SocConfig = Dict[str, Any]
