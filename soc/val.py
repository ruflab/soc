import torch
from typing import Optional, Dict, Union
from .typing import SocDataMetadata

Num = Union[int, float]


def compare_by_idx(
    t1: torch.Tensor,
    t2: Union[torch.Tensor, Num],
    start_i: int,
    end_i: Optional[int] = None,
    dtype: torch.dtype = torch.float32
) -> int:
    if end_i is None:
        if isinstance(t2, torch.Tensor):
            equal_tensor = (t1[:, start_i:] == t2[:, start_i:])
        else:
            equal_tensor = (t1[:, start_i:] == t2)
    elif start_i < end_i:
        if isinstance(t2, torch.Tensor):
            equal_tensor = (t1[:, start_i:end_i] == t2[:, start_i:end_i])
        else:
            equal_tensor = (t1[:, start_i:end_i] == t2)
    else:
        raise Exception(
            "start_i ({}) must be strictly smaller than end_i ({})".format(start_i, end_i)
        )

    return equal_tensor.type(dtype).mean()  # type: ignore


def get_stats(metadata: SocDataMetadata, x: torch.Tensor,
              y: Union[torch.Tensor, Num]) -> Dict[str, torch.Tensor]:
    dtype = x.dtype
    stats_dict = {}
    for label, idx_list in metadata.items():
        tmp = []
        for indexes in idx_list:
            tmp.append(compare_by_idx(x, y, indexes[0], indexes[1], dtype))
        stats_dict['acc_' + label] = torch.mean(torch.tensor(tmp))

    return stats_dict
