import torch
from typing import Optional, Dict, Union, List
from .typing import SocDataMetadata
from .datasets import utils as ds_utils

Num = Union[int, float]


def compare_by_idx(
    t1: torch.Tensor,
    t2: Union[torch.Tensor, Num],
    start_i: int,
    end_i: Optional[int] = None,
    dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    assert len(t1.shape) > 2  # [bs, S, C, H, W] or [bs, S, F]

    if end_i is None:
        if isinstance(t2, torch.Tensor):
            equal_tensor = (t1[:, :, start_i:] == t2[:, :, start_i:])
        else:
            equal_tensor = (t1[:, :, start_i:] == t2)
    elif start_i < end_i:
        if isinstance(t2, torch.Tensor):
            equal_tensor = (t1[:, :, start_i:end_i] == t2[:, :, start_i:end_i])
        else:
            equal_tensor = (t1[:, :, start_i:end_i] == t2)
    else:
        raise Exception(
            "start_i ({}) must be strictly smaller than end_i ({})".format(start_i, end_i)
        )

    return equal_tensor.type(dtype).mean()  # type: ignore


def get_stats(metadata: SocDataMetadata, x: torch.Tensor, y: Union[torch.Tensor,
                                                                   Num]) -> Dict[str, torch.Tensor]:
    dtype = x.dtype
    stats_dict = {}
    for label, indexes in metadata.items():
        stats_dict['acc_' + label] = compare_by_idx(x, y, indexes[0], indexes[1], dtype)

    return stats_dict


def hexlayout_acc(
    indexes: List[int],
    t1_logits_seq: torch.Tensor,
    t2_true_seq: torch.Tensor,
) -> torch.Tensor:
    start_i, end_i = indexes

    hexlayout_preds = ds_utils.unnormalize_hexlayout(t1_logits_seq[:, :, start_i:end_i])
    hexlayout_true = ds_utils.unnormalize_hexlayout(t2_true_seq[:, :, start_i:end_i])

    acc_eq = (hexlayout_preds == hexlayout_true)
    acc = acc_eq.type(t1_logits_seq.dtype).mean()  # type: ignore

    return acc


def numberlayout_acc(
    indexes: List[int],
    t1_logits_seq: torch.Tensor,
    t2_true_seq: torch.Tensor,
) -> torch.Tensor:
    start_i, end_i = indexes

    numberlayout_preds = ds_utils.unnormalize_numberlayout(t1_logits_seq[:, :, start_i:end_i])
    numberlayout_true = ds_utils.unnormalize_numberlayout(t2_true_seq[:, :, start_i:end_i])

    acc_eq = (numberlayout_preds == numberlayout_true)
    acc = acc_eq.type(t1_logits_seq.dtype).mean()  # type: ignore

    return acc


def robber_acc(
    indexes: List[int],
    t1_logits_seq: torch.Tensor,
    t2_true_seq: torch.Tensor,
) -> torch.Tensor:
    bs = t1_logits_seq.shape[0]
    S = t1_logits_seq.shape[1]
    start_i, end_i = indexes

    robber_pred_idx = torch.argmax(t1_logits_seq[:, :, start_i].reshape(bs * S, -1), dim=1)
    robber_true_pos = torch.argmax(t2_true_seq[:, :, start_i].reshape(bs * S, -1), dim=1)

    acc_eq = (robber_pred_idx == robber_true_pos)
    acc = acc_eq.type(t1_logits_seq.dtype).mean()  # type: ignore

    return acc


def pieces_acc(
    indexes: List[int],
    t1_logits_seq: torch.Tensor,
    t2_true_seq: torch.Tensor,
) -> torch.Tensor:
    start_i, end_i = indexes

    pieces_preds = torch.round(torch.sigmoid(t1_logits_seq[:, :, start_i:end_i]))
    pieces_true = t2_true_seq[:, :, start_i:end_i]

    acc_eq = (pieces_preds == pieces_true)
    acc = acc_eq.type(t1_logits_seq.dtype).mean()  # type: ignore

    return acc


def gamestate_acc(
    indexes: List[int],
    t1_logits_seq: torch.Tensor,
    t2_true_seq: torch.Tensor,
) -> torch.Tensor:
    bs = t1_logits_seq.shape[0]
    S = t1_logits_seq.shape[1]
    start_i, end_i = indexes

    gamestate_pred_idx = torch.argmax(t1_logits_seq[:, :, start_i:end_i].reshape(bs * S, -1), dim=1)
    gamestate_true_pos = torch.argmax(t2_true_seq[:, :, start_i:end_i].reshape(bs * S, -1), dim=1)

    acc_eq = (gamestate_pred_idx == gamestate_true_pos)
    acc = acc_eq.type(t1_logits_seq.dtype).mean()  # type: ignore

    return acc


def diceresult_acc(
    indexes: List[int],
    t1_logits_seq: torch.Tensor,
    t2_true_seq: torch.Tensor,
) -> torch.Tensor:
    bs = t1_logits_seq.shape[0]
    S = t1_logits_seq.shape[1]
    start_i, end_i = indexes

    diceresult_pred_idx = torch.argmax(
        t1_logits_seq[:, :, start_i:end_i].reshape(bs * S, -1), dim=1
    )
    diceresult_true_pos = torch.argmax(t2_true_seq[:, :, start_i:end_i].reshape(bs * S, -1), dim=1)

    acc_eq = (diceresult_pred_idx == diceresult_true_pos)
    acc = acc_eq.type(t1_logits_seq.dtype).mean()  # type: ignore

    return acc


def startingplayer_acc(
    indexes: List[int],
    t1_logits_seq: torch.Tensor,
    t2_true_seq: torch.Tensor,
) -> torch.Tensor:
    bs = t1_logits_seq.shape[0]
    S = t1_logits_seq.shape[1]
    start_i, end_i = indexes

    startingplayer_pred_idx = torch.argmax(
        t1_logits_seq[:, :, start_i:end_i].reshape(bs * S, -1), dim=1
    )
    startingplayer_true_pos = torch.argmax(
        t2_true_seq[:, :, start_i:end_i].reshape(bs * S, -1), dim=1
    )

    acc_eq = (startingplayer_pred_idx == startingplayer_true_pos)
    acc = acc_eq.type(t1_logits_seq.dtype).mean()  # type: ignore

    return acc


def currentplayer_acc(
    indexes: List[int],
    t1_logits_seq: torch.Tensor,
    t2_true_seq: torch.Tensor,
) -> torch.Tensor:
    bs = t1_logits_seq.shape[0]
    S = t1_logits_seq.shape[1]
    start_i, end_i = indexes

    currentplayer_pred_idx = torch.argmax(
        t1_logits_seq[:, :, start_i:end_i].reshape(bs * S, -1), dim=1
    )
    currentplayer_true_pos = torch.argmax(
        t2_true_seq[:, :, start_i:end_i].reshape(bs * S, -1), dim=1
    )

    acc_eq = (currentplayer_pred_idx == currentplayer_true_pos)
    acc = acc_eq.type(t1_logits_seq.dtype).mean()  # type: ignore

    return acc


def devcardsleft_acc(
    indexes: List[int],
    t1_logits_seq: torch.Tensor,
    t2_true_seq: torch.Tensor,
) -> torch.Tensor:
    start_i, end_i = indexes

    devcardsleft_preds = ds_utils.unnormalize_devcardsleft(t1_logits_seq[:, :, start_i:end_i])
    devcardsleft_true = ds_utils.unnormalize_devcardsleft(t2_true_seq[:, :, start_i:end_i])

    acc_eq = (devcardsleft_preds == devcardsleft_true)
    acc = acc_eq.type(t1_logits_seq.dtype).mean()  # type: ignore

    return acc


def playeddevcard_acc(
    indexes: List[int],
    t1_logits_seq: torch.Tensor,
    t2_true_seq: torch.Tensor,
) -> torch.Tensor:
    start_i, end_i = indexes

    playeddevcard_preds = torch.round(torch.sigmoid(t1_logits_seq[:, :, start_i]))
    playeddevcard_true = t2_true_seq[:, :, start_i]

    acc_eq = (playeddevcard_preds == playeddevcard_true)
    acc = acc_eq.type(t1_logits_seq.dtype).mean()  # type: ignore

    return acc


def players_acc(
    indexes: List[int],
    t1_logits_seq: torch.Tensor,
    t2_true_seq: torch.Tensor,
) -> torch.Tensor:
    start_i, end_i = indexes

    players_preds = torch.round(t1_logits_seq[:, :, start_i:end_i])
    players_true = t2_true_seq[:, :, start_i:end_i]

    acc_eq = (players_preds == players_true)
    acc = acc_eq.type(t1_logits_seq.dtype).mean()  # type: ignore

    return acc


def actions_acc(
    indexes: List[int],
    t1_logits_seq: torch.Tensor,
    t2_true_seq: torch.Tensor,
) -> torch.Tensor:
    bs = t1_logits_seq.shape[0]
    S = t1_logits_seq.shape[1]
    start_i, end_i = indexes

    actions_pred_idx = torch.argmax(t1_logits_seq[:, :, start_i:end_i].reshape(bs * S, -1), dim=1)
    actions_true_pos = torch.argmax(t2_true_seq[:, :, start_i:end_i].reshape(bs * S, -1), dim=1)

    acc_eq = (actions_pred_idx == actions_true_pos)
    acc = acc_eq.type(t1_logits_seq.dtype).mean()  # type: ignore

    return acc


def compute_accs(
    metadata: SocDataMetadata,
    t1_logits_seq: torch.Tensor,
    t2_true_seq: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    accs = {}
    for k, v in metadata.items():
        accs[k + '_acc'] = acc_mapping[k](v, t1_logits_seq, t2_true_seq)  # type: ignore

    return accs


acc_mapping = {
    'hexlayout': hexlayout_acc,
    'numberlayout': numberlayout_acc,
    'robberhex': robber_acc,
    'piecesonboard': pieces_acc,
    'gamestate': gamestate_acc,
    'diceresult': diceresult_acc,
    'startingplayer': startingplayer_acc,
    'currentplayer': currentplayer_acc,
    'devcardsleft': devcardsleft_acc,
    'playeddevcard': playeddevcard_acc,
    'players': players_acc,
    'actions': actions_acc,
}
