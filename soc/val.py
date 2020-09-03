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
        stats_dict[label + '_acc'] = compare_by_idx(x, y, indexes[0], indexes[1], dtype)

    return stats_dict


def compute_field_acc_post_action(
    field_key: str,
    indexes: List[int],
    x_seq: torch.Tensor,
    y_a_true_seq: torch.Tensor,
    t1_logits_seq: torch.Tensor,
    t2_true_seq: torch.Tensor,
) -> torch.Tensor:
    # We are interested only in the last action of the history
    select = ds_utils.find_actions_idxs(x_seq, 'TRADE')
    select = select[:, -1:]

    if torch.all(torch.eq(select, False)):
        return torch.tensor(-1.)

    # We evaluate only the next state
    t1_logits_seq_trunc = t1_logits_seq[select, 0:1]
    t2_true_seq_trunc = t2_true_seq[select, 0:1]
    if select.sum() == 1:
        t1_logits_seq_trunc = t1_logits_seq_trunc.unsqueeze(0)
        t2_true_seq_trunc = t2_true_seq_trunc.unsqueeze(0)

    acc = acc_mapping[field_key](indexes, t1_logits_seq_trunc, t2_true_seq_trunc)

    return acc


def mean_acc(
    indexes: List[int],
    t1_logits_seq: torch.Tensor,
    t2_true_seq: torch.Tensor,
) -> torch.Tensor:
    start_i, end_i = indexes

    hexlayout_preds = torch.round(t1_logits_seq[:, :, start_i:end_i])
    hexlayout_true = t2_true_seq[:, :, start_i:end_i]

    acc_eq = (hexlayout_preds == hexlayout_true)
    acc = acc_eq.type(t1_logits_seq.dtype).mean()  # type: ignore

    return acc


def gameturn_acc(
    indexes: List[int],
    t1_logits_seq: torch.Tensor,
    t2_true_seq: torch.Tensor,
) -> torch.Tensor:
    start_i, end_i = indexes

    gameturn_preds = torch.round(t1_logits_seq[:, :, start_i])
    gameturn_true = t2_true_seq[:, :, start_i]

    acc_eq = (gameturn_preds == gameturn_true)
    acc = acc_eq.type(t1_logits_seq.dtype).mean()  # type: ignore

    return acc


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
    start_i, end_i = indexes

    gamestate_logits_seq = t1_logits_seq[:, :, start_i:end_i]
    gamestate_true_seq = t2_true_seq[:, :, start_i:end_i]
    if len(t1_logits_seq.shape) == 5:
        # We are dealing with spatial data, let's get back to linear data
        gamestate_logits_seq = gamestate_logits_seq.mean(dim=[3, 4])
        gamestate_true_seq = gamestate_true_seq.mean(dim=[3, 4])

    bs = t1_logits_seq.shape[0]
    S = t1_logits_seq.shape[1]
    gamestate_pred_idx = torch.argmax(gamestate_logits_seq.reshape(bs * S, -1), dim=1)
    gamestate_true_pos = torch.argmax(gamestate_true_seq.reshape(bs * S, -1), dim=1)

    acc_eq = (gamestate_pred_idx == gamestate_true_pos)
    acc = acc_eq.type(t1_logits_seq.dtype).mean()  # type: ignore

    return acc


def diceresult_acc(
    indexes: List[int],
    t1_logits_seq: torch.Tensor,
    t2_true_seq: torch.Tensor,
) -> torch.Tensor:
    start_i, end_i = indexes

    diceresult_logits_seq = t1_logits_seq[:, :, start_i:end_i]
    diceresult_true_seq = t2_true_seq[:, :, start_i:end_i]
    if len(t1_logits_seq.shape) == 5:
        # We are dealing with spatial data, let's get back to linear data
        diceresult_logits_seq = diceresult_logits_seq.mean(dim=[3, 4])
        diceresult_true_seq = diceresult_true_seq.mean(dim=[3, 4])

    bs = t1_logits_seq.shape[0]
    S = t1_logits_seq.shape[1]
    diceresult_pred_idx = torch.argmax(diceresult_logits_seq.reshape(bs * S, -1), dim=1)
    diceresult_true_pos = torch.argmax(diceresult_true_seq.reshape(bs * S, -1), dim=1)

    acc_eq = (diceresult_pred_idx == diceresult_true_pos)
    acc = acc_eq.type(t1_logits_seq.dtype).mean()  # type: ignore

    return acc


def startingplayer_acc(
    indexes: List[int],
    t1_logits_seq: torch.Tensor,
    t2_true_seq: torch.Tensor,
) -> torch.Tensor:
    start_i, end_i = indexes

    startingplayer_logits_seq = t1_logits_seq[:, :, start_i:end_i]
    startingplayer_true_seq = t2_true_seq[:, :, start_i:end_i]
    if len(t1_logits_seq.shape) == 5:
        # We are dealing with spatial data, let's get back to linear data
        startingplayer_logits_seq = startingplayer_logits_seq.mean(dim=[3, 4])
        startingplayer_true_seq = startingplayer_true_seq.mean(dim=[3, 4])

    bs = t1_logits_seq.shape[0]
    S = t1_logits_seq.shape[1]
    startingplayer_pred_idx = torch.argmax(startingplayer_logits_seq.reshape(bs * S, -1), dim=1)
    startingplayer_true_pos = torch.argmax(startingplayer_true_seq.reshape(bs * S, -1), dim=1)

    acc_eq = (startingplayer_pred_idx == startingplayer_true_pos)
    acc = acc_eq.type(t1_logits_seq.dtype).mean()  # type: ignore

    return acc


def currentplayer_acc(
    indexes: List[int],
    t1_logits_seq: torch.Tensor,
    t2_true_seq: torch.Tensor,
) -> torch.Tensor:
    start_i, end_i = indexes

    currentplayer_logits_seq = t1_logits_seq[:, :, start_i:end_i]
    currentplayer_true_seq = t2_true_seq[:, :, start_i:end_i]
    if len(t1_logits_seq.shape) == 5:
        # We are dealing with spatial data
        currentplayer_logits_seq = currentplayer_logits_seq.mean(dim=[3, 4])
        currentplayer_true_seq = currentplayer_true_seq.mean(dim=[3, 4])

    bs = t1_logits_seq.shape[0]
    S = t1_logits_seq.shape[1]
    currentplayer_pred_idx = torch.argmax(currentplayer_logits_seq.reshape(bs * S, -1), dim=1)
    currentplayer_true_pos = torch.argmax(currentplayer_true_seq.reshape(bs * S, -1), dim=1)

    acc_eq = (currentplayer_pred_idx == currentplayer_true_pos)
    acc = acc_eq.type(t1_logits_seq.dtype).mean()  # type: ignore

    return acc


def devcardsleft_acc(
    indexes: List[int],
    t1_logits_seq: torch.Tensor,
    t2_true_seq: torch.Tensor,
) -> torch.Tensor:
    start_i, end_i = indexes

    devcardsleft_logits_seq = t1_logits_seq[:, :, start_i:end_i]
    devcardsleft_true_seq = t2_true_seq[:, :, start_i:end_i]
    if len(t1_logits_seq.shape) == 5:
        # We are dealing with spatial data
        devcardsleft_logits_seq = devcardsleft_logits_seq.mean(dim=[3, 4])
        devcardsleft_true_seq = devcardsleft_true_seq.mean(dim=[3, 4])

    bs = t1_logits_seq.shape[0]
    S = t1_logits_seq.shape[1]
    devcardsleft_pred_idx = torch.argmax(devcardsleft_logits_seq.reshape(bs * S, -1), dim=1)
    devcardsleft_true_pos = torch.argmax(devcardsleft_true_seq.reshape(bs * S, -1), dim=1)

    acc_eq = (devcardsleft_pred_idx == devcardsleft_true_pos)
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


def playersresources_acc(
    indexes: List[int],
    t1_logits_seq: torch.Tensor,
    t2_true_seq: torch.Tensor,
) -> torch.Tensor:
    start_i, end_i = indexes

    playersresources_preds = torch.round(t1_logits_seq[:, :, start_i:end_i])
    playersresources_true = t2_true_seq[:, :, start_i:end_i]

    acc_eq = (playersresources_preds == playersresources_true)
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
    start_i, end_i = indexes

    actions_logits_seq = t1_logits_seq[:, :, start_i:end_i]
    actions_true_seq = t2_true_seq[:, :, start_i:end_i]
    if len(t1_logits_seq.shape) == 5:
        # We are dealing with spatial data, let's get back to linear data
        actions_logits_seq = actions_logits_seq.mean(dim=[3, 4])
        actions_true_seq = actions_true_seq.mean(dim=[3, 4])

    bs = t1_logits_seq.shape[0]
    S = t1_logits_seq.shape[1]
    actions_pred_idx = torch.argmax(actions_logits_seq.reshape(bs * S, -1), dim=1)
    actions_true_pos = torch.argmax(actions_true_seq.reshape(bs * S, -1), dim=1)

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
        if k[:4] == 'mean':
            accs[k + '_acc'] = mean_acc(v, t1_logits_seq, t2_true_seq)  # type: ignore
        else:
            accs[k + '_acc'] = acc_mapping[k](v, t1_logits_seq, t2_true_seq)  # type: ignore

    return accs


acc_mapping = {
    'gameturn': gameturn_acc,
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
    'playersresources': playersresources_acc,
    'players': players_acc,
    'actions': actions_acc,
}
