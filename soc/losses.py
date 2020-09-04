import torch
import torch.nn.functional as F
from typing import List, Dict
from .typing import SocDataMetadata


def mse_loss(
    indexes: List[int],
    t1_logits_seq: torch.Tensor,
    t2_true_seq: torch.Tensor,
) -> torch.Tensor:
    start_i, end_i = indexes

    map_logits = t1_logits_seq[:, :, start_i:end_i]
    map_true = t2_true_seq[:, :, start_i:end_i]

    loss = F.mse_loss(map_logits, map_true)

    return loss


def gameturn_loss(
    indexes: List[int],
    t1_logits_seq: torch.Tensor,
    t2_true_seq: torch.Tensor,
) -> torch.Tensor:
    start_i, end_i = indexes

    gameturn_logits = t1_logits_seq[:, :, start_i]
    gameturn_true = t2_true_seq[:, :, start_i]

    # Regression losses need to be balanced with cross_entropy losses
    # To do so we add a coefficient for thos losses
    # The coefficient depends on the normalization applied
    # which defines how precise the output should be to make the right prediction
    coef = 20
    loss = coef * F.mse_loss(gameturn_logits, gameturn_true)

    return loss


def hexlayout_loss(
    indexes: List[int],
    t1_logits_seq: torch.Tensor,
    t2_true_seq: torch.Tensor,
) -> torch.Tensor:
    start_i, end_i = indexes

    hexlayout_logits = t1_logits_seq[:, :, start_i:end_i]
    hexlayout_true = t2_true_seq[:, :, start_i:end_i]

    # Regression losses need to be balanced with cross_entropy losses
    # To do so we add a coefficient for thos losses
    # The coefficient depends on the normalization applied
    # which defines how precise the output should be to make the right prediction
    coef = 20
    loss = coef * F.mse_loss(hexlayout_logits, hexlayout_true)

    return loss


def numberlayout_loss(
    indexes: List[int],
    t1_logits_seq: torch.Tensor,
    t2_true_seq: torch.Tensor,
) -> torch.Tensor:
    start_i, end_i = indexes

    numberlayout_logits = t1_logits_seq[:, :, start_i:end_i]
    numberlayout_true = t2_true_seq[:, :, start_i:end_i]

    # Regression losses need to be balanced with cross_entropy losses
    # To do so we add a coefficient for thos losses
    # The coefficient depends on the normalization applied
    # which defines how precise the output should be to make the right prediction
    coef = 20
    loss = coef * F.mse_loss(numberlayout_logits, numberlayout_true)

    return loss


def robber_loss(
    indexes: List[int],
    t1_logits_seq: torch.Tensor,
    t2_true_seq: torch.Tensor,
) -> torch.Tensor:
    bs = t1_logits_seq.shape[0]
    S = t1_logits_seq.shape[1]
    start_i, end_i = indexes

    robber_logits = t1_logits_seq[:, :, start_i].reshape(bs * S, -1)
    robber_true_pos = torch.argmax(t2_true_seq[:, :, start_i].reshape(bs * S, -1), dim=1)

    loss = F.cross_entropy(robber_logits, robber_true_pos)

    return loss


def pieces_loss(
    indexes: List[int],
    t1_logits_seq: torch.Tensor,
    t2_true_seq: torch.Tensor,
    compute_random_mask=False
) -> torch.Tensor:
    start_i, end_i = indexes

    pieces_logits = t1_logits_seq[:, :, start_i:end_i]
    pieces_true = t2_true_seq[:, :, start_i:end_i]

    if compute_random_mask:
        # We compute a mask to subsample zero values
        loss_mask = torch.zeros_like(pieces_true, device=pieces_true.device, requires_grad=False)
        loss_mask[pieces_true > 0] = 1
        random_c = torch.randperm(loss_mask.shape[2])[:2]
        loss_mask[:, :, random_c] = 1

        loss = F.binary_cross_entropy_with_logits(pieces_logits * loss_mask, pieces_true)
    else:
        loss = F.binary_cross_entropy_with_logits(pieces_logits, pieces_true)

    return loss


def gamestate_loss(
    indexes: List[int],
    t1_logits_seq: torch.Tensor,
    t2_true_seq: torch.Tensor,
) -> torch.Tensor:
    bs = t1_logits_seq.shape[0]
    S = t1_logits_seq.shape[1]
    start_i, end_i = indexes

    gamestate_logits = t1_logits_seq[:, :, start_i:end_i].reshape(bs * S, -1)
    gamestate_true = torch.argmax(t2_true_seq[:, :, start_i:end_i].reshape(bs * S, -1), dim=1)

    loss = F.cross_entropy(gamestate_logits, gamestate_true)

    return loss


def diceresult_loss(
    indexes: List[int],
    t1_logits_seq: torch.Tensor,
    t2_true_seq: torch.Tensor,
) -> torch.Tensor:
    bs = t1_logits_seq.shape[0]
    S = t1_logits_seq.shape[1]
    start_i, end_i = indexes

    diceresult_logits = t1_logits_seq[:, :, start_i:end_i].reshape(bs * S, -1)
    diceresult_true = torch.argmax(t2_true_seq[:, :, start_i:end_i].reshape(bs * S, -1), dim=1)

    loss = F.cross_entropy(diceresult_logits, diceresult_true)

    return loss


def startingplayer_loss(
    indexes: List[int],
    t1_logits_seq: torch.Tensor,
    t2_true_seq: torch.Tensor,
) -> torch.Tensor:
    bs = t1_logits_seq.shape[0]
    S = t1_logits_seq.shape[1]
    start_i, end_i = indexes

    startingplayer_logits = t1_logits_seq[:, :, start_i:end_i].reshape(bs * S, -1)
    startingplayer_true = torch.argmax(t2_true_seq[:, :, start_i:end_i].reshape(bs * S, -1), dim=1)

    loss = F.cross_entropy(startingplayer_logits, startingplayer_true)

    return loss


def currentplayer_loss(
    indexes: List[int],
    t1_logits_seq: torch.Tensor,
    t2_true_seq: torch.Tensor,
) -> torch.Tensor:
    bs = t1_logits_seq.shape[0]
    S = t1_logits_seq.shape[1]
    start_i, end_i = indexes

    currentplayer_logits = t1_logits_seq[:, :, start_i:end_i].reshape(bs * S, -1)
    currentplayer_true = torch.argmax(t2_true_seq[:, :, start_i:end_i].reshape(bs * S, -1), dim=1)

    loss = F.cross_entropy(currentplayer_logits, currentplayer_true)

    return loss


def devcardsleft_loss(
    indexes: List[int],
    t1_logits_seq: torch.Tensor,
    t2_true_seq: torch.Tensor,
) -> torch.Tensor:
    bs = t1_logits_seq.shape[0]
    S = t1_logits_seq.shape[1]
    start_i, end_i = indexes

    devcardsleft_logits = t1_logits_seq[:, :, start_i:end_i].reshape(bs * S, -1)
    devcardsleft_true = torch.argmax(t2_true_seq[:, :, start_i:end_i].reshape(bs * S, -1), dim=1)

    loss = F.cross_entropy(devcardsleft_logits, devcardsleft_true)

    return loss


def playeddevcard_loss(
    indexes: List[int],
    t1_logits_seq: torch.Tensor,
    t2_true_seq: torch.Tensor,
) -> torch.Tensor:
    start_i, end_i = indexes

    playeddevcard_logits = t1_logits_seq[:, :, start_i]
    playeddevcard_true = t2_true_seq[:, :, start_i]

    loss = F.binary_cross_entropy_with_logits(playeddevcard_logits, playeddevcard_true)

    return loss


def playersresources_loss(
    indexes: List[int],
    t1_logits_seq: torch.Tensor,
    t2_true_seq: torch.Tensor,
) -> torch.Tensor:
    start_i, end_i = indexes

    playersresources_logits = t1_logits_seq[:, :, start_i:end_i]
    playersresources_true = t2_true_seq[:, :, start_i:end_i]

    coef = 20
    loss = coef * F.mse_loss(playersresources_logits, playersresources_true)

    return loss


def players_loss(
    indexes: List[int],
    t1_logits_seq: torch.Tensor,
    t2_true_seq: torch.Tensor,
) -> torch.Tensor:
    start_i, end_i = indexes

    players_logits = t1_logits_seq[:, :, start_i:end_i]
    players_true = t2_true_seq[:, :, start_i:end_i]

    loss = F.mse_loss(players_logits, players_true)

    return loss


def actions_loss(
    indexes: List[int],
    t1_logits_seq: torch.Tensor,
    t2_true_seq: torch.Tensor,
) -> torch.Tensor:
    bs = t1_logits_seq.shape[0]
    S = t1_logits_seq.shape[1]
    start_i, end_i = indexes

    actions_logits = t1_logits_seq[:, :, start_i:end_i].reshape(bs * S, -1)
    actions_true = torch.argmax(t2_true_seq[:, :, start_i:end_i].reshape(bs * S, -1), dim=1)

    loss = F.cross_entropy(actions_logits, actions_true)

    return loss


def compute_losses(
    metadata: SocDataMetadata,
    t1_logits_seq: torch.Tensor,
    t2_true_seq: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    losses = {}
    for k, v in metadata.items():
        if k[:4] == 'mean':
            losses[k + '_loss'] = mse_loss(v, t1_logits_seq, t2_true_seq)  # type: ignore
        else:
            losses[k + '_loss'] = loss_mapping[k](v, t1_logits_seq, t2_true_seq)  # type: ignore

    return losses


loss_mapping = {
    'gameturn': gameturn_loss,
    'hexlayout': hexlayout_loss,
    'numberlayout': numberlayout_loss,
    'robberhex': robber_loss,
    'piecesonboard': pieces_loss,  # Linear
    'gamestate': gamestate_loss,
    'diceresult': diceresult_loss,
    'startingplayer': startingplayer_loss,
    'currentplayer': currentplayer_loss,
    'devcardsleft': devcardsleft_loss,
    'playeddevcard': playeddevcard_loss,
    'players': players_loss,
    'playersresources': playersresources_loss,
    'actions': actions_loss,
}
