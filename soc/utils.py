import torch


def pad_collate_fn(inputs):
    """
        Pad the different inputs

        inputs is a list of (state_seq, actions_seq)
    """
    batch_states_seq = []
    batch_actions_seq = []
    for t in inputs:
        states_seq, actions_seq = t

        batch_states_seq.append(torch.tensor(states_seq))
        batch_actions_seq.append(torch.tensor(actions_seq))

    batch_states_seq_t = torch.nn.utils.rnn.pad_sequence(batch_states_seq, batch_first=True)
    batch_actions_seq_t = torch.nn.utils.rnn.pad_sequence(batch_actions_seq, batch_first=True)

    return batch_states_seq_t, batch_actions_seq_t
