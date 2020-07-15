import re
import numpy as np
from typing import List, Tuple, Dict

# Types
IntVector = List[int]

# For some reasons those mappings do not face the same direction...
_complete_map = [
         0x17, 0x39, 0x5B, 0x7D,  # noqa
      0x15, 0x37, 0x59, 0x7B, 0x9D,  # noqa
   0x13, 0x35, 0x57, 0x79, 0x9B, 0xBD,  # noqa
0x11, 0x33, 0x55, 0x77, 0x99, 0xBB, 0xDD,  # noqa
    0x31, 0x53, 0x75, 0x97, 0xB9, 0xDB,  # noqa
        0x51, 0x73, 0x95, 0xB7, 0xD9,  # noqa
            0x71, 0x93, 0xB5, 0xD7  # noqa
]  # yapf: disable

_lands_hex_1d_mapping = {
    0x37: 5,
    0x59: 6,
    0x7B: 7,
    0x35: 10,
    0x57: 11,
    0x79: 12,
    0x9B: 13,
    0x33: 16,
    0x55: 17,
    0x77: 18,
    0x99: 19,
    0xBB: 20,
    0x53: 23,
    0x75: 24,
    0x97: 25,
    0xB9: 26,
    0x73: 29,
    0x95: 30,
    0xB7: 31
}

_lands_1d_hex_mapping = {v: k for k, v in _lands_hex_1d_mapping.items()}

_index_mapping_2d_1d = np.array([
    [0, 1, 2, 3, -1, -1, -1],
    [4, 5, 6, 7, 8, -1, -1],
    [9, 10, 11, 12, 13, 14, -1],
    [15, 16, 17, 18, 19, 20, 21],
    [-1, 22, 23, 24, 25, 26, 27],
    [-1, -1, 28, 29, 30, 31, 32],
    [-1, -1, -1, 33, 34, 35, 36]
])  # yapf: disable


_lands_adjacent_nodes_mapping = {
    k: [k + 0x01, k + 0x12, k + 0x21, k + 0x10, k - 0x01, k - 0x10]
    for k in _lands_hex_1d_mapping.keys()
}

_nodes_adjacent_lands_mapping: Dict[int, List] = {}
for land, nodes in _lands_adjacent_nodes_mapping.items():
    for node in nodes:
        if node not in _nodes_adjacent_lands_mapping.keys():
            _nodes_adjacent_lands_mapping[node] = [land]
        else:
            _nodes_adjacent_lands_mapping[node].append(land)

_lands_adjacent_segments_mapping = {
    k: [k + 0x01, k + 0x11, k + 0x10, k - 0x01, k - 0x11, k - 0x10]
    for k in _lands_hex_1d_mapping.keys()
}

_segments_adjacent_lands_mapping: Dict[int, List] = {}
for land, nodes in _lands_adjacent_segments_mapping.items():
    for node in nodes:
        if node not in _segments_adjacent_lands_mapping.keys():
            _segments_adjacent_lands_mapping[node] = [land]
        else:
            _segments_adjacent_lands_mapping[node].append(land)

_lands_building_rel_pos = {
    0x01: 0,
    0x12: 1,
    0x21: 2,
    0x10: 3,
    -0x01: 4,
    -0x10: 5,
}
_lands_road_rel_pos = {
    0x01: 0,
    0x11: 1,
    0x10: 2,
    -0x01: 3,
    -0x11: 4,
    -0x10: 5,
}


def parse_layout(raw_data: str) -> IntVector:
    if isinstance(raw_data, list):
        return raw_data

    data = raw_data[1:-1]
    data_arr = data.split(',')
    data_cleaned = [int(datum) for datum in data_arr]

    return data_cleaned


def mapping_1d_2d(data: IntVector) -> np.ndarray:
    assert len(data) == 37
    data_2d = np.array([
        data[:4] + [-1] * 3,
        data[4:9] + [-1] * 2,
        data[9:15] + [-1] * 1,
        data[15:22],
        [-1] * 1 + data[22:28],
        [-1] * 2 + data[28:33],
        [-1] * 3 + data[33:],
    ], dtype=np.int64)  # yapf: disable

    return data_2d[np.newaxis, :, :]


def mapping_2d_1d(data: np.ndarray) -> List:
    assert data.shape == (7, 7) or data.shape == (7, 7, 1)

    data_1d = data[data != -1]
    data_1d.flatten()

    return list(data_1d)


def get_1d_id(id_2d: Tuple) -> int:
    assert len(id_2d) == 2, '{} is not coordinate'.format(id_2d)
    assert id_2d[0] < 7
    assert id_2d[1] < 7

    id_1d = _index_mapping_2d_1d[id_2d]

    if id_1d == -1:
        raise ValueError('The 1d index of {} does not exist.'.format(id_2d))

    return id_1d


def get_2d_id(id_1d: int) -> Tuple:
    assert id_1d > -1
    assert id_1d < 37

    id_2d = np.where(_index_mapping_2d_1d == id_1d)

    return id_2d[0], id_2d[1]


def get_1d_id_from_hex(hex_i: int) -> int:
    if hex_i not in _lands_hex_1d_mapping.keys():
        raise ValueError('Hex number {} is missing'.format(hex_i))

    return _lands_hex_1d_mapping[hex_i]


def get_one_hot_plan(coord: Tuple) -> np.ndarray:
    plan = np.zeros([7, 7])
    plan[coord] = 1
    plan = plan[np.newaxis, :, :]

    return plan


def get_replicated_plan(i: int) -> np.ndarray:
    plan = np.ones([1, 7, 7]) * i

    return plan


def parse_pieces(pieces: str) -> np.ndarray:
    """
        Parse the java pieces string

        A java piece is represented s a list:
            - index 0: Type (road: 0, settlement: 1, city: 2)
            - index 1: coordinate (hex)
            - index 2: user id

        We represent the pieces as 18 plans for each player:
            - plan 0-5: road type
            - plan 6-11: settlement type
            - plan 12-17: city type
        Each plan is for an orientation around the hex tile:
            - For roads: NE, E, SE, SW, W, NW
            - For buildings: N, NE, SE, S, SW, NW
    """
    if isinstance(pieces, str) and pieces == '{}':
        return np.zeros([4 * 18, 7, 7])
    if isinstance(pieces, list) and len(pieces) == 0:
        return np.zeros([4 * 18, 7, 7])

    if isinstance(pieces, str):
        pieces = pieces[1:-1]
        pieces_arr = [piece[1:-1].split(',') for piece in re.findall(r'\{\d+,\d+,\d+\}', pieces)]
        pieces_cleaned = map(lambda piece_desc: [int(p) for p in piece_desc], pieces_arr)
    else:
        pieces_cleaned = pieces

    pieces_plans = np.zeros([4 * 18, 7, 7])
    for piece in pieces_cleaned:
        building_type = piece[0]
        piece_hex_coord = piece[1]
        player_id = piece[2]

        if building_type == 0:
            lands_hex = _segments_adjacent_lands_mapping[piece_hex_coord]

            lands_2d = map(lambda hex: get_2d_id(_lands_hex_1d_mapping[hex]), lands_hex)
            for i, id_2d in enumerate(lands_2d):
                current_land_hex = lands_hex[i]
                diff = piece_hex_coord - current_land_hex
                plan_id = player_id * 18 + building_type * 6 + _lands_road_rel_pos[diff]

                pieces_plans[plan_id, id_2d[0], id_2d[1]] = 1
        else:
            lands_hex = _nodes_adjacent_lands_mapping[piece_hex_coord]

            lands_1d = map(lambda hex: _lands_hex_1d_mapping[hex], lands_hex)
            lands_2d = map(lambda v_1d: get_2d_id(v_1d), lands_1d)
            for i, id_2d in enumerate(lands_2d):
                current_land_hex = lands_hex[i]
                diff = piece_hex_coord - current_land_hex
                plan_id = player_id * 18 + building_type * 6 + _lands_building_rel_pos[diff]

                pieces_plans[plan_id, id_2d[0], id_2d[1]] = 1

    return pieces_plans


def parse_player_infos(p_infos: str) -> np.ndarray:
    """
        Parse the JAVA representation for players
        4 arrays of 41 datum
        It is a list of int ordered as:
            - 0, player's ID (from the db, int)
            - 1, public vp (int)
            - 2, total vp (int)
            - 3, largest army (LA, bool)
            - 4, longest road (LR, bool)
            - 5, total number of development cards in hand (int)
            - 6, number of dev cards which represent a vp (int)

            - 7:11, all the unplayed dev cards
            - 12:16, all the newly bought dev cards
            - 17, how many knight (moving robber) cards played by this player

            - 18:23, number of each resource (clay, ore, sheep, wheat, wood, unknown)

            - 24:28, which resource types the player is touching
            - 19:34, which port types the player is touching
            - 35, roads left to build
            - 36, settlements left to build
            - 37, cities left to build

            - 38, how many road building cards played by this player
            - 39, how many monopoly cards played by this player
            - 40, how many discovery cards played by this player

        All booleans (LA, LR, touching stuff are represented in 1 for true or 0 for false).
    """
    if isinstance(p_infos, str):
        p_infos_separated = [
            re.sub(r'\{|\}', '', e).split(',') for e in re.findall(r'\{.*?\}', p_infos)
        ]
        p_infos_cleaned = [map(int, arr) for arr in p_infos_separated]
    else:
        p_infos_cleaned = p_infos

    all_player_infos = []
    for player_info in p_infos_cleaned:
        p_info = np.concatenate([get_replicated_plan(v) for v in player_info], axis=0)
        all_player_infos.append(p_info)

    return np.concatenate(all_player_infos, axis=0)


_ACTIONS = {
    'TRADE': 1.0,
    'ENDTURN': 2.0,
    'ROLL': 3.0,
    'BUILD': 4.0,
    'BUILDROAD': 4.1,
    'BUILDSETT': 4.2,
    'BUILDCITY': 4.3,
    'MOVEROBBER': 5.0,
    'CHOOSEPLAYER': 6.0,
    'DISCARD': 7.0,
    'BUYDEVCARD': 8.0,
    'PLAYDEVCARD': 9.0,
    'PLAYKNIGHT': 9.1,
    'PLAYMONO': 9.2,
    'PLAYDISC': 9.3,
    'PLAYROAD': 9.4,
    'WIN': 10.0,
}


def parse_actions(action: float):
    actions_plan = np.zeros([len(_ACTIONS), 7, 7])
    idx = list(_ACTIONS.values()).index(action)

    actions_plan[idx, :, :] = 1

    return actions_plan
