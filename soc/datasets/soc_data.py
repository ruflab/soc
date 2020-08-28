ML_TYPE = {
    'regression': 0,
    'binary': 1,
    'categories': 2,
    'spatial_categories': 3,
}

###
# Fields
###

FIELDS = [
    'hexlayout',
    'numberlayout',
    'robberhex',
    'piecesonboard',
    'gamestate',
    'diceresult',
    'startingplayer',
    'currentplayer',
    'devcardsleft',
    'playeddevcard',
    'players',
]

DICE_RESULTS = {a: b for a, b in enumerate(range(2, 13))}
DICE_RESULTS[11] = -1  # Before the game start
DICE_RESULTS[12] = 0  # after the user start a turn, before he actually roll the dice
DICE_RESULT_SIZE = len(DICE_RESULTS)

GAME_PHASES = {
    'NEW': 0,
    'READY': 1,
    'SETOPTIONS_EXCL': 2,
    'SETOPTIONS_INCL': 3,
    'READY_RESET_WAIT_ROBOT_DISMISS': 4,
    'START1A': 5,
    'START1B': 6,
    'START2A': 10,
    'START2B': 11,
    'PLAY': 15,
    'PLAY1': 20,
    'PLAY1_LEGACY': 21,
    'PLACING_ROAD': 30,
    'PLACING_SETTLEMENT': 31,
    'PLACING_CITY': 32,
    'PLACING_ROBBER': 33,
    'PLACING_FREE_ROAD1': 40,
    'PLACING_FREE_ROAD2': 41,
    'WAITING_FOR_DISCARDS': 50,
    'WAITING_FOR_CHOICE': 51,
    'WAITING_FOR_DISCOVERY': 52,
    'WAITING_FOR_MONOPOLY': 53,
    'SPECIAL_BUILDING': 100,
    'OVER': 1000,
}
GAME_PHASES_NAMES = list(GAME_PHASES.keys())
GAME_PHASE_SIZE = len(GAME_PHASES)

ACTIONS = {
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
ACTIONS_NAMES = list(ACTIONS.keys())
ACTION_SIZE = len(ACTIONS)

STATE_COLS = {
    # Spatial
    'hexlayout': ML_TYPE['regression'],
    'numberlayout': ML_TYPE['regression'],
    'robberhex': ML_TYPE['spatial_categories'],
    'piecesonboard': ML_TYPE['binary'],  # Linear
    'gamestate': ML_TYPE['categories'],
    'diceresult': ML_TYPE['categories'],
    'startingplayer': ML_TYPE['categories'],
    'currentplayer': ML_TYPE['categories'],
    'devcardsleft': ML_TYPE['regression'],
    'playeddevcard': ML_TYPE['binary'],
    'players': ML_TYPE['regression'],
}

STATE_COLS_SIZE = {
    # Spatial
    'hexlayout': 1,
    'numberlayout': 1,
    'robberhex': 1,
    'piecesonboard': 4 * 18,
    # Linear
    'gamestate': GAME_PHASE_SIZE,
    'diceresult': DICE_RESULT_SIZE,
    'startingplayer': 4,
    'currentplayer': 4,
    'devcardsleft': 1,
    'playeddevcard': 1,
    'players': 4 * 41,
}

BOARD_SIZE = [7, 7]

SPATIAL_STATE_SIZE = sum([
    STATE_COLS_SIZE['hexlayout'],
    STATE_COLS_SIZE['numberlayout'],
    STATE_COLS_SIZE['robberhex'],
    STATE_COLS_SIZE['piecesonboard'],
])
STATE_SIZE = sum(STATE_COLS_SIZE.values())
