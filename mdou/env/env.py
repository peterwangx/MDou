from collections import Counter
import numpy as np
import torch
import ctypes
from ctypes import *
import math

from mdou.env.game import GameEnv

Card2Column = {3: 0, 4: 1, 5: 2, 6: 3, 7: 4, 8: 5, 9: 6, 10: 7,
               11: 8, 12: 9, 13: 10, 14: 11, 17: 12, 20:13, 30:14}

NumOnes2Array = {0: np.array([0, 0, 0, 0]),
                 1: np.array([1, 0, 0, 0]),
                 2: np.array([1, 1, 0, 0]),
                 3: np.array([1, 1, 1, 0]),
                 4: np.array([1, 1, 1, 1])}

deck = []
for i in range(3, 15):
    deck.extend([i for _ in range(4)])
deck.extend([17 for _ in range(4)])
deck.extend([20, 30])

class Env:
    """
    Doudizhu multi-agent wrapper
    """
    def __init__(self, objective):
        """
        Objective is wp/adp/logadp. It indicates whether considers
        bomb in reward calculation. Here, we use dummy agents.
        This is because, in the orignial game, the players
        are `in` the game. Here, we want to isolate
        players and environments to have a more gym style
        interface. To achieve this, we use dummy players
        to play. For each move, we tell the corresponding
        dummy player which action to play, then the player
        will perform the actual action in the game engine.
        """
        self.objective = objective

        # Initialize players
        # We use three dummy player for the target position
        self.players = {}
        for position in ['landlord', 'landlord_up', 'landlord_down']:
            self.players[position] = DummyAgent(position)

        # Initialize the internal environment
        self._env = GameEnv(self.players)

        self.infoset = None

        isWindows = True
        import platform
        sysstr = platform.system()
        if (sysstr == "Windows"):
            isWindows =  True
        elif (sysstr == "Linux"):
            isWindows = False
        else:
            print("Other System ")
            isWindows = False

        if isWindows:
            self.libSplit = ctypes.WinDLL("./mdou/split/SplitDou.dll")
        else:
            self.libSplit = ctypes.cdll.LoadLibrary("./mdou/split/libSplit.so")

    def reset(self):
        """
        Every time reset is called, the environment
        will be re-initialized with a new deck of cards.
        This function is usually called when a game is over.
        """
        self._env.reset()

        # Randomly shuffle the deck
        _deck = deck.copy()
        np.random.shuffle(_deck)
        card_play_data = {'landlord': _deck[:20],
                          'landlord_up': _deck[20:37],
                          'landlord_down': _deck[37:54],
                          'three_landlord_cards': _deck[17:20],
                          }
        for key in card_play_data:
            card_play_data[key].sort()

        # Initialize the cards
        self._env.card_play_init(card_play_data)
        self.infoset = self._game_infoset

        return _get_obs_universal(self.infoset, self.libSplit, True, self.objective)

    def step(self, action):
        """
        Step function takes as input the action, which
        is a list of integers, and output the next obervation,
        reward, and a Boolean variable indicating whether the
        current game is finished. It also returns an empty
        dictionary that is reserved to pass useful information.
        """
        assert action in self.infoset.legal_actions
        self.players[self._acting_player_position].set_action(action)
        self._env.step()
        self.infoset = self._game_infoset
        done = False
        reward = 0.0
        if self._game_over:
            done = True
            reward = self._get_reward()
            obs = None
        else:
            obs = _get_obs_universal(self.infoset, self.libSplit, True, self.objective)
        return obs, reward, done, {}

    def _get_reward(self, pos=None):
        """
        This function is called in the end of each
        game. It returns either 1/-1 for win/loss,
        or ADP, i.e., every bomb will double the score.
        """
        winner = self._game_winner
        bomb_num = self._game_bomb_num
        if winner == 'landlord':
            if self.objective == 'adp':
                return 2.0 ** bomb_num
            elif self.objective == 'logadp':
                return bomb_num + 1.0
            else:
                return 1.0
        else:
            if self.objective == 'adp':
                return -2.0 ** bomb_num
            elif self.objective == 'logadp':
                return -bomb_num - 1.0
            else:
                return -1.0

    @property
    def _game_infoset(self):
        """
        Here, inforset is defined as all the information
        in the current situation, incuding the hand cards
        of all the players, all the historical moves, etc.
        That is, it contains perferfect infomation. Later,
        we will use functions to extract the observable
        information from the views of the three players.
        """
        return self._env.game_infoset

    @property
    def _game_bomb_num(self):
        """
        The number of bombs played so far. This is used as
        a feature of the neural network and is also used to
        calculate ADP.
        """
        return self._env.get_bomb_num()

    @property
    def _game_winner(self):
        """ A string of landlord/peasants
        """
        return self._env.get_winner()

    @property
    def _acting_player_position(self):
        """
        The player that is active. It can be landlord,
        landlod_down, or landlord_up.
        """
        return self._env.acting_player_position

    @property
    def _game_over(self):
        """ Returns a Boolean
        """
        return self._env.game_over

class DummyAgent(object):
    """
    Dummy agent is designed to easily interact with the
    game engine. The agent will first be told what action
    to perform. Then the environment will call this agent
    to perform the actual action. This can help us to
    isolate environment and agents towards a gym like
    interface.
    """
    def __init__(self, position):
        self.position = position
        self.action = None

    def act(self, infoset):
        """
        Simply return the action that is set previously.
        """
        assert self.action in infoset.legal_actions
        return self.action

    def set_action(self, action):
        """
        The environment uses this function to tell
        the dummy agent what to do.
        """
        self.action = action

def get_obs(infoset):
    """
    This function obtains observations with imperfect information
    from the infoset. It has three branches since we encode
    different features for different positions.
    
    This function will return dictionary named `obs`. It contains
    several fields. These fields will be used to train the model.
    One can play with those features to improve the performance.

    `position` is a string that can be landlord/landlord_down/landlord_up

    `x_batch` is a batch of features (excluding the hisorical moves).
    It also encodes the action feature

    `z_batch` is a batch of features with hisorical moves only.

    `legal_actions` is the legal moves

    `x_no_action`: the features (exluding the hitorical moves and
    the action features). It does not have the batch dim.

    `z`: same as z_batch but not a batch.
    """
    if infoset.player_position == 'landlord':
        return _get_obs_landlord(infoset)
    elif infoset.player_position == 'landlord_up':
        return _get_obs_landlord_up(infoset)
    elif infoset.player_position == 'landlord_down':
        return _get_obs_landlord_down(infoset)
    else:
        raise ValueError('')

def _get_one_hot_array(num_left_cards, max_num_cards):
    """
    A utility function to obtain one-hot endoding
    """
    one_hot = np.zeros(max_num_cards)
    one_hot[num_left_cards - 1] = 1

    return one_hot

def _cards2array(list_cards):
    """
    A utility function that transforms the actions, i.e.,
    A list of integers into card matrix. Here we remove
    the six entries that are always zero and flatten the
    the representations.
    """
    if len(list_cards) == 0:
        return np.zeros(54, dtype=np.int8)

    matrix = np.zeros([4, 13], dtype=np.int8)
    jokers = np.zeros(2, dtype=np.int8)
    counter = Counter(list_cards)
    for card, num_times in counter.items():
        if card < 20:
            matrix[:, Card2Column[card]] = NumOnes2Array[num_times]
        elif card == 20:
            jokers[0] = 1
        elif card == 30:
            jokers[1] = 1
    return np.concatenate((matrix.flatten('F'), jokers))

def _cards2vector(list_cards):
    if len(list_cards) == 0:
        return np.zeros(15, dtype=np.int8)

    vector = np.zeros(15, dtype=np.int8)
    counter = Counter(list_cards)
    for card, num_times in counter.items():
        vector[Card2Column[card]] = num_times

    return vector

def _action_seq_list2array(action_seq_list):
    """
    A utility function to encode the historical moves.
    We encode the historical 15 actions. If there is
    no 15 actions, we pad the features with 0. Since
    three moves is a round in DouDizhu, we concatenate
    the representations for each consecutive three moves.
    Finally, we obtain a 5x162 matrix, which will be fed
    into LSTM for encoding.
    """
    action_seq_array = np.zeros((len(action_seq_list), 54))
    for row, list_cards in enumerate(action_seq_list):
        action_seq_array[row, :] = _cards2array(list_cards)
    action_seq_array = action_seq_array.reshape(5, 162)
    return action_seq_array

def _action_seq_list2simplearray(action_seq_list):
    action_seq_array = np.zeros((len(action_seq_list), 15))
    for row, list_cards in enumerate(action_seq_list):
        action_seq_array[row, :] = _cards2vector(list_cards)

    return action_seq_array

def _action_seq_list2vector(action_seq_list):
    action_seq_array = np.zeros((len(action_seq_list),15))
    for row, list_cards in enumerate(action_seq_list):
        action_seq_array[row, :] = _cards2vector(list_cards)
    action_seq_array = action_seq_array.reshape(5, 45)
    return action_seq_array

#The latest nine actions, ([identity, camp, number of cards left], cards played)
#9*2*54--->18*54
def _action_seq_list2universalarray(player_act_seq, action_seq_list):
    action_seq_array = np.zeros((len(action_seq_list)*2, 54))

    for row, list_cards in enumerate(action_seq_list):
        if (player_act_seq[row] == []):
            continue

        left = []
        if (player_act_seq[row]['position'] == 'landlord'):
            left = _get_one_hot_array(player_act_seq[row]['num'], 20)
            action_seq_array[row * 2, :] = np.hstack((np.tile([0, 0], 9), np.ones(16), left))
        elif (player_act_seq[row]['position'] == 'landlord_down'):
            left = _get_one_hot_array(player_act_seq[row]['num'], 17)
            action_seq_array[row * 2, :] = np.hstack((np.tile([0, 1], 9), np.zeros(19), left))
        elif (player_act_seq[row]['position'] == 'landlord_up'):
            left = _get_one_hot_array(player_act_seq[row]['num'], 17)
            action_seq_array[row * 2, :] = np.hstack((np.tile([1, 0], 9), np.zeros(19), left))
        else:
            raise ValueError("error")

        action_seq_array[row* 2 + 1, :] = _cards2array(list_cards)

    return action_seq_array

def _process_action_seq(sequence, length=15):
    """
    A utility function encoding historical moves. We
    encode 15 moves. If there is no 15 moves, we pad
    with zeros.
    """
    sequence = sequence[-length:].copy()
    if len(sequence) < length:
        empty_sequence = [[] for _ in range(length - len(sequence))]
        empty_sequence.extend(sequence)
        sequence = empty_sequence
    return sequence


def _process_state_seq(sequence, length=8):
    """
    A utility function encoding historical states. We
    encode 15 states. If there is no 15 states, we pad
    with zeros.
    """
    sequence = sequence[-length:].copy()
    if len(sequence) < length:
        empty_sequence = [[] for _ in range(length - len(sequence))]
        empty_sequence.extend(sequence)
        sequence = empty_sequence
    return sequence

def _state_seq_list2array(state_seq_list, dim):
    """
    A utility function to encode the historical states.
    We encode the historical 8 states. If there is
    no 8 states, we pad the features with 0. Since
    the dim of landloard states is 120, while the dim of
    farmer is 150, which will be fed into LSTM for encoding.
    """
    state_seq_array = np.zeros((len(state_seq_list), dim))
    for row, list_cards in enumerate(state_seq_list):
        if len(list_cards) == 0:
            state_seq_array[row, :] = np.zeros(dim, dtype=np.int8)
        else:
            state_seq_array[row, :] = np.copy(list_cards)

    return state_seq_array

def _get_one_hot_bomb(bomb_num):
    """
    A utility function to encode the number of bombs
    into one-hot representation.
    """
    one_hot = np.zeros(15)
    one_hot[bomb_num] = 1
    return one_hot

def _get_obs_landlord(infoset):
    """
    Obttain the landlord features. See Table 4 in
    https://arxiv.org/pdf/2106.06135.pdf
    """
    num_legal_actions = len(infoset.legal_actions)
    my_handcards = _cards2array(infoset.player_hand_cards)
    my_handcards_batch = np.repeat(my_handcards[np.newaxis, :],
                                   num_legal_actions, axis=0)

    other_handcards = _cards2array(infoset.other_hand_cards)
    other_handcards_batch = np.repeat(other_handcards[np.newaxis, :],
                                      num_legal_actions, axis=0)

    last_action = _cards2array(infoset.last_move)
    last_action_batch = np.repeat(last_action[np.newaxis, :],
                                  num_legal_actions, axis=0)

    my_action_batch = np.zeros(my_handcards_batch.shape)
    for j, action in enumerate(infoset.legal_actions):
        my_action_batch[j, :] = _cards2array(action)

    landlord_up_num_cards_left = _get_one_hot_array(
        infoset.num_cards_left_dict['landlord_up'], 17)
    landlord_up_num_cards_left_batch = np.repeat(
        landlord_up_num_cards_left[np.newaxis, :],
        num_legal_actions, axis=0)

    landlord_down_num_cards_left = _get_one_hot_array(
        infoset.num_cards_left_dict['landlord_down'], 17)
    landlord_down_num_cards_left_batch = np.repeat(
        landlord_down_num_cards_left[np.newaxis, :],
        num_legal_actions, axis=0)

    landlord_up_played_cards = _cards2array(
        infoset.played_cards['landlord_up'])
    landlord_up_played_cards_batch = np.repeat(
        landlord_up_played_cards[np.newaxis, :],
        num_legal_actions, axis=0)

    landlord_down_played_cards = _cards2array(
        infoset.played_cards['landlord_down'])
    landlord_down_played_cards_batch = np.repeat(
        landlord_down_played_cards[np.newaxis, :],
        num_legal_actions, axis=0)

    bomb_num = _get_one_hot_bomb(
        infoset.bomb_num)
    bomb_num_batch = np.repeat(
        bomb_num[np.newaxis, :],
        num_legal_actions, axis=0)

    x_batch = np.hstack((my_handcards_batch,
                         other_handcards_batch,
                         last_action_batch,
                         landlord_up_played_cards_batch,
                         landlord_down_played_cards_batch,
                         landlord_up_num_cards_left_batch,
                         landlord_down_num_cards_left_batch,
                         bomb_num_batch,
                         my_action_batch))
    x_no_action = np.hstack((my_handcards,
                             other_handcards,
                             last_action,
                             landlord_up_played_cards,
                             landlord_down_played_cards,
                             landlord_up_num_cards_left,
                             landlord_down_num_cards_left,
                             bomb_num))
    z = _action_seq_list2array(_process_action_seq(
        infoset.card_play_action_seq))
    z_batch = np.repeat(
        z[np.newaxis, :, :],
        num_legal_actions, axis=0)
    obs = {
            'position': 'landlord',
            'x_batch': x_batch.astype(np.float32),
            'z_batch': z_batch.astype(np.float32),
            'legal_actions': infoset.legal_actions,
            'x_no_action': x_no_action.astype(np.int8),
            'z': z.astype(np.int8),
          }
    return obs

def _get_obs_landlord_up(infoset):
    """
    Obttain the landlord_up features. See Table 5 in
    https://arxiv.org/pdf/2106.06135.pdf
    """
    num_legal_actions = len(infoset.legal_actions)
    my_handcards = _cards2array(infoset.player_hand_cards)
    my_handcards_batch = np.repeat(my_handcards[np.newaxis, :],
                                   num_legal_actions, axis=0)

    other_handcards = _cards2array(infoset.other_hand_cards)
    other_handcards_batch = np.repeat(other_handcards[np.newaxis, :],
                                      num_legal_actions, axis=0)

    last_action = _cards2array(infoset.last_move)
    last_action_batch = np.repeat(last_action[np.newaxis, :],
                                  num_legal_actions, axis=0)

    my_action_batch = np.zeros(my_handcards_batch.shape)
    for j, action in enumerate(infoset.legal_actions):
        my_action_batch[j, :] = _cards2array(action)

    last_landlord_action = _cards2array(
        infoset.last_move_dict['landlord'])
    last_landlord_action_batch = np.repeat(
        last_landlord_action[np.newaxis, :],
        num_legal_actions, axis=0)
    landlord_num_cards_left = _get_one_hot_array(
        infoset.num_cards_left_dict['landlord'], 20)
    landlord_num_cards_left_batch = np.repeat(
        landlord_num_cards_left[np.newaxis, :],
        num_legal_actions, axis=0)

    landlord_played_cards = _cards2array(
        infoset.played_cards['landlord'])
    landlord_played_cards_batch = np.repeat(
        landlord_played_cards[np.newaxis, :],
        num_legal_actions, axis=0)

    last_teammate_action = _cards2array(
        infoset.last_move_dict['landlord_down'])
    last_teammate_action_batch = np.repeat(
        last_teammate_action[np.newaxis, :],
        num_legal_actions, axis=0)
    teammate_num_cards_left = _get_one_hot_array(
        infoset.num_cards_left_dict['landlord_down'], 17)
    teammate_num_cards_left_batch = np.repeat(
        teammate_num_cards_left[np.newaxis, :],
        num_legal_actions, axis=0)

    teammate_played_cards = _cards2array(
        infoset.played_cards['landlord_down'])
    teammate_played_cards_batch = np.repeat(
        teammate_played_cards[np.newaxis, :],
        num_legal_actions, axis=0)

    bomb_num = _get_one_hot_bomb(
        infoset.bomb_num)
    bomb_num_batch = np.repeat(
        bomb_num[np.newaxis, :],
        num_legal_actions, axis=0)

    x_batch = np.hstack((my_handcards_batch,
                         other_handcards_batch,
                         landlord_played_cards_batch,
                         teammate_played_cards_batch,
                         last_action_batch,
                         last_landlord_action_batch,
                         last_teammate_action_batch,
                         landlord_num_cards_left_batch,
                         teammate_num_cards_left_batch,
                         bomb_num_batch,
                         my_action_batch))
    x_no_action = np.hstack((my_handcards,
                             other_handcards,
                             landlord_played_cards,
                             teammate_played_cards,
                             last_action,
                             last_landlord_action,
                             last_teammate_action,
                             landlord_num_cards_left,
                             teammate_num_cards_left,
                             bomb_num))
    z = _action_seq_list2array(_process_action_seq(
        infoset.card_play_action_seq))
    z_batch = np.repeat(
        z[np.newaxis, :, :],
        num_legal_actions, axis=0)
    obs = {
            'position': 'landlord_up',
            'x_batch': x_batch.astype(np.float32),
            'z_batch': z_batch.astype(np.float32),
            'legal_actions': infoset.legal_actions,
            'x_no_action': x_no_action.astype(np.int8),
            'z': z.astype(np.int8),
          }
    return obs

def _get_obs_landlord_down(infoset):
    """
    Obttain the landlord_down features. See Table 5 in
    https://arxiv.org/pdf/2106.06135.pdf
    """
    num_legal_actions = len(infoset.legal_actions)
    my_handcards = _cards2array(infoset.player_hand_cards)
    my_handcards_batch = np.repeat(my_handcards[np.newaxis, :],
                                   num_legal_actions, axis=0)

    other_handcards = _cards2array(infoset.other_hand_cards)
    other_handcards_batch = np.repeat(other_handcards[np.newaxis, :],
                                      num_legal_actions, axis=0)

    last_action = _cards2array(infoset.last_move)
    last_action_batch = np.repeat(last_action[np.newaxis, :],
                                  num_legal_actions, axis=0)

    my_action_batch = np.zeros(my_handcards_batch.shape)
    for j, action in enumerate(infoset.legal_actions):
        my_action_batch[j, :] = _cards2array(action)

    last_landlord_action = _cards2array(
        infoset.last_move_dict['landlord'])
    last_landlord_action_batch = np.repeat(
        last_landlord_action[np.newaxis, :],
        num_legal_actions, axis=0)
    landlord_num_cards_left = _get_one_hot_array(
        infoset.num_cards_left_dict['landlord'], 20)
    landlord_num_cards_left_batch = np.repeat(
        landlord_num_cards_left[np.newaxis, :],
        num_legal_actions, axis=0)

    landlord_played_cards = _cards2array(
        infoset.played_cards['landlord'])
    landlord_played_cards_batch = np.repeat(
        landlord_played_cards[np.newaxis, :],
        num_legal_actions, axis=0)

    last_teammate_action = _cards2array(
        infoset.last_move_dict['landlord_up'])
    last_teammate_action_batch = np.repeat(
        last_teammate_action[np.newaxis, :],
        num_legal_actions, axis=0)
    teammate_num_cards_left = _get_one_hot_array(
        infoset.num_cards_left_dict['landlord_up'], 17)
    teammate_num_cards_left_batch = np.repeat(
        teammate_num_cards_left[np.newaxis, :],
        num_legal_actions, axis=0)

    teammate_played_cards = _cards2array(
        infoset.played_cards['landlord_up'])
    teammate_played_cards_batch = np.repeat(
        teammate_played_cards[np.newaxis, :],
        num_legal_actions, axis=0)

    landlord_played_cards = _cards2array(
        infoset.played_cards['landlord'])
    landlord_played_cards_batch = np.repeat(
        landlord_played_cards[np.newaxis, :],
        num_legal_actions, axis=0)

    bomb_num = _get_one_hot_bomb(
        infoset.bomb_num)
    bomb_num_batch = np.repeat(
        bomb_num[np.newaxis, :],
        num_legal_actions, axis=0)

    x_batch = np.hstack((my_handcards_batch,
                         other_handcards_batch,
                         landlord_played_cards_batch,
                         teammate_played_cards_batch,
                         last_action_batch,
                         last_landlord_action_batch,
                         last_teammate_action_batch,
                         landlord_num_cards_left_batch,
                         teammate_num_cards_left_batch,
                         bomb_num_batch,
                         my_action_batch))
    x_no_action = np.hstack((my_handcards,
                             other_handcards,
                             landlord_played_cards,
                             teammate_played_cards,
                             last_action,
                             last_landlord_action,
                             last_teammate_action,
                             landlord_num_cards_left,
                             teammate_num_cards_left,
                             bomb_num))
    z = _action_seq_list2array(_process_action_seq(
        infoset.card_play_action_seq))
    z_batch = np.repeat(
        z[np.newaxis, :, :],
        num_legal_actions, axis=0)
    obs = {
            'position': 'landlord_down',
            'x_batch': x_batch.astype(np.float32),
            'z_batch': z_batch.astype(np.float32),
            'legal_actions': infoset.legal_actions,
            'x_no_action': x_no_action.astype(np.int8),
            'z': z.astype(np.int8),
          }
    return obs



def _get_obs_universal(infoset, libSplit=None, train=False, objective = 'wp'):
    # The number of bombs in the current state. 1*54.
    bomb_num = np.hstack((_get_one_hot_bomb(infoset.bomb_num), np.zeros((54-15))))
    #The union of the other two players' cards. 1*54.
    other_handcards = _cards2array(infoset.other_hand_cards)
    #The three face-down cards. 1*54.
    three_landlord_cards = _cards2array((infoset.three_landlord_cards))

    #Identity[Landlord(00), Peasant - Down(01), Peasant - Up(10)] repeats 9 times,
    # Camp[Landlord(1), Peasant(0)] repeats 16 times, Number of self cards[1*20] one-hot vector
    # 18+16+20---> 1*54
    landlord_num_cards_left = _get_one_hot_array(infoset.num_cards_left_dict['landlord'], 20)
    landlord_up_num_cards_left = _get_one_hot_array(infoset.num_cards_left_dict['landlord_up'], 17)
    landlord_down_num_cards_left = _get_one_hot_array(infoset.num_cards_left_dict['landlord_down'], 17)

    self_id_camp_left = []
    if infoset.player_position == 'landlord':
        self_id_camp_left = np.hstack((np.tile([0,0],9), np.ones(16), landlord_num_cards_left))
    elif infoset.player_position == 'landlord_down':
        self_id_camp_left = np.hstack((np.tile([0,1],9), np.zeros(19), landlord_down_num_cards_left))
    elif infoset.player_position == 'landlord_up':
        self_id_camp_left = np.hstack((np.tile([1,0],9), np.zeros(19), landlord_up_num_cards_left))
    else:
        raise  ValueError("error role id")

    #Cards in self hand , 1*54
    my_handcards = _cards2array(infoset.player_hand_cards)

    #The cards that the Landlord has played before. 1*54
    landlord_played_cards = _cards2array(infoset.played_cards['landlord'])
    #The cards that the previous Peasant has played. 1*54
    landlord_up_played_cards = _cards2array(infoset.played_cards['landlord_up'])
    #The cards that the next Peasant has played before. 1*54
    landlord_down_played_cards = _cards2array(infoset.played_cards['landlord_down'])

    #number of cards left for Landlord and Peasants. 1*54
    num_card_left = np.hstack((landlord_num_cards_left, landlord_down_num_cards_left, landlord_up_num_cards_left))

    #The identity, camp and the number of cards left of the player who  most recent. 1*54
    last_id_camp_left = []
    if len(infoset.last_move)<=0:
        last_id_camp_left = np.zeros(54, dtype=np.int8)
    else:
        if infoset.last_pid == 'landlord':
            last_id_camp_left = np.hstack((np.tile([0, 0], 9), np.ones(16), landlord_num_cards_left))
        elif infoset.last_pid == 'landlord_down':
            last_id_camp_left = np.hstack((np.tile([0, 1], 9), np.zeros(19), landlord_down_num_cards_left))
        elif infoset.last_pid == 'landlord_up':
            last_id_camp_left = np.hstack((np.tile([1, 0], 9), np.zeros(19), landlord_up_num_cards_left))
        else:
            raise  ValueError("error role id")
    #cards of the most recent action 1*54
    last_action = _cards2array(infoset.last_move)

    #Last 9 historical actions including the identity, camp, the number of cards left of the player who takes the action 1*54 and the cards 1*54
    #18*54
    seq = _action_seq_list2universalarray(_process_action_seq(infoset.player_act_seq, 9), _process_action_seq(infoset.card_play_action_seq, 9))

    x_no_action =np.vstack((bomb_num,
                  other_handcards,
                  three_landlord_cards,
                  self_id_camp_left,
                  my_handcards,
                  landlord_played_cards,
                  landlord_down_played_cards,
                  landlord_up_played_cards,
                  num_card_left,
                  last_id_camp_left,
                  last_action,
                  seq
                  ))

    is_split = False
    num_legal_actions = len(infoset.legal_actions)
    if train and num_legal_actions > 1:

        exp_split = 0.1

        if exp_split > 0 and np.random.rand() < exp_split:
            is_split = True
            hand_vector = np.zeros(15, dtype=np.int8)
            counter = Counter(infoset.player_hand_cards)
            len_hand = len(infoset.player_hand_cards)
            for card, num_times in counter.items():
                hand_vector[Card2Column[card]] = num_times
            left_split_length = np.zeros(num_legal_actions, dtype=np.int8)

            for index, action in enumerate(infoset.legal_actions):
                child_vector = np.zeros(15, dtype=np.int8)
                counter = Counter(action)
                for card, num_times in counter.items():
                    child_vector[Card2Column[card]] = num_times
                INPUT = c_int * 15
                input = INPUT()
                for i in range(15):
                    input[i] = hand_vector[i] - child_vector[i]

                # return type
                libSplit.getMinHands.restype = c_int
                left_split_length[index] = libSplit.getMinHands(len_hand-len(action), input)

            min_length = min(left_split_length)
            small_list_action = []
            big_list_action = []

            for index, v in enumerate(left_split_length):
                if v<=min_length+3:
                    small_list_action.append(infoset.legal_actions[index])
                else:
                    big_list_action.append(infoset.legal_actions[index])

            infoset.legal_actions = small_list_action
            num_legal_actions = len(infoset.legal_actions)
    #The three face-down cards batch*1*54
    my_action_batch = np.zeros((num_legal_actions, 54))
    for j, action in enumerate(infoset.legal_actions):
        my_action_batch[j, :] = _cards2array(action)
    _z_batch = np.repeat(x_no_action[np.newaxis, :, :], num_legal_actions, axis=0)
    my_action_batch = my_action_batch[:,np.newaxis,:]
    x_batch = np.zeros([len(_z_batch), 30, 54], int)
    for i in range(0,len(_z_batch)):
        x_batch[i] = np.vstack((_z_batch[i], my_action_batch[i]))

    obs = {
            'position': infoset.player_position,
            'x_batch': x_batch.astype(np.float32),
            'legal_actions': infoset.legal_actions,
            'x_no_action': x_no_action.astype(np.int8),
            'is_split':is_split
          }

    return obs

def _load_model(position, model_path, model_type, evaluate_device_cpu):
    from mdou.dmc.models import model_dict

    if str(position+'_'+model_type) not in model_dict.keys():
        raise ValueError("invalid model key")

    model = model_dict[position+'_'+model_type]()
    model_state_dict = model.state_dict()
    if torch.cuda.is_available() and not evaluate_device_cpu:
        pretrained = torch.load(model_path, map_location='cuda:0')
    else:
        pretrained = torch.load(model_path, map_location='cpu')
    pretrained = {k: v for k, v in pretrained.items() if k in model_state_dict}
    model_state_dict.update(pretrained)
    model.load_state_dict(model_state_dict)
    if torch.cuda.is_available() and not evaluate_device_cpu:
        model.cuda()
    model.eval()
    return model