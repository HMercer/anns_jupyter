import numpy as np
from skimage import transform # Help us to preprocess the frames
from skimage.color import rgb2gray # Help us to gray our frames
from scipy.special import softmax
from collections import deque# Ordered collection with ends
import matplotlib.pyplot as plt # Display graphs
import random
from space_inv_src.NNet import DQNetwork


def preprocess_frame(frame, normalized_frame_size):
    # Greyscale frame
    gray = rgb2gray(frame)

    # Crop the screen (remove the part below the player)
    # [Up: Down, Left: right]
    cropped_frame = gray[8:-12, 4:-12]

    # Normalize Pixel Values
    normalized_frame = cropped_frame / 255.0

    # Resize
    # Thanks to Mikołaj Walkowiak
    preprocessed_frame = transform.resize(normalized_frame,normalized_frame_size) #[110, 84])

    return preprocessed_frame  # 110x84x1 frame

def stack_frames(stacked_frames, state, is_new_episode, stack_size,normalized_frame_size):
    frame = preprocess_frame(state,normalized_frame_size)

    if is_new_episode:
        stacked_frames = deque([np.zeros(normalized_frame_size, dtype=np.int) for i in range(stack_size)], maxlen=4)

        for i in range(stack_size):
            stacked_frames.append(frame)

        stacked_state = np.stack(stacked_frames,axis=2)

    else:
        stacked_frames.append(frame)
        stacked_state = np.stack(stacked_frames, axis=2)

    return stacked_state, stacked_frames


class Memory():
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        buffer_size = len(self.buffer)
        index = np.random.choice(np.arange(buffer_size),
                                 size= batch_size,
                                 replace=False)

        return [self.buffer[i] for i in index]


def predict_action(explore_start, explore_stop, decay_rate, decay_step, state, possible_actions, sess, DQNetwork:DQNetwork):
    exp_exp_tradeoff = np.random.rand()

    explore_probability = explore_stop + (explore_start - explore_stop) * np.exp(-decay_rate * decay_step)
    if (explore_probability > exp_exp_tradeoff):
        # Make a random action (exploration)
        choice = random.randint(1, len(possible_actions)) - 1
        action = possible_actions[choice]

    else:
        # this is jus output.eval
        Qs = sess.run(DQNetwork.output, feed_dict = {DQNetwork.inputs_:state.reshape((1,*state.shape))})

        choice = np.argmax(Qs)
        action = possible_actions[choice]

    return action, explore_probability

def init_memory(memory_size, pretrain_length, env,stacked_frames, stack_size, normalized_frame_size, possible_actions):
    memory = Memory(max_size=memory_size)
    for i in range(pretrain_length):
        # initiate state and stacked frames
        if i == 0:
            state = env.reset()

            state, stacked_frames = stack_frames(
                stacked_frames, state, True, stack_size, normalized_frame_size
            )

        # get random
        choice = random.randint(1, len(possible_actions)) - 1
        action = possible_actions[choice]

        next_state, reward, done, _ = env.step(action)

        next_state, stacked_frames = stack_frames(
            stacked_frames, next_state, False, stack_size, normalized_frame_size
        )

        if done:
            next_state = np.zeros(state.shape)
            memory.add((state, action, reward, next_state, done))
            state = env.reset()

            # Stack the frames
            state, stacked_frames = stack_frames(
                stacked_frames, state, True, stack_size, normalized_frame_size
            )
        else:
            memory.add((state, action, reward, next_state, done))

            # Our new state is now the next_state
            state = next_state
    return memory