import tensorflow as tf  # Deep Learning library
import numpy as np  # Handle matrices
import retro  # Retro Environment

from skimage import transform  # Help us to preprocess the frames
from skimage.color import rgb2gray  # Help us to gray our frames
from scipy.special import softmax

import matplotlib.pyplot as plt  # Display graphs

from collections import deque  # Ordered collection with ends

import random

from src.rl.space_invaders.space_inv_src.NNet import *
from src.rl.space_invaders.space_inv_src.utils import *

env = retro.make(game="SpaceInvaders-Atari2600")

print("The size of our frame is: ", env.observation_space)
print("The action size is : ", env.action_space.n)

# Here we create an hot encoded version of our actions
# possible_actions = [[1, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0, 0]...]
possible_actions = np.array(np.identity(env.action_space.n, dtype=int).tolist())
normalized_frame_size = (110, 84)


stack_size = 4

stacked_frames = deque(
    [np.zeros(normalized_frame_size, dtype=np.int) for i in range(stack_size)], maxlen=4
)

state_size = [
    110,
    84,
    4,
]  # Our input is a stack of 4 frames hence 110x84x4 (Width, height, channels)
action_size = env.action_space.n  # 8 possible actions
learning_rate = 0.00025  # Alpha (aka learning rate)

### TRAINING HYPERPARAMETERS
total_episodes = 50  # Total episodes for training
max_steps = 50000  # Max possible steps in an episode
batch_size = 64  # Batch size

# Exploration parameters for epsilon greedy strategy
explore_start = 1.0  # exploration probability at start
explore_stop = 0.01  # minimum exploration probability
decay_rate = 0.00001  # exponential decay rate for exploration prob

# Q learning hyperparameters
gamma = 0.9  # Discounting rate

### MEMORY HYPERPARAMETERS
pretrain_length = (
    batch_size
)  # Number of experiences stored in the Memory when initialized for the first time
memory_size = 1000000  # Number of experiences the Memory can keep

### PREPROCESSING HYPERPARAMETERS
stack_size = 4  # Number of frames stacked

### MODIFY THIS TO FALSE IF YOU JUST WANT TO SEE THE TRAINED AGENT
training = True

## TURN THIS TO TRUE IF YOU WANT TO RENDER THE ENVIRONMENT
episode_render = False

tf.reset_default_graph()

# Instantiate the DQNetwork
DQNetwork = DQNetwork(state_size, action_size, learning_rate)

memory = Memory(max_size=memory_size)

# first, populating memory with random actions and its consequences
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

writer = tf.summary.FileWriter("/tensorboard/dqn/1")

## Losses
tf.summary.scalar("Loss", DQNetwork.loss)

write_op = tf.summary.merge_all()


saver = tf.train.Saver()
# train
if training == True:
    with tf.Session() as sess:
        # Initialize the variables
        sess.run(tf.global_variables_initializer())

        # Initialize the decay rate (that will use to reduce epsilon)
        decay_step = 0

        for episode in range(total_episodes):
            step = 0
            episode_rewards = []
            state = env.reset()

            # preprocess start and stack frames
            state, stacked_frames = stack_frames(
                stacked_frames, state, True, stack_size, normalized_frame_size
            )

            while step < max_steps:
                step += 1
                decay_step += 1

                action, explore_probability = predict_action(
                    explore_start,
                    explore_stop,
                    decay_rate,
                    decay_step,
                    state,
                    possible_actions,
                    sess,
                    DQNetwork
                )

                next_state, reward, done, _ = env.step(action)

                if episode_render:
                    env.render()

                episode_rewards.append(reward)

                if done:
                    next_state = np.zeros((110, 84), dtype=np.int)
                    next_state, stacked_frames = stack_frames(
                        stacked_frames,
                        next_state,
                        False,
                        stack_size,
                        normalized_frame_size,
                    )

                    step = max_steps

                    total_reward = np.sum(episode_rewards)

                    # TODO ???????????????????????????
                    print(
                        "Episode: {}".format(episode),
                        "Total reward: {}".format(total_reward),
                        "Explore P: {:.4f}".format(explore_probability),
                        "Training Loss {:.4f}".format(loss),
                    )

                    # add empty frame to memotry
                    memory.add((state, action, reward, next_state, done))

                else:
                    next_state, stacked_frames = stack_frames(
                        stacked_frames,
                        next_state,
                        False,
                        stack_size,
                        normalized_frame_size,
                    )
                    memory.add((state, action, reward, next_state, done))
                    state = next_state

                batch = memory.sample(batch_size)
                states_mb = np.array([each[0] for each in batch], ndmin=3)
                actions_mb = np.array([each[1] for each in batch])
                rewards_mb = np.array([each[2] for each in batch])
                next_states_mb = np.array([each[3] for each in batch], ndmin=3)
                dones_mb = np.array([each[4] for each in batch])

                # here we are calculating Qs of the next step. it is either a Reward or R+gamma+Q of best actions
                target_Qs_batch = []
                Qs_next_state = sess.run(
                    DQNetwork.output, feed_dict={DQNetwork.inputs_: next_states_mb}
                )
                for i in range(0, len(batch)):
                    terminal = dones_mb[i]

                    if terminal:
                        target_Qs_batch.append(rewards_mb[i])

                    else:
                        target = rewards_mb[i] + gamma * np.max(Qs_next_state[i])
                        target_Qs_batch.append(target)

                # TODO ???
                targets_mb = np.array([each for each in target_Qs_batch])

                # having updated Q values,
                loss, _ = sess.run(
                    [DQNetwork.loss, DQNetwork.optimizer],
                    feed_dict={
                        DQNetwork.inputs_: states_mb,
                        DQNetwork.target_Q: targets_mb,
                        DQNetwork.actions_: actions_mb,
                    },
                )

                summary = sess.run(
                    write_op,
                    feed_dict={
                        DQNetwork.inputs_: states_mb,
                        DQNetwork.target_Q: targets_mb,
                        DQNetwork.actions_: actions_mb,
                    },
                )
                writer.add_summary(summary, episode)
                writer.flush()

            if episode % 5 == 0:
                save_path = saver.save(sess, "/models/model.ckpt")
                print("Model Saved")

with tf.Session() as sess:
    total_test_rewards = []

    # Load the model
    saver.restore(sess, "/models/model.ckpt")

    for episode in range(1):
        total_rewards = 0

        state = env.reset()
        state, stacked_frames = stack_frames(stacked_frames, state, True, stack_size,normalized_frame_size)

        print("****************************************************")
        print("EPISODE ", episode)

        while True:
            # Reshape the state
            state = state.reshape((1, *state_size))
            # Get action from Q-network
            # Estimate the Qs values state
            Qs = sess.run(DQNetwork.output, feed_dict={DQNetwork.inputs_: state})

            # Take the biggest Q value (= the best action)
            choice = np.argmax(Qs)
            action = possible_actions[choice]

            # Perform the action and get the next_state, reward, and done information
            next_state, reward, done, _ = env.step(action)
            env.render()

            total_rewards += reward

            if done:
                print("Score", total_rewards)
                total_test_rewards.append(total_rewards)
                break

            next_state, stacked_frames = stack_frames(
                stacked_frames, next_state, False, stack_size,normalized_frame_size
            )
            state = next_state

    env.close()
