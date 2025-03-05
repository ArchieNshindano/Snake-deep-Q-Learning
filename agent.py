from collections import deque

import pygame
from PIL import  Image

from game import SnakeGameAI, Direction, Block, BLOCK_SIZE
from model import make_model
import numpy as np
import tensorflow as tf
from tensorflow import keras
import random

import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''

# physical_devices = tf.config.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(physical_devices[len(physical_devices)-1], True)

MAX_MEMORY = 100_000
BATCH_SIZE = 32
LR = 0.001


class Agent:

    def __init__(self):
        self.n_games = 0
        self.epsilon = 1
        self.epsilon_min = 0.0001
        self.epsilon_decay = 0.998

        self.gamma = 0.95
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = make_model(
            input_shape=[11], hidden_size=128, output_size=3)
        self.loss_fn = keras.losses.MeanSquaredError()
        self.optimizer = keras.optimizers.Adam(learning_rate=LR)


    def _epsilon_greedy_policy(self, state, image):
        if self.epsilon > self.epsilon_min:
            self.epsilon = self.epsilon * self.epsilon_decay
        else:
            self.epsilon = self.epsilon_min

        new_action = [0, 0, 0]

        if np.random.rand() < self.epsilon:
            action_choice = np.random.randint(0, 2)
            new_action[action_choice] = 1
        else:
            Q_values = self.model.predict([state[np.newaxis], image[np.newaxis]])
            action_choice = np.argmax(Q_values[0])
            new_action[action_choice] = 1

        return new_action

    def sample_experiences(self, batch_size):
        indices = np.random.randint(len(self.memory), size=batch_size)
        batch = [self.memory[index] for index in indices]
        states, actions, rewards, next_states, dones, images = [
            np.array([experience[field_index] for experience in batch])
            for field_index in range(6)
        ]
        # Reshape images to (batch_size, 50, 50, 3)
        images = np.stack(images, axis=0)
        return states, actions, rewards, next_states, dones, images

    def play_one_step(self, env, state, pygame, width=100, height=100, isInALoop = False):

        window = pygame.display.set_mode((width, height))
        image_bytes = pygame.image.tobytes(window, "RGB")
        resized_image = Image.frombytes('RGB', (width, height), image_bytes).resize((28, 28)).convert('L')
        image = np.array(resized_image).reshape(28,28,1).astype(float)

        action = self._epsilon_greedy_policy(state, image)

        next_state, reward, done, info = env.play_step(action)


        new_image_bytes = pygame.image.tobytes(window, "RGB")
        resized_image = Image.frombytes('RGB', (width, height), new_image_bytes).resize((28, 28)).convert('L')
        new_image = np.array(resized_image).reshape(28,28,1).astype(float)


        if not isInALoop:
         self.memory.append((state, action, reward, next_state, done,new_image))

        return next_state, action, reward, done, info

    def training_step(self, batch_size):
        experiences = self.sample_experiences(batch_size)
        states, actions, rewards, next_states, dones, images = experiences

        # Compute target Q-values
        next_Q_values = self.model.predict([next_states, images])
        max_next_Q_values = np.max(next_Q_values, axis=1)
        target_Q_values = rewards + self.gamma * max_next_Q_values * (1 - dones)

        # Train the model
        with tf.GradientTape() as tape:
            all_Q_values = self.model([states, images])
            Q_values = tf.reduce_sum(all_Q_values * actions, axis=1, keepdims=True)
            loss = tf.reduce_mean(self.loss_fn(target_Q_values, Q_values))
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))


if __name__ == "__main__":
    agent = Agent()


    env = SnakeGameAI()

    isInALoop = False

    env.reset()
    record = 0
    width = 400
    height = 400

    state = env._get_state()


    while True:

        isInALoop = env.detect_loop()

        next_state, action, reward, done, info = agent.play_one_step(
            env, state, env.pygame, width, height, isInALoop)

        state = next_state

        if done:
            agent.n_games += 1
            env.reset()

        print(f"Record: {record} vs Score: {env.score}")


        if env.score > record:
            record = env.score
            agent.model.save(r"C:\Users\admin\Desktop\ORGANISED\CODE\models\snake2.keras")

        if len(agent.memory) > BATCH_SIZE:
            agent.training_step(BATCH_SIZE)
