import numpy as np
import random
import gym
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Activation
from tensorflow.keras.initializers import HeUniform, HeNormal
from collections import deque
import time

class BaseAgent:
    def __init__(self, action_space):
        self.action_space = action_space
    
    def get_action(self, obs):
        raise NotImplementedError("This method should be overridden by subclasses")

    def learn(self, obs, action, reward, next_obs):
        raise NotImplementedError("This method should be overridden by subclasses")

    def _get_state(self, obs):
        state = np.array(obs)
        reshaped_state = state.reshape((1, -1)) 
        return reshaped_state


class SimpleAgent(BaseAgent):
    def __init__(self, action=2):
        self.action = action
    
    def get_action(self, obs):
        """
        Always returns the same action, regardless of the observation.
        """
        return self.action

    def learn(self, obs, action, reward, next_obs):
        """
        This agent doesn't learn, so this method does nothing.
        """
        pass

class QLearningAgent(BaseAgent):
    def __init__(self, action_space, learning_rate=0.1, discount_factor=0.99, exploration_rate=0.1, max_no_op=30):
        super().__init__(action_space)
        self.q_table = {}
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.max_no_op = max_no_op
        self.no_op_action_space = gym.spaces.Discrete(max_no_op + 1)  # 0 to max_no_op steps

    def get_action(self, obs):
        state = self._get_state(obs)  # Inherited from BaseAgent
        if random.random() < self.exploration_rate:
            action = self.action_space.sample()
            no_op_steps = self.no_op_action_space.sample()
        else:
            action_values = self.q_table.get(state.tobytes(), np.zeros(self.action_space.n + self.max_no_op + 1))
            action = np.argmax(action_values[:-self.max_no_op])
            no_op_steps = np.argmax(action_values[-self.max_no_op:])
        
        return action, no_op_steps

    def learn(self, obs, action, reward, next_obs, no_op_steps):
        state = self._get_state(obs).tobytes()
        next_state = self._get_state(next_obs).tobytes()
        
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.action_space.n + self.max_no_op + 1)
        
        max_next_q = np.max(self.q_table.get(next_state, np.zeros(self.action_space.n + self.max_no_op + 1)))
        # Update Q-value for the chosen action
        self.q_table[state][action] += self.learning_rate * (reward + self.discount_factor * max_next_q - self.q_table[state][action])
        # Update Q-value for the chosen no-op step count
        self.q_table[state][-self.max_no_op + no_op_steps] += self.learning_rate * (reward + self.discount_factor * max_next_q - self.q_table[state][-self.max_no_op + no_op_steps])

    def get_q_values(self, obs):
        state = self._get_state(obs).tobytes()
        return self.q_table.get(state, np.zeros(self.action_space.n + self.max_no_op + 1))

class DQNAgentNoop(BaseAgent):
    def __init__(self, action_space, max_no_op=30):
        super().__init__(action_space)  # Pass action_space to BaseAgent
        self.action_size = action_space.n
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0   # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.0001
        self.max_no_op = max_no_op
        self.no_op_action_space = gym.spaces.Discrete(max_no_op + 1)
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

        # Move model to GPU if available
        if tf.config.list_physical_devices('GPU'):
            with tf.device('/GPU:0'):
                self.model = self._build_model()
                self.target_model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Dense(128, input_shape=(128,), activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(self.action_size + self.max_no_op + 1, activation='linear'))
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate))
        return model

    def get_action(self, obs):
        state = self._get_state(obs)
        #print(f"[DEBUG] State passed to model for prediction: {state.shape}")
        if np.random.rand() <= self.epsilon:
            action = np.random.choice(self.action_size)
            no_op_steps = self.no_op_action_space.sample()
            #print(f"[DEBUG] Random action chosen: {action}, No-op steps: {no_op_steps}")
            return action, no_op_steps
        act_values = self.model.predict(state)
        #print(f"[DEBUG] Predicted action values: {act_values.shape}")
        action = np.argmax(act_values[0, :self.action_size])
        no_op_steps = np.argmax(act_values[0, self.action_size:self.action_size + self.max_no_op + 1])
        return action, no_op_steps

    def learn(self, obs, action, reward, next_obs, done, no_op_steps):
        state = self._get_state(obs)
        next_state = self._get_state(next_obs)
        state = np.reshape(state, (1, -1))
        next_state = np.reshape(next_state, (1, -1))
        #print(f"[DEBUG] Learn - State shape: {state.shape}, Next state shape: {next_state.shape}")
        target = reward
        if not done:
            target = reward + self.gamma * np.amax(self.target_model.predict(next_state)[0])
        target_f = self.model.predict(state)
        #print(f"[DEBUG] Learn - Target before fitting: {target_f.shape}")
        target_f[0][action] = target
        target_f[0][-self.max_no_op + no_op_steps] = target
        self.model.fit(state, target_f, epochs=1, verbose=0)

    def remember(self, state, action, reward, next_state, done, no_op_steps):
        self.memory.append((state, action, reward, next_state, done, no_op_steps))


    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return

        minibatch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones, no_op_steps = zip(*minibatch)

        # Convert lists to numpy arrays
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_states = np.array(next_states)
        dones = np.array(dones)
        no_op_steps = np.array(no_op_steps)  # Add this line if needed

        # Predict Q-values for current and next states
        target_f = self.model.predict(states)
        next_state_values = np.amax(self.target_model.predict(next_states), axis=1)

        # Compute the targets for each action
        targets = rewards + self.gamma * next_state_values * (1 - dones)
        target_f[np.arange(batch_size), actions] = targets

        # Train the model
        self.model.fit(states, target_f, epochs=1, verbose=0)

    def update_target_model(self):
        #print("[DEBUG] Updating target model weights.")
        self.target_model.set_weights(self.model.get_weights())
    
    def get_q_values(self, obs):
        state = self._get_state(obs)
        state = np.reshape(state, (1, -1))
        q_values = self.model.predict(state)
        return q_values[0]
    

class DQNAgent(BaseAgent):
    def __init__(self, action_space):
        super().__init__(action_space)  # Pass action_space to BaseAgent
        self.action_size = action_space.n
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0   # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.0001
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

        # Move model to GPU if available
        if tf.config.list_physical_devices('GPU'):
            with tf.device('/GPU:0'):
                self.model = self._build_model()
                self.target_model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Dense(128, input_shape=(128,), activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate))
        return model

    def get_action(self, obs):
        state = self._get_state(obs)
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def learn(self, obs, action, reward, next_obs, done):
        state = self._get_state(obs)
        next_state = self._get_state(next_obs)
        state = np.reshape(state, (1, -1))
        next_state = np.reshape(next_state, (1, -1))
        target = reward
        if not done:
            target = reward + self.gamma * np.amax(self.target_model.predict(next_state)[0])
        target_f = self.model.predict(state)
        target_f[0][action] = target
        self.model.fit(state, target_f, epochs=1, verbose=0)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return

        minibatch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)

        # Convert lists to numpy arrays
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_states = np.array(next_states)
        dones = np.array(dones)

        # Predict Q-values for current and next states
        target_f = self.model.predict(states)
        next_state_values = np.amax(self.target_model.predict(next_states), axis=1)

        # Compute the targets for each action
        targets = rewards + self.gamma * next_state_values * (1 - dones)
        target_f[np.arange(batch_size), actions] = targets

        # Train the model
        self.model.fit(states, target_f, epochs=1, verbose=0)

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def get_q_values(self, obs):
        state = self._get_state(obs)
        state = np.reshape(state, (1, -1))
        q_values = self.model.predict(state)
        return q_values[0]

class NoopImgDQN(DQNAgentNoop):
    def __init__(self, action_space, max_no_op=30):
        super().__init__(action_space, max_no_op)
        self.model = self._build_model()
        self._reinitialize_weights(self.model) 
        self.target_model = self._build_model()
        self._reinitialize_weights(self.target_model)  
        self.target_model.set_weights(self.model.get_weights())  # Sync weights
        self.update_target_model()

        # Move model to GPU if available
        if tf.config.list_physical_devices('GPU'):
            with tf.device('/GPU:0'):
                self.model = self._build_model()
                self._reinitialize_weights(self.model)
                self.target_model = self._build_model()
                self._reinitialize_weights(self.target_model)
                self.target_model.set_weights(self.model.get_weights())  # Sync weights

        # Debugging: Log initial predictions
        print(f"[DEBUG] Initial model predictions: {self.model.predict(np.zeros((1, 84, 84, 4)))}")
        print(f"[DEBUG] Initial target model predictions: {self.target_model.predict(np.zeros((1, 84, 84, 4)))}")

    def _build_model(self):
        model = Sequential([
            Conv2D(8, (3, 3), strides=1, input_shape=(84, 84, 4), activation='relu'),
            Conv2D(16, (3, 3), strides=2, activation='relu'),
            Flatten(),
            Dense(128, activation='relu'),
            Dense(self.action_size + self.max_no_op + 1, activation='linear')
        ])
        optimizer = Adam(learning_rate=self.learning_rate, clipvalue=1.0)  
        model.compile(loss='mse', optimizer=optimizer)
        return model

    def _reinitialize_weights(self, model):
        initializer = HeNormal()
        for layer in model.layers:
            if isinstance(layer, (Dense, Conv2D)):
                weights_shape = tf.shape(layer.kernel)
                print(f"[DEBUG] Reinitializing weights for {layer.name} with shape {weights_shape}")
                layer.kernel.assign(initializer(weights_shape))
                if layer.bias is not None:
                    layer.bias.assign(tf.zeros_like(layer.bias))
                # Check if any NaNs were introduced
                if tf.reduce_any(tf.math.is_nan(layer.kernel)):
                    print(f"[ERROR] NaN detected in {layer.name} kernel after reinitialization!")
                if tf.reduce_any(tf.math.is_nan(layer.bias)):
                    print(f"[ERROR] NaN detected in {layer.name} bias after reinitialization!")


    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return

        minibatch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones, no_op_steps = zip(*minibatch)

        # Convert lists to numpy arrays
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_states = np.array(next_states)
        dones = np.array(dones)

        # Transpose states and next_states from (batch_size, 4, 84, 84) to (batch_size, 84, 84, 4)
        states = np.transpose(states, (0, 2, 3, 1))
        next_states = np.transpose(next_states, (0, 2, 3, 1))

        # Predict Q-values for current and next states
        target_f = self.model.predict(states)
        next_state_values = np.amax(self.target_model.predict(next_states), axis=1)

        # Check for NaNs in next_state_values and clip values
        next_state_values = np.where(np.isnan(next_state_values), 0, next_state_values)
        next_state_values = np.clip(next_state_values, -1e6, 1e6)

        # Compute the targets for each action
        targets = rewards + self.gamma * next_state_values * (1 - dones)
        
        # Check for NaNs in targets and clip values
        targets = np.where(np.isnan(targets), 0, targets)
        targets = np.clip(targets, -1e6, 1e6)

        # Update the target values
        target_f[np.arange(batch_size), actions] = targets

        # Debugging: Inspect a small portion of the data
        print(f"[DEBUG] Sample state: {states[0]}")
        print(f"[DEBUG] Sample target: {target_f[0]}")
        print(f"[DEBUG] Rewards: {rewards}")
        print(f"[DEBUG] Next state values: {next_state_values}")
        print(f"[DEBUG] Computed targets: {targets}")

        # Test running a single batch manually
        print("[DEBUG] Running a single batch manually")
        sample_batch = states[:batch_size]
        sample_target = target_f[:batch_size]
        self.model.fit(sample_batch, sample_target, epochs=1, verbose=1)
        print("[DEBUG] Single batch completed")

        # Create a TensorFlow dataset
        dataset = tf.data.Dataset.from_tensor_slices((states, target_f))
        
        # Shuffle, batch, and prefetch the data
        dataset = dataset.shuffle(buffer_size=1024).batch(batch_size).prefetch(buffer_size=1)

        print("[DEBUG] Starting model.fit() with tf.data.Dataset")

        # Train the model using the dataset
        self.model.fit(dataset, epochs=1, verbose=0)
        
        print("[DEBUG] Completed model.fit()")