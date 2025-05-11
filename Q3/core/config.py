STATE_DIM           = 67
ACTION_DIM          = 21
GAMMA               = 0.99     # Discount factor for future rewardsD
TAU                 = 0.005    # Soft update rate for target networks
BATCH_SIZE          = 256      # Mini-batch size for sampling from replay buffer
REPLAY_BUFFER_SIZE  = 300000   # Capacity of the replay buffer

ACTOR_LR            = 3e-4     # Learning rate for the actor (policy) network
CRITIC_LR           = 3e-4     # Learning rate for the critic (Q-value) networks
ALPHA_LR            = 3e-4     # Learning rate for entropy coefficient α

TARGET_ENTROPY      = -21.0

TOTAL_EPISODES      = 10000
EPISODE_MAX_STEP    = 1000
HIDDEN_LAYER_SIZE   = 64
LOG_STD_MIN         = -20
LOG_STD_MAX         = 2

SEED                = 42