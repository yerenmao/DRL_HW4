STATE_DIM           = 5
ACTION_DIM          = 1
GAMMA               = 0.99     # Discount factor for future rewardsD
TAU                 = 0.005    # Soft update rate for target networks
BATCH_SIZE          = 64       # Mini-batch size for sampling from replay buffer
REPLAY_BUFFER_SIZE  = 50000    # Capacity of the replay buffer

ACTOR_LR            = 3e-4     # Learning rate for the actor (policy) network
CRITIC_LR           = 3e-4     # Learning rate for the critic (Q-value) networks
ALPHA_LR            = 3e-4     # Learning rate for entropy coefficient α

TARGET_ENTROPY      = -1.0

TOTAL_EPISODES      = 500
EPISODE_MAX_STEP    = 1000
HIDDEN_LAYER_SIZE   = 64
LOG_STD_MIN         = -20
LOG_STD_MAX         = 2

SEED                = 42