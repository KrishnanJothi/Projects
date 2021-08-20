from rlbench.environment import Environment
from rlbench.action_modes import ArmActionMode, GripperActionMode, ActionMode
from rlbench.observation_config import ObservationConfig
from rlbench.tasks import ReachTarget
import numpy as np
import tensorflow as tf
from scipy.stats import multivariate_normal
import yaml


def act(obs, act_network, action_size, scale_vector):
    # Forward propagation for the given low-dim state input
    obs = obs / scale_vector
    mu, sigma = act_network(obs)

    # Checking if Neural Network outputs Nan
    if any(tf.reshape(tf.math.is_nan(mu), shape=(action_size,))):
        print('Neural network mu output is nan during generation')
    if any(tf.reshape(tf.math.is_nan(sigma), shape=(action_size,))):
        print('Neural network sigma output is nan during generation')

    # Building covariance matrix
    cov_mat = build_diag_cov(sigma, action_size)

    # To avoid Nan values in sampled action
    while True:
        # Reshaping the obtained mean array and then sampling
        Muh = np.reshape(mu.numpy(), action_size)
        arm = np.random.multivariate_normal(Muh, cov_mat)

        if not any(tf.math.is_nan(arm)):
            break
        print('Sampled action is itself Nan..... So resampling')

    # Calculating the PDF value for the sampled action
    sci_pdf = multivariate_normal.pdf(arm, mean=Muh, cov=cov_mat, allow_singular=True)

    return np.concatenate([np.array(arm)], axis=-1), sci_pdf


def build_diag_cov(diagonal, dim):
    cov = tf.zeros([dim, dim], tf.float64)
    diagonal = tf.reshape(diagonal, shape=(dim,))  # reshaping
    cov = tf.compat.v1.matrix_set_diag(cov, diagonal)  # set diagonal
    return cov


def scaling_vector(loc):
    panda_config = loc
    with open(panda_config, 'r') as stream:
        doc = yaml.load(stream)

    # set dimensions
    dim_observations = doc["Robot"]["Dimensions"]["observations"]

    # --- set observation and action limits for scaling
    robot_limits = doc["Robot"]["Limits"]
    # max actions
    max_actions = robot_limits["actions"]
    # gripper open
    gripper_open_limits = np.array(robot_limits["gripper_open"])
    # joint velocity limits
    joint_vel_limits = np.array(robot_limits["joint_vel"])
    # joint position -> max in pos/neg direction defines limit (for symmetric scaling)
    lower_joint_pos_limits_deg = robot_limits["lower_joint_pos"]
    range_joint_pos_deg = robot_limits["range_joint_pos"]
    lower_joint_pos_limits_rad = (np.array(lower_joint_pos_limits_deg) / 360.0) * 2 * np.pi
    range_joint_pos_rad = (np.array(range_joint_pos_deg) / 360.0) * 2 * np.pi
    upper_joint_pos_limits_rad = lower_joint_pos_limits_rad + range_joint_pos_rad
    joint_pos_limits_rad = np.amax((np.abs(lower_joint_pos_limits_rad), upper_joint_pos_limits_rad), axis=0)
    # joint forces
    joint_force_limits = np.array(robot_limits["joint_force"])
    # gripper pose -> first 3 entry are x,y,z and rest are quaternions
    gripper_pose_limits = np.array(robot_limits["gripper_pose"])
    # gripper joint position
    gripper_joint_pos_limits = np.array(robot_limits["gripper_pos"])
    # gripper touch forces -> 2* x,y,z
    gripper_force_limits = np.array(robot_limits["gripper_force"])
    # concatenate to scaling vector
    if doc["Agent"]["Setup"]["scale_robot_observations"]:
        # only the robot state is scaled
        obs_scaling_vector = np.concatenate((gripper_open_limits,
                                             joint_vel_limits,
                                             joint_pos_limits_rad,
                                             joint_force_limits,
                                             gripper_pose_limits,
                                             gripper_joint_pos_limits,
                                             gripper_force_limits))
        # append with ones for low-dim task state
        obs_scaling_vector = np.append(
            obs_scaling_vector, np.ones(dim_observations - len(obs_scaling_vector)))

    else:
        obs_scaling_vector = None

    return obs_scaling_vector


#####################################################################################################################
'''                                              MAIN CODE                                                        '''
#####################################################################################################################

obs_config = ObservationConfig()
obs_config.set_all_low_dim(True)
obs_config.set_all_high_dim(False)


# selecting the action mode
action_mode = ActionMode(ArmActionMode.ABS_JOINT_VELOCITY, GripperActionMode.OPEN_AMOUNT)

# creating an environment object
env = Environment(
    action_mode, obs_config=obs_config, headless=False)
env.launch()

# Defining the degrees of freedom depending on the task involved, Eg: For reach task (Dof = action_size-1 = 7)
Dof = env.action_size - 1

# task selection
task = env.get_task(ReachTarget)

# Loading pretrained model
actor_network = tf.keras.models.load_model('pretrained .h5 model file')

# scaling vector
scale_vec = scaling_vector("/home/irp/PycharmProjects/RLPanda/default_config.yaml")

no_of_episodes = 15
eps_length = 40

for episode in range(no_of_episodes):

    descriptions, obs = task.reset()

    for i in range(eps_length):
        # Extracting the low-dim inputs
        obs = tf.convert_to_tensor(obs.get_low_dim_data())
        obs = tf.expand_dims(obs, 0)

        # sampling an action according to the current policy and also the corresponding PDF value
        action, pdf_val = act(obs, actor_network, Dof, scale_vec)

        # fixed joints
        fixed = []
        fixed1 = []

        # Gripper action is chosen based on the task (fixed or stochastic)
        gripper = [1.0]  # Always open
        obs, reward, terminate = task.step(np.concatenate([fixed, action, fixed1, gripper], axis=-1))


env.shutdown()