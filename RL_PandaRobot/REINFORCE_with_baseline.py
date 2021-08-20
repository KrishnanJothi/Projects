from rlbench.environment import Environment
from rlbench.action_modes import ArmActionMode, GripperActionMode, ActionMode
from rlbench.observation_config import ObservationConfig
from rlbench.tasks import ReachTarget
import numpy as np
import tensorflow as tf
from scipy.stats import multivariate_normal
from tensorflow import keras
from tensorflow.keras import layers, initializers
import datetime
import yaml
import gc


def act(scaled_obs, act_network, action_size):
    mu, sigma = act_network(scaled_obs)

    # Checking if Neural Network outputs Nan
    if any(tf.reshape(tf.math.is_nan(mu), shape=(action_size,))):
        print('Neural network mu output is nan during generation')
    if any(tf.reshape(tf.math.is_nan(sigma), shape=(action_size,))):
        print('Neural network sigma output is nan during generation')

        # Building covariance matrix
    cov_mat = tf.compat.v1.matrix_set_diag(tf.zeros([action_size, action_size], tf.float64),
                                           tf.reshape(sigma, shape=(action_size,)))

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


def ConstructActorNetwork(obs_size, action_size, bias_mu_val, bias_sigma_val):

    inputs = layers.Input(shape=(obs_size,))  # input dimension
    hidden1 = layers.Dense(100, activation="relu", kernel_initializer=initializers.he_normal(),)(inputs)
    hidden2 = layers.Dense(100, activation="relu", kernel_initializer=initializers.he_normal(),)(hidden1)
    hidden3 = layers.Dense(100, activation="relu", kernel_initializer=initializers.he_normal(),)(hidden2)
    mu = layers.Dense(action_size, activation="tanh", kernel_initializer=initializers.Zeros(),
                      bias_initializer=initializers.Constant(bias_mu_val), dtype=tf.float64)(hidden3)
    sigma = layers.Dense(action_size, activation="softplus", kernel_initializer=initializers.Zeros(),
                         bias_initializer=initializers.Constant(bias_sigma_val), dtype=tf.float64)(hidden3)
    actor_Network = keras.Model(inputs=inputs, outputs=[mu, sigma])

    return actor_Network


def ConstructBaselineNetwork(obs_size):
    model = keras.Sequential()
    model.add(layers.Dense(20, activation="relu", kernel_initializer=initializers.he_normal(), input_shape=(obs_size,)))
    model.add(layers.Dense(20, activation="relu", kernel_initializer=initializers.he_normal()))
    model.add(layers.Dense(20, activation="relu", kernel_initializer=initializers.he_normal()))
    model.add(layers.Dense(1, activation="softplus", kernel_initializer=initializers.Zeros(),
                           bias_initializer=initializers.Constant(0.0), dtype=tf.float64))
    return model


def abs_fun(x):
    y = tf.keras.activations.softplus(x)
    return y


def generate_eps(eps_length, action_size, task, baseline_Net, actor_network, opt_baseline, Loss_obj, scale_vector):
    # clearing the data for a new episode
    observations = []
    actions = []
    rewards = []
    Neg_Log = []
    success = False

    descriptions, obs = task.reset()

    for i in range(eps_length):
        # Extracting the low-dim inputs
        obs = tf.convert_to_tensor(obs.get_low_dim_data())
        obs = tf.expand_dims(obs, 0)

        # appending
        observations.append(obs)
 
        # sampling an action according to the current policy and also the corresponding PDF value
        action, pdf_val = act(obs/scale_vector, actor_network, action_size)

        # Log of the PDF value
        log_val = tf.math.log(pdf_val + 1e-37)

        # Storing negative of the log (for the 'Loss' plot)
        Neg_Log.append(-log_val)

        # fixed joints (if lists are empty, all joints are free)
        fixed = []
        fixed1 = []

        # Gripper action is chosen based on the task (fixed or stochastic)
        gripper = [1.0]  # Always open
        obs, reward, terminate = task.step(np.concatenate([fixed, action, fixed1, gripper], axis=-1))

        # Checking if the sampled action leads to Nan
        if any(tf.reshape(tf.math.is_nan(obs.get_low_dim_data()), shape=(40,))) \
            or any(tf.reshape(tf.math.is_nan(reward), shape=(1,))):
            print('Sampled action led to Nan observation/reward.....')

        # appending
        rewards.append(reward)
        actions.append(action)

        if not success:
            success, term = task._task.success()

    # Calculating the return from the array of rewards
    returns = discount_rewards(reward=rewards)

    returns_base = baselineFunction(returns, observations, baseline_Net, opt_baseline, Loss_obj, scale_vector)

    if success:
        Status = "Done"
    else:
        Status = "Not Done"

    return observations, actions, returns, returns_base, Status, Neg_Log


def discount_rewards(reward):
    # Compute the gamma-discounted rewards over an episode
    gamma = 0.99  # discount rate
    running_add = 0
    discounted_r = np.zeros_like(reward)
    for i in reversed(range(0, len(reward))):
        running_add = running_add * gamma + reward[i]
        discounted_r[i] = running_add

    return discounted_r


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


def baselineFunction(discounted_rewards, observations, model, opt_baseline, loss_object, scaling_vec):
    basefunc = []
    for i in range(0, len(discounted_rewards)):
        val = model(observations[i] / scaling_vec)
        basefunc.append(discounted_rewards[i] - val)

    NetworkUpdate(opt_baseline, model, observations, discounted_rewards, loss_object, scaling_vec)

    return basefunc


def NetworkUpdate(optimiser, model, observations, targets, loss_obj, scale_vec):
    for k in range(len(observations)):
        with tf.GradientTape() as tape:
            predict = model(observations[k] / scale_vec, training=True)
            loss = loss_obj(targets[k], predict)

        # Compute Gradients
        grads = tape.gradient(loss, model.trainable_variables)

        # Apply gradients to update network weights
        optimiser.apply_gradients(zip(grads, model.trainable_variables))


def CustomLossGaussian(low_state, action, Return, action_size, actor_network, scale_vector):
    nn_mu, nn_sigma = actor_network(low_state / scale_vector, training=True)

    # Checking if Neural Network outputs Nan
    if any(tf.reshape(tf.math.is_nan(nn_mu), shape=(action_size,))):
        print('Neural network mu output is nan during training')
    if any(tf.reshape(tf.math.is_nan(nn_sigma), shape=(action_size,))):
        print('Neural network sigma output is nan during training')

    # Build covariance matrix
    nn_cov_mat = tf.compat.v1.matrix_set_diag(tf.zeros([action_size, action_size], tf.float64),
                                              tf.reshape(nn_sigma, shape=(action_size,)))

    # Obtain pdf of Multivariate Gaussian distribution
    nn_mu = tf.reshape(nn_mu, shape=(action_size, 1))
    action = tf.reshape(action, shape=(action_size, 1))
    pdf_value = custom_multivariate_normal(action, action_size, nn_mu, nn_cov_mat)

    # return loss_actor
    return - Return * tf.math.log(pdf_value + 1e-37)


def custom_multivariate_normal(x, d, mean, covariance):
    """pdf of the multivariate normal distribution using Tensorflow."""
    x_m = x - mean
    return 1. / (tf.math.sqrt((2 * np.pi) ** d * tf.linalg.det(covariance))) * tf.math.exp(
        -tf.tensordot(tf.transpose(tf.linalg.solve(covariance, x_m)), x_m, 1) / 2)


#####################################################################################################################
'''                                              MAIN CODE                                                        '''
#####################################################################################################################

def main():
    obs_config = ObservationConfig(None)
    obs_config.set_all_low_dim(True)
    obs_config.set_all_high_dim(False)

    # selecting the action mode
    action_mode = ActionMode(ArmActionMode.ABS_JOINT_VELOCITY, GripperActionMode.OPEN_AMOUNT)

    # creating an environment object
    env = Environment(
        action_mode, obs_config=obs_config, headless=True, static_positions=True)
    env.launch()

    # Defining the degrees of freedom depending on the task involved, Eg: For reach task (Dof = action_size-1 = 7)
    Dof = env.action_size - 1

    # task selection
    task = env.get_task(ReachTarget)

    # parameters for neural network
    bias_mu = 0.0
    bias_sigma = 10

    # Panda Robot configuration .yaml file to compute the scaling vector
    scale_vec = scaling_vector("/home/irp/PycharmProjects/RLPanda/default_config.yaml")

    # Defining the neural network architectures
    # first term in the argument is chosen to be no of input low_dim_states
    actor_network = ConstructActorNetwork(40, Dof, bias_mu, bias_sigma)
    
    # Defining the baseline network 
    baseline_Network = ConstructBaselineNetwork(40)
    baseline_Network.compile(loss="mean_squared_error", optimizer="adam", metrics=["mean_squared_error"])

    # Loading pretrained model (continue training)
    # actor_network = tf.keras.models.load_model("directory",compile=False)
    # baseline_Network = tf.keras.models.load_model("directory", compile=False)

    episode_length = 40
    no_of_episodes = 40000

    # Defining the optimizers for actor and baseline networks
    opt = keras.optimizers.Adam(learning_rate=0.00001, clipnorm=1.0)
    opt_baseline = keras.optimizers.Adam(learning_rate=0.001)
    loss_baseline_obj = tf.keras.losses.MeanSquaredError()

    # TensorBoard logging
    Name = "RLPanda"
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = '/home/irp/PycharmProjects/RLPanda/logs/' + Name + current_time + '/train'

    # continue training and save logs in existing directory
    # train_log_dir = "existing directory with saved logs"

    train_summary_writer = tf.summary.create_file_writer(train_log_dir)

    # Logging gradients and weights to csv files
    gradfile = open("gradients" + Name + current_time + ".csv", "a")
    weightfile = open("weights" + Name + current_time + ".csv", "a")
    obsfile = open("Observations" + Name + current_time + ".csv", "a")
    actfile = open("Actions" + Name + current_time + ".csv", "a")

    # variables related to Tensorboard logging
    done_ = 0
    count = 0  # pay attention to 'count' while continuing training
    value = 0

    status_interval = 20  # For success frequency measure
    saving_interval = 200  # Model saving interval
    log_file_interval = 100  # Log file saving interval

    for episode in range(0, 0 + no_of_episodes):

        # log file
        if (episode + 1) % log_file_interval == 0:
            train_summary_writer = tf.summary.create_file_writer(train_log_dir)

        # Generating multiple episodes
        Observations, Actions, Returns, Return_Base, status, Loss = generate_eps(episode_length, Dof,
                                                                                 task, baseline_Network,
                                                                                 actor_network, opt_baseline,
                                                                                 loss_baseline_obj, scale_vec)
        if status == "Done":
            done_ = done_ + 1

        # Calculating average episodic return
        Avg_eps_return = np.sum(Returns) / len(Returns)

        # Calculating average negative log probability of an episode
        Loss_value = np.sum(Loss) / len(Loss)

        # Writing to csv files
        gradfile.write(f"\n\n Mean gradients of Episode{episode + 1}: Optimizer --> " + str(opt.get_config()))
        weightfile.write(
            f"\n\n Mean weights of Episode{episode + 1} (after update): Optimizer --> " + str(opt.get_config()))
        obsfile.write(f"\n\n Observations of Episode{episode + 1}: ")
        actfile.write(f"\n\n Actions of Episode{episode + 1}: ")

        # Computing loss
        for k in range(len(Observations)):
            with tf.GradientTape() as tape:
                # Compute Gaussian loss
                loss_value = CustomLossGaussian(Observations[k], Actions[k], Return_Base[k], Dof, actor_network,
                                                scale_vec)

            # Compute Gradients
            grads = tape.gradient(loss_value, actor_network.trainable_variables)

            # Apply gradients to update network weights
            opt.apply_gradients(zip(grads, actor_network.trainable_variables))

            # writing observations and actions in csv
            obsfile.write('\n' + f"step{k + 1}: " + str(Observations[k]))
            actfile.write('\n' + f"step{k + 1}: " + str(Actions[k]))

            # saving gradients to csv file, to check gradient explosion
            gradfile.write('\n' + f"step{k + 1}: " + str([np.mean(grads[i]) for i in range(8)]))

            # saving weights to csv file, to check nan
            weightfile.write('\n' + f"step{k + 1}: " + str([np.mean(actor_network.trainable_variables[i].numpy()) for i in range(8)]))

        # success frequency plot
        if (episode + 1) % status_interval == 0:
            value = done_ / status_interval
            count = count + 1
            done_ = 0

        # Logging the data
        with train_summary_writer.as_default():
            tf.summary.scalar('Average_Episodic_Return', Avg_eps_return, step=episode)
            tf.summary.scalar('Uncertainty_in_Action_sampling', Loss_value, step=episode)
            tf.summary.scalar('Success frequency', value, step=count)

        # Printing in the Console
        template = 'Episode {}, Average Episodic Return: {}, Loss (Sampling Uncertainty): {}, Task completion: {}'
        print(template.format(episode + 1, Avg_eps_return, Loss_value, status))

        # Save the entire model as a '.h5' at a particular interval
        if (episode + 1) % saving_interval == 0:
            model_log_dir = 'Saved_models_Reinforce/h5/' + Name + current_time + 'episode-' + str(episode + 1)
            model_log_dir_1 = 'SavedBaseline_Reinforce/h5/' + Name + current_time + 'episode-' + str(episode + 1)
            actor_network.save(model_log_dir + '.h5', include_optimizer=True)
            baseline_Network.save(model_log_dir_1 + '.h5', include_optimizer=True)

        # Clearing the memory
        del Observations
        del Actions
        del Returns
        del Return_Base
        del status
        del Loss
        del Avg_eps_return
        del Loss_value
        gc.collect()

    # closing the opened files
    gradfile.close()
    weightfile.close()
    obsfile.close()
    actfile.close()

    # Terminating the RLBench environment
    env.shutdown()


if __name__ == "__main__":
    main()
