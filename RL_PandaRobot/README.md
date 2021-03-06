# Evaluation of REINFORCE on benchmark tasks for Panda in RLBench

## Trajectory of the trained 7 Dof Panda agent
https://user-images.githubusercontent.com/75635905/130522631-399409eb-a648-4217-88a1-a8a55d6ab26b.mp4

## Learning Goals

- Understand the REINFORCE (Classical Monte Carlo Policy Gradient) Algorithm
- Understand the framework of RLBench Robot Learning Environment & Benchmark
- Understand the implementation of REINFORCE, and its integration with continuous control simulated environment 
- Understand the effect of baseline in reducing the variance and improving the learning speed
- Understand the effect of learning rate decay on REINFORCE
- Train the Panda agent to perform a benchmark task in RLBench, and interpret the results
- Understand the limitations of the algorithm and the directions of future research

## Steps involved in Algorithm Development and Deployment

- Task selection -> RLBench includes 100 unique tasks ranging from easy to more challenging ones, which can even stress-test well-known state-of-the-art algorithms in use today.
- Stochastic Policy Representation -> Eg., Gaussian distribution, Gaussian with diagonal covariance, Gaussian Mixture Model, beta distribution, etc.
- Reward shaping -> Engineering a reward function depends on the goal and the difficulty level of the task to be trained. Due to human cognitive bias, the transformation of human knowledge into numeric reward values is often not ideal, and hence it is challenging for complicated tasks.
- Trajectory generation -> Since REINFORCE is well defined only for the episodic case, trajectory generation is a mandatory step, before updating the parameters of the model
- Algorithm deployment -> Using the training data obtained from the generated trajectory, the parameters of the model are updated using stochastic gradient ascent REINFORCE update formula. The weights of the baseline network is updated in a supervised manner, with the "state-value" as the target.

## Results

- Reduced almost 75% of the training time by using the state-value function as a baseline.
- Decaying the learning rate (hardly studied in RL) by a factor of 10 helped to avoid the early deterministic behavior of the agent.
- Achieved 99% success rate on a simplified benchmark task in nearly 25000 episodes.
- Investigated the need to maintain the stochasticity of action sampling since the agent is unable to explore even after 200k episodes when learning a complex stochastic policy.

## Prerequisites

- OS -> Linux
- Coppeliasim 4.1 [Downloads](https://www.coppeliarobotics.com/downloads)
- PyRep (requires Linux) [Github Repo](https://github.com/stepjam/PyRep)
- RLBench [Github Repo](https://github.com/stepjam/RLBench)
- Python 3.7
- Tensorflow 2.2 (or higher)
