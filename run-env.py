import gym
import click

from RandomAgent import RandomAgent

@click.command()
@click.option('--env_name', default='MountainCar-v0')
@click.option('--episodes', default=1)
@click.option('--max_t', default=100)
@click.option('--render/--no-render', default=True)
def main(env_name, episodes, max_t, render):
    # Instantiate the environment
    env = gym.make(env_name)

    # Instantiate agents
    random_mountain_agent = RandomAgent(env.action_space)

    # Loop through the episodes
    for episode in range(episodes):
        # Reset to obtain initial observation
        observation = env.reset()

        # Run the episode
        for t in range(max_t):
            if render: env.render()
            print(observation)

            # Agents act here based on current observation
            action = random_mountain_agent.act(observation)

            observation, reward, done, info = env.step(action) # step based on action

            if done:
                print('Episode finished after {} time steps'.format(t+1))
                break

    env.close()

if __name__ == '__main__':
    main()
