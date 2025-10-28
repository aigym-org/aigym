"""Example usage of the Web Gym environment."""

from typing import Generator

import ollama

import aigym.pprint as pprint
from aigym.agent import Agent
from aigym.env import WikipediaGymEnv


def main():

    # define a policy function for the agent using Ollama
    def policy(prompt: str) -> Generator[str, None, None]:
        for chunk in ollama.generate(
            model="gemma3:4b",
            prompt=prompt,
            stream=True,
            options={"temperature": 1.25,
                "top_k": 64,
                "top_p": 0.95,
            },
        ):
            yield chunk.response

    # initialize the agent with the policy function in streaming mode
    agent = Agent(policy=policy, stream=True)

    # initialize the environment and reset it to create a new travel path
    env = WikipediaGymEnv(n_hops=2)

    # this will create a travel path between two pages that are two hops away
    observation, info = env.reset()
    succeeded = False

    pprint.print_travel_path(env.travel_path)

    # allow the agent to take 10 steps to try to find the target page
    for step in range(10):
        pprint.print_observation(observation)
        pprint.print_context(observation)
        action = agent.act(observation)

        if action.action is None:
            print(f"No valid action taken at step {step}")
            continue

        pprint.print_action(action)

        # take a step in the environment
        observation, reward, terminated, truncated, info = env.step(action)

        # break early if the episode is terminated
        if terminated or truncated:
            succeeded = True
            print(f"Episode terminated or truncated at step {step}")
            break

    print(f"Finished after {step} tries")
    if succeeded:
        print("✅ Target found")
    else:
        print("❌ Target not found")
    env.close()


if __name__ == "__main__":
    main()
