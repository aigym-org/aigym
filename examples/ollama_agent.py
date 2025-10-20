"""Example usage of the Web Gym environment."""

from typing import Generator

import ollama
from rich import print as rprint

import aigym.pprint as pprint
from aigym.agent import Agent
from aigym.env import WikipediaGymEnv


def main():
    def policy(prompt: str) -> Generator[str, None, None]:
        for chunk in ollama.generate(
            model="gemma3:4b",
            prompt=prompt,
            stream=True,
            options={
                "temperature": 1.25,
                "top_k": 64,
                "top_p": 0.95,
            },
        ):
            yield chunk.response

    agent = Agent(
        policy=policy,
        stream=True,
    )

    n_hops = 2
    n_tries_per_hop = 5
    n_tries = n_hops * n_tries_per_hop
    env = WikipediaGymEnv(n_hops=n_hops)
    observation, info = env.reset()
    succeeded = False

    pprint.print_travel_path(env.travel_path)
    for step in range(1, n_tries + 1):
        pprint.print_observation(observation)
        pprint.print_context(observation)
        action = agent.act(observation)
        if action.action is None:
            rprint(f"No action taken at step {step}")
            continue
        pprint.print_action(action)
        observation, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            succeeded = True
            rprint(f"Episode terminated or truncated at step {step}")
            break

    rprint(f"Finished after {step} tries")
    if succeeded:
        rprint("✅ Target found")
    else:
        rprint("❌ Target not found")
    env.close()


if __name__ == "__main__":
    main()
