"""Example usage of the Web Gym environment."""

from typing import Generator

from openai import OpenAI
from rich import print as rprint

import aigym.pprint as pprint
from aigym.agent import Agent
from aigym.env import WikipediaGymEnv


def main():
    client = OpenAI()

    def policy(prompt: str) -> Generator[str, None, None]:
        for chunk in client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="gpt-4o-mini",
            max_tokens=2000,
            temperature=0.2,
            stream=True,
        ):
            delta = chunk.choices[0].delta.content
            if delta is None:
                yield ""
                break
            yield delta

    agent = Agent(policy=policy)
    env = WikipediaGymEnv(n_hops=2)
    observation, info = env.reset()
    
    for step in range(1, 101):
        pprint.print_observation(observation)
        pprint.print_context(observation)
        action = agent.act(observation)
        if action is None:
            rprint(f"No action taken at step {step}")
            continue
        pprint.print_action(action)
        observation, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            rprint(f"Episode terminated or truncated at step {step}")
            break

    rprint("Task finished!")
    env.close()


if __name__ == "__main__":
    main()
