"""Example usage of the Web Gym environment."""

import os
from typing import Generator

import dotenv
from google import genai
from google.genai import types
from rich import print as rprint

import aigym.pprint as pprint
from aigym.agent import Agent
from aigym.env import WikipediaGymEnv


def main():
    dotenv.load_dotenv()
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    def policy(prompt: str) -> Generator[str, None, None]:
        for chunk in client.models.generate_content_stream(
            model="gemini-2.0-flash",
            contents=[prompt],
            config=types.GenerateContentConfig(
                max_output_tokens=5000,
                temperature=0.2,
            ),
        ):
            delta = chunk.candidates[0].content.parts[0].text
            if delta is None:
                yield ""
                break
            yield delta

    n_hops = 2
    n_tries_per_hop = 5
    n_tries = n_hops * n_tries_per_hop
    agent = Agent(policy=policy)
    env = WikipediaGymEnv(n_hops=n_hops)
    observation, info = env.reset_manual(
        [
            "https://en.wikipedia.org/wiki/The_Primevals",
            "https://en.wikipedia.org/wiki/Juliet_Mills",
            "https://en.wikipedia.org/wiki/Miniseries",
        ]
    )
    succeeded = False

    pprint.print_travel_path(env.travel_path)
    for step in range(1, n_tries + 1):
        pprint.print_observation(observation)
        pprint.print_context(observation)
        action = agent.act(observation)
        if action is None:
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
