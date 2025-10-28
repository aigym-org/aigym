<div align="center">
    <img src="./static/aigym-logo.png" alt="aigym logo" width="200">
    <h1>aigym</h1>
    <p>Self-supervised reinforcement learning environments for LLM fine-tuning</p>
</div>

---

`aigym` is a library that provides a suite of novel reinforcement learning (RL) environments
for the purpose of fine-tuning pre-trained language models for various reasoning tasks.

Built on top of the [gymnasium](https://gymnasium.farama.org/) API, the objective of this project is to expose a light-weight and extensible environments to fine-tune language models with techniques like [PPO](https://arxiv.org/abs/1707.06347) and [GRPO](https://arxiv.org/abs/2402.03300).

It is designed to complement training frameworks like [trl](https://huggingface.co/docs/trl/en/index),
[transformers](https://huggingface.co/docs/transformers/en/index), [pytorch](https://pytorch.org/),
and [pytorch lightning](https://lightning.ai/pytorch-lightning).

See the project roadmap [here](./ROADMAP.md)

## Installation

```bash
pip install aigym
```

## Development Installation

Install `uv`:

```bash
pip install uv
```

Create a virtual environment:

```bash
uv venv --python 3.12
```

Activate the virtual environment:

```bash
source .venv/bin/activate
```

Install the package:

```bash
uv sync --extra ollama --group dev
```

Install `ollama` to run a local model: https://ollama.com/download

## Quickstart

```python
from typing import Generator

import ollama

from aigym.agent import Agent
from aigym.env import WikipediaGymEnv


# define a policy function for the agent using Ollama
def policy(prompt: str) -> Generator[str, None, None]:
    for chunk in ollama.generate(
        model="gemma3:1b",
        prompt=prompt,
        stream=True,
    ):
        yield chunk.response

# initialize the agent with the policy function in streaming mode
agent = Agent(policy=policy, stream=True)

# initialize the wikipedia maze environment
env = WikipediaGymEnv(n_hops=2)

# create a travel path between two pages that are two hops away
observation, info = env.reset()

# allow the agent to take 10 steps to try to find the target page
for step in range(10):

    # generate an action
    action = agent.act(observation)
    if action.action is None:
        print(f"No valid action taken at step {step}")
        continue

    # take a step in the environment
    observation, reward, terminated, truncated, info = env.step(action)

    # break early if the episode is terminated
    if terminated or truncated:
        print(f"Episode terminated or truncated at step {step}")
        break
```

## Usage

The `examples` directory contains examples on how to use the `aigym` environments.
Run an ollama-based agent on the Wikipedia maze environment:

### Basic example: inference only

This example uses `ollama` to run a local model and performs rollouts of the
Wikipedia maze environment.

```bash
python examples/ollama_agent.py
```

### Training example

This example uses the `examples/agent_training.py` script to train a small model
on the Wikipedia maze environment.

```bash
python examples/agent_training.py --model_id google/gemma-3-270m-it
```

> [!NOTE]
> Because the model is low capacity, it may take some time for it to generate
> any valid actions at all, since the action space requires outputting
> correctly formatted `<think>` and `<answer>` tags, where the `<answer>` contains
> valid json.

### Training on Flyte

Flyte is an AI orchestration platform that provides an easy way to run workloads
on the cloud, including data processing, model training, model inference, and
agentic pipelines.

You can train an agent on a Flyte cluster using the `examples/agent_training_flyte.py` example:

![Flyte agent training](./static/aigym-on-flyte.gif)

Install flyte:

```bash
uv pip install '.[flyte]'
```

Then create a configuration:

```bash
flyte create config \
--endpoint demo.hosted.unionai.cloud \
--builder remote \
--project aigym \
--domain development
```

> [!NOTE]
> Modify the `--endpoint` flag to point to your Flyte cluster.

This will create a `config.yaml` file in the current directory.

<details>
<summary>Basic example:</summary>

This is the easiest difficulty setting that goes 1 hop away from the start url.

```bash
PYTHONPATH=. python examples/agent_training_flyte.py \
    --n_hops 1 \
    --model_id google/gemma-3-12b-it \
    --enable_gradient_checkpointing
```
</details>


<details>
<summary>Increased difficulty setting: five hops away</summary>

```bash
PYTHONPATH=. python examples/agent_training_flyte.py \
    --model_id google/gemma-3-12b-it \
    --enable_gradient_checkpointing \
    --n_episodes 100 \
    --lora_r 64 \
    --n_hops 5 \
    --n_tries_per_hop 4 \
    --rollout_min_new_tokens 256 \
    --rollout_max_new_tokens 512 \
    --group_size 4 \
    --wandb_project aigym-agent-training \
    --attn_implementation eager
```
</details>

<details>
<summary>Anchor the start url to the "Mammal" page</summary>

```bash
PYTHONPATH=. python examples/agent_training_flyte.py \
    --model_id google/gemma-3-12b-it \
    --start_url_anchors '["https://en.wikipedia.org/wiki/Mammal"]' \
    --enable_gradient_checkpointing \
    --n_episodes 1000 \
    --lr 1e-3 \
    --max_grad_norm 4.0 \
    --lora_r 64 \
    --n_hops 2 \
    --n_tries_per_hop 2 \
    --static_env \
    --rollout_min_new_tokens 256 \
    --rollout_max_new_tokens 512 \
    --group_size 4 \
    --wandb_project aigym-agent-training \
    --attn_implementation eager
```
</details>


<details>
<summary>Sweep with different number of hops</summary>

```bash
PYTHONPATH=. python examples/agent_training_flyte_sweep.py \
    --model_id google/gemma-3-12b-it \
    --enable_gradient_checkpointing \
    --n_episodes 100 \
    --n_hops_list "[1, 2, 3, 4, 5]" \
    --n_tries_per_hop 1 \
    --rollout_min_new_tokens 256 \
    --rollout_max_new_tokens 1024 \
    --group_size 4 \
    --wandb_project aigym-agent-training \
    --attn_implementation eager
```
</details>