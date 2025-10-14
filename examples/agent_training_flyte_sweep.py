import json
import typing
from dataclasses import dataclass, asdict

import flyte

from examples.agent_training_flyte import image, training_env, TrainingArgs, agent_training


@dataclass
class SweepArgs(TrainingArgs):
    n_hops_list: str | None = None


sweep_env = flyte.TaskEnvironment(
    name="aigym-agent-sweep",
    resources=flyte.Resources(cpu="2", memory="1Gi"),
    image=image,
    depends_on=[training_env],
)


@sweep_env.task
async def agent_training_sweep(sweep_args: SweepArgs):
    training_kwargs = {
        k: v
        for k, v in asdict(sweep_args).items()
        if k not in ["n_hops_list", "n_hops"]
    }
    assert sweep_args.n_hops_list is not None
    n_hops_list = json.loads(sweep_args.n_hops_list)
    for n_hops in n_hops_list:
        training_args = TrainingArgs(**training_kwargs, n_hops=n_hops)
        run_name = (
            training_args.experiment_name
            or f"aigym-agent-training-{training_args.n_hops}-hops"
        )
        await agent_training.override(short_name=run_name)(training_args)


if __name__ == "__main__":
    from transformers import HfArgumentParser

    parser = HfArgumentParser(SweepArgs)
    sweep_args, *_ = parser.parse_args_into_dataclasses()
    sweep_args = typing.cast(SweepArgs, sweep_args)

    flyte.init_from_config("./config.yaml")
    run_name = sweep_args.experiment_name or "aigym-agent-training-sweep"
    run = flyte.run(agent_training_sweep, sweep_args=sweep_args)
    print(run.url)
