import pathlib

import flyte
import flyte.io
import flyte.report
import torch

import aigym.types as types
from aigym.env import WikipediaGymEnv
from examples.agent_training import TrainingArgs, main

image = (
    flyte.Image.from_debian_base(name="aigym-agent-training")
    # .with_uv_project(
    #     pyproject_file=pathlib.Path("pyproject.toml"),
    #     extra_args="--extra peft --extra flyte --extra wandb --extra trl",
    #     pre=True,
    # )
    .with_pip_packages(
        "beautifulsoup4",
        "html2text",
        "httpx",
        "gymnasium",
        "markdown",
        "markdownify",
        "numpy",
        "pydantic",
        "pygame",
        "python-dotenv",
        "rich",
        "tiktoken",
        "wandb",
        "peft",
        "transformers",
        "torch>=2.7.0",
        "bitsandbytes",
        "flyte==2.0.0b24",
    )
)


reporting_env = flyte.TaskEnvironment(
    name="aigym-agent-reporting",
    resources=flyte.Resources(cpu="2", memory="500Mi"),
    reusable=flyte.ReusePolicy(replicas=1, idle_ttl=60),
    image=image.with_pip_packages("unionai-reuse"),
)

env = flyte.TaskEnvironment(
    name="aigym-agent-training",
    resources=flyte.Resources(cpu="16", memory="64Gi", gpu="L40s:4"),
    image=image,
    secrets=[
        flyte.Secret(key="huggingface_token", as_env_var="HF_TOKEN"),
        flyte.Secret(key="wandb_api_key", as_env_var="WANDB_API_KEY"),
    ],
    depends_on=[reporting_env],
)

class ReportLogger:

    HTML_TEMPLATE = """
    <style>
    * {{
        font-family: 'Open Sans', sans-serif;
    }}

    .report-content ol, .report-content ul {{
        list-style: inherit;
    }}

    .report-content pre {{
        background-color: #f6f8fa;
        border-radius: 6px;
        padding: 16px;
        overflow: auto;
    }}

    .report-content code {{
        font-family: ui-monospace, SFMono-Regular, SF Mono, Menlo, Consolas, Liberation Mono, monospace;
        font-size: 85%;
    }}
    </style>
    <body>
    <div class="report-content">
    {html}
    </div>
    </body>
    """

    def __init__(self):
        self.html = ""

    def log_training_run_info(self, env: WikipediaGymEnv, wandb_url: str | None):
        html = "<h1>🚀 Training Run Info</h1>"
        html += "<ul>"
        html += f'<li><strong>Wandb URL:</strong> <a href="{wandb_url}" target="_blank">{wandb_url}</a></li>'
        html += "</ul>"
        flyte.report.log(self.HTML_TEMPLATE.format(html=html))
        flyte.report.flush()

    def log_environment(self, env: WikipediaGymEnv):
        text = "<h1>🛣️ Travel Path</h1>"
        text += "<ol>"
        for i, url in enumerate(env.travel_path):
            text += f'<li><a href="{url}" target="_blank">{url}</a></li>'
        text += "</ol>"
        self.html += text

    def log_actions(self, actions: list[types.Action], step_action_index: int | None):
        self.html += "<h1>Actions</h1>"
        for i, action in enumerate(actions):
            if step_action_index is not None and step_action_index == i:
                selected_action_text = f" 👈 selected"
            else:
                selected_action_text = ""

            completion = str(action.completion).replace("<", "&lt;").replace(">", "&gt;")
            text = f"""
                <div class="action {'invalid' if action.action is None else 'valid'}">
                    <h3>{'❌ Invalid Action' if action.action is None else '✅ Action'} [{i}]{selected_action_text}</h3>
                    <ul>
                    {'<li><strong>Error type:</strong> ' + str(action.error_type) + '</li>' if action.action is None else ''}
                    {'<li><strong>Action:</strong> ' + str(action.action) + '</li>' if action.action else ''}
                    {'<li><strong>URL:</strong> <a href="' + str(action.url) + ' " target="_blank">' + str(action.url) + '</a></li>' if action.url else ''}
                    {'<li><strong>Reasoning:</strong> ' + action.reason_summary + '</li>' if action.reason_summary else ''}
                    <li><strong>Parse type:</strong> {str(action.parse_type)}</li>
                    <li><strong>Completion:</strong> <pre><code>{completion}</code></pre></li>
                    </ul>
                </div>
            """
            self.html += text

    def log_observation(self, observation: types.Observation):
        text = "<h1>🌍 Observation</h1>"
        text += "<ul>"
        text += f'<li>Current URL: <a href="{observation.url}" target="_blank">{observation.url}</a></li>'
        text += f'<li>Next URL: <a href="{observation.next_url}" target="_blank">{observation.next_url}</a></li>'
        text += f'<li>Target URL: <a href="{observation.target_url}" target="_blank">{observation.target_url}</a></li>'
        text += "</ul>"
        self.html += text

        self.html += "<h2>Context</h2>"
        text = f'<p>Page: <a href="{observation.url}" target="_blank">{observation.url}</a></p>'
        if len(observation.chunk_names) > 0:
            text += "<p>Table of contents:</p>"
            text += "<ul>"
            for i, chunk_name in enumerate(observation.chunk_names):
                text += f'<li><a href="{observation.url}#{chunk_name}" target="_blank">{chunk_name}</a></li>'
            text += "</ul>"

        self.html += text
        self.html += f"""
        <details>
        <summary>Raw text</summary>
        <pre><code>{observation.context}</code></pre>
        </details>
        """

    def log_metrics(self, metrics: dict[str, float]):
        text = "<h1>📊 Metrics</h1>"
        text += "<ul>"
        for key, value in metrics.items():
            text += f'<li><strong>{key}:</strong> {value}</li>'
        text += "</ul>"
        self.html += text

    def log_rewards(self, rewards: torch.Tensor, returns: torch.Tensor):
        text = "<h1>💰 Rewards</h1>"
        text += "<ul>"
        text += f'<li><strong>Raw rewards:</strong> {rewards.squeeze().tolist()}</li>'
        text += f'<li><strong>Returns:</strong> {returns.squeeze().tolist()}</li>'
        text += "</ul>"
        self.html += text

    def flush(self, episode: int, step: int):
        with flyte.group(name=f"report-episode-{episode:04d}"):
            report_step.override(short_name=f"step{step:04d}")(
                self.HTML_TEMPLATE.format(html=self.html)
            )
        self.html = ""


@reporting_env.task(report=True)
def report_step(html: str):
    flyte.report.log(html)
    flyte.report.flush()


async def report_model_dir(model_dir: pathlib.Path):
    text = "<h1>💾 Model Files</h1><ul>"
    for f in model_dir.glob("**/*"):
        text += f"<li>{f.name}</li>"
    text += "</ul>"
    await flyte.report.log.aio(text, do_flush=True)


@env.task(report=True)
async def agent_training(args: TrainingArgs) -> flyte.io.Dir:
    main(args, ReportLogger())
    await report_model_dir(pathlib.Path(args.model_save_dir))
    return await flyte.io.Dir.from_local(args.model_save_dir)


if __name__ == "__main__":
    from transformers import HfArgumentParser

    parser = HfArgumentParser(TrainingArgs)
    training_args, *_ = parser.parse_args_into_dataclasses()

    flyte.init_from_config("./config.yaml")
    run = (
        flyte.run(
            agent_training.override(short_name=training_args.experiment_name),
            args=training_args,
        )
    )
    print(run.url)
