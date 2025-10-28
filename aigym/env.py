"""Environment for navigating the web."""

import random
import urllib.parse
from urllib.parse import ParseResult
from typing import Any

import gymnasium as gym
from rich import print as rprint

import aigym.prompts as prompts
from aigym.exceptions import NoPathsFoundError, InvalidActionError
from aigym.spaces import Tokens, WebGraph, WikipediaGraph
from aigym.types import Action, InternalEnvState, Observation


class Env(gym.Env):
    """AIGym environment."""

    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(
        self,
        web_graph: WebGraph,
        n_hops: int | None = None,
        tokenizer: Any | None = None,
        render_mode: str | None = None,
        prompt_template: str | None = None,
        url_boundaries: list[str] | None = None,
        start_url_anchors: list[str] | None = None,
        **kwargs,
    ):
        """Initialize the environment.

        Args:
            web_graph: The web graph to use for the environment.
            n_hops: The start url will be sampled n_hops away from the target
                page. For each hop, the search ensures that the page links
                back to the previous page.
            tokenizer: The tokenizer to use for the action space.
            render_mode: The mode to render the environment in.
            chunk_pattern: regex pattern to chunk on
            prompt_template: The template to use for the prompt.
            url_boundaries: The url boundaries to use for the environment.
            start_url_anchors: Always start in one of these url anchors.
        """
        # this is a gym.Env attribute
        self.render_mode = render_mode

        # aigym-specific attributes
        self.graph: WebGraph = web_graph
        self.action_space: Tokens = Tokens(tokenizer=tokenizer)
        self.n_hops = n_hops
        self.prompt_template = prompt_template
        self.url_boundaries = url_boundaries
        self.start_url_anchors = start_url_anchors or []

        # per episode attributes
        self.start_url = None
        self.target_url = None
        self.resolved_target_url = None
        self.travel_checkpoints = []
        self.travel_path = []

        # TODO: add invalid actions per episode to add to the observation

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        # initialize the window that will display the environment and the clock
        # to ensure the environment is rendered at the correct framerate in
        # human mode
        self.window = None
        self.clock = None

        self._state = InternalEnvState()

    @property
    def travel_map(self) -> dict:
        map = {}
        for current_url, next_url in zip(self.travel_path[:-1], self.travel_path[1:]):
            map[current_url] = next_url

        # the last page is the target page, so it should not have a next page
        map[self.travel_path[-1]] = None
        return map

    def _initialize_target_url(self, start_url: str, n_hops: int, n_retries: int = 100) -> tuple[str, list[str]]:
        _start_page = self.graph.get_page(
            start_url,
        ).page_chunks[0]

        travel_path = [_start_page.url]
        _page = _start_page

        for retry in range(n_retries):
            try:
                for i in range(1, n_hops + 1):
                    next_page = self.graph.random_hop(
                        _page,
                        set(travel_path + [urllib.parse.urlparse(x).path for x in travel_path]),
                    )
                    travel_path.append(next_page.url)
                    _page = next_page

                if len(travel_path) != len(set(travel_path)):
                    # try again if there are duplicate pages in the travel path
                    continue
                else:
                    break

            except NoPathsFoundError:
                if retry < n_retries - 1:
                    travel_path = []
                    _page = _start_page
                    continue
                raise

        assert len(travel_path) == len(set(travel_path)), f"Travel path contains duplicates: {travel_path}"
        return _page.url, travel_path

    def random_start(self) -> str:
        return str(self.graph.session.get(self.graph.RANDOM_URL, follow_redirects=True).url)

    def random_start_url_anchor(self) -> str:
        url = random.choice(self.start_url_anchors)
        return str(self.graph.session.get(url, follow_redirects=True).url)

    def _get_first_observation(self):
        current_web_page = self.graph.get_page(self.start_url).page_chunks[0]

        # set new internal state
        self._state.current_web_page = current_web_page
        self._state.current_chunk_index = 0  # consider making this random
        try:
            next_url = self.travel_map[self._state.current_web_page.url]
        except KeyError as e:
            raise KeyError(
                f"Next url not found for {self._state.current_web_page.url} with travel map {self.travel_map}"
            ) from e

        observation = Observation(
            url=self._state.current_web_page.url,
            context=self._state.current_web_page.context,
            prompt=self.format_prompt(self._state.current_web_page.context, self._state.current_web_page.url, self.target_url, self.url_boundaries),
            chunk_names=list(x for x in self._state.current_web_page.page_chunk_map if x is not None),
            url_boundaries=self.url_boundaries,
            target_url=self.target_url,
            next_url=next_url,
            travel_path=self.travel_path,
            current_chunk=self._state.current_chunk_index + 1,
            total_chunks=len(self._state.current_web_page.content_chunks),
        )
        info = {"travel_path": self.travel_path}
        return observation, info

    def reset_manual(
        self,
        travel_path: list[str],
    ):
        self.start_url = travel_path[0]
        self.target_url = travel_path[-1]
        self.resolved_target_url = self.graph.get_page(self.target_url).url
        self.travel_path = travel_path
        self.travel_checkpoints = [self.start_url]
        return self._get_first_observation()

    def reset(
        self,
        start_url: str | None = None,
        seed: int | None = None,
        options: dict | None = None,
        n_retries: int = 100,
    ) -> tuple[Observation, dict]:
        """Reset the environment."""
        if start_url is not None:
            self.start_url = start_url
        elif self.start_url_anchors:
            self.start_url = self.random_start_url_anchor()
        else:
            self.start_url = self.random_start()

        # NOTE: look into removing this retry loop since there already is one
        # in the _initialize_target_url function
        for retry in range(n_retries):
            try:
                self.target_url, self.travel_path = self._initialize_target_url(self.start_url, self.n_hops, n_retries)
                break
            except NoPathsFoundError as exc:
                if retry < n_retries - 1:
                    self.start_url = self.random_start()
                    print(f"Retry {retry} failed with error: {exc}, random start with new start url: {self.start_url}")
                    continue
                raise

        self.resolved_target_url = self.graph.get_page(self.target_url).url
        self.start_url = self.travel_path[0]
        self.travel_checkpoints = [self.start_url]
        observation, info = self._get_first_observation()
        return observation, info

    @staticmethod
    def _is_target_page(url: ParseResult, target_url: ParseResult) -> bool:
        return (
            url.netloc == target_url.netloc
            and (url.path == target_url.path or url.path.lower() == target_url.path.lower())
            and url.fragment == target_url.fragment
        )

    def _current_page_is_target(self):
        _current_url = urllib.parse.urlparse(self._state.current_web_page.url)
        _resolved_current_url = urllib.parse.urlparse(str(self.graph.get_page(_current_url.geturl()).url))
        _target_url = urllib.parse.urlparse(self.target_url)
        _resolved_target_url = urllib.parse.urlparse(self.resolved_target_url)
        return (
            self._is_target_page(_resolved_current_url, _target_url)
            or self._is_target_page(_resolved_current_url, _resolved_target_url)
        )

    def step(self, action: Action) -> tuple[Observation, float, bool, bool, dict]:
        """Take a step in the environment."""
        if action.action == "visit_url":
            try:
                self._state.current_web_page = self.graph.get_page(action.url)
            except ValueError as e:
                raise InvalidActionError(f"Invalid step taken at {action.url}. Error: {e}") from e
            self._state.current_chunk_key = urllib.parse.urlparse(action.url).fragment
            self._state.current_chunk_index = 0
        else:
            raise InvalidActionError(f"invalid action: {action}")

        current_page = self._state.current_web_page
        _next_url = self.travel_map[self.travel_checkpoints[-1]]

        # if the action matches the next url add it to the travel checkpoints
        if current_page.url == _next_url:
            next_url = self.travel_map[current_page.url]
            self.travel_checkpoints.append(current_page.url)
        else:
            next_url = _next_url

        observation = Observation(
            url=current_page.url,
            context=current_page.context,
            prompt=self.format_prompt(
                context=current_page.context,
                current_url=current_page.url,
                target_url=self.target_url,
                url_boundaries=self.url_boundaries,
            ),
            url_boundaries=self.url_boundaries,
            chunk_names=list(x for x in current_page.page_chunk_map if x is not None),
            target_url=self.target_url,
            next_url=next_url,
            travel_path=self.travel_path,
            current_chunk=current_page.content_chunk_index,
            total_chunks=len(current_page.page_chunks),
        )
        terminated = self._current_page_is_target()
        # alternatively, this would be distance to the target, but that would
        # require a routine to do random walks on the web graph starting from
        # the target
        reward = 1 if terminated else 0
        truncated = False
        info = {}
        return observation, reward, terminated, truncated, info

    def format_prompt(
        self,
        context: str,
        current_url: str,
        target_url: str,
        url_boundaries: list[str] | None,
    ) -> str:
        return self.prompt_template.format(
            observation=context,
            current_url=current_url,
            target_url=target_url,
            url_boundaries=", ".join(url_boundaries) if url_boundaries else "NONE",
        )

    def render(self):
        """Render the environment."""
        raise NotImplementedError

    def close(self):
        """Close the environment."""
        ...


"""
This pattern chunks markdown-format wikipedia pages where headers are formatted
as follows:

This is a header
----------------

This is a paragraph

### This is a subheader

This is a subheader paragraph

This is another header
----------------

This is another paragraph

### This is another subheader

This is another paragraph
"""
HEADER_CHUNK_PATTERN = r"(\n.+\n-+\n|\n### .+\n)"


class WikipediaGymEnv(Env):
    """Wikipedia Gym environment."""

    def __init__(
        self,
        *args,
        wikipedia_graph: WikipediaGraph | None = None,
        prompt_template: str | None = None,
        chunk_pattern: str | None = None,
        chunk_char_limit: int | None = 10_000,
        url_boundaries: list[str] | None = None,
        **kwargs,
    ):
        if wikipedia_graph is None:
            wikipedia_graph = WikipediaGraph(
                chunk_pattern=chunk_pattern or HEADER_CHUNK_PATTERN,
                chunk_char_limit=chunk_char_limit,
            )
        super().__init__(
            wikipedia_graph,
            *args,
            prompt_template=prompt_template or prompts.WIKIPEDEA_ACTION_TEMPLATE,
            url_boundaries=url_boundaries or ["https://en.wikipedia.org"],
            **kwargs,
        )
