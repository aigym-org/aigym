"""Environment for navigating the web."""

import urllib.parse
from typing import Any

import gymnasium as gym
from rich import print as rprint

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
        """
        # this is a gym.Env attribute
        self.render_mode = render_mode

        # aigym-specific attributes
        self.graph: WebGraph = web_graph
        self.action_space: Tokens = Tokens(tokenizer=tokenizer)
        self.n_hops = n_hops

        self.start_url = None
        self.target_url = None
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

    def _initialize_target_url(self, start_url: str, n_hops: int, n_retries: int = 30) -> tuple[str, list[str]]:
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
            chunk_names=list(x for x in self._state.current_web_page.page_chunk_map if x is not None),
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
        self.travel_path = travel_path
        self.travel_checkpoints = [self.start_url]
        return self._get_first_observation()

    def reset(
        self,
        start_url: str | None = None,
        seed: int | None = None,
        options: dict | None = None,
        n_retries: int = 30,
    ) -> tuple[Observation, dict]:
        """Reset the environment."""
        if start_url is not None:
            self.start_url = start_url
        else:
            self.start_url = self.random_start()

        for retry in range(n_retries):
            try:
                self.target_url, self.travel_path = self._initialize_target_url(self.start_url, self.n_hops)
                break
            except NoPathsFoundError as exc:
                if retry < n_retries - 1:
                    self.start_url = self.random_start()
                    print(f"Retry {retry} failed with error: {exc}, random start with new start url: {self.start_url}")
                    continue
                raise

        self.start_url = self.travel_path[0]
        self.travel_checkpoints = [self.start_url]
        observation, info = self._get_first_observation()
        return observation, info

    def _current_page_is_target(self):
        _current_url = urllib.parse.urlparse(self._state.current_web_page.url)
        _target_url = urllib.parse.urlparse(self.target_url)
        return (
            _current_url.netloc == _target_url.netloc
            and (_current_url.path == _target_url.path or _current_url.path.lower() == _target_url.path.lower())
            and _current_url.fragment == _target_url.fragment
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
        chunk_pattern: str | None = None,
        chunk_char_limit: int | None = 5000,
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
            **kwargs,
        )
