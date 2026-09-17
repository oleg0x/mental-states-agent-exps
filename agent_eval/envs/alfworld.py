from typing import Dict, List, Tuple

import textworld
import yaml
from alfworld.agents.environment.alfred_tw_env import (
    AlfredDemangler,
    AlfredExpert,
    AlfredInfos,
)

from agent_eval.envs.base import BaseEnv
from agent_eval.envs.react import format_admissible_actions, parse_react_action
from agent_eval.paths import ALFWORLD_DATA_DIR
from agent_eval.state import State
from agent_eval.tasks import AlfWorldTask


ALFWORLD_REACT_TEMPLATE_NO_HIS = """
You are an expert agent operating in the ALFRED Embodied Environment.
Your current observation is: {current_observation}
Your admissible actions of the current situation are: [{admissible_actions}].

Now it's your turn to take an action.
You should first reason step-by-step about the current situation. This reasoning process MUST be enclosed within <think> </think> tags.
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags.
"""


ALFWORLD_REACT_TEMPLATE = """
You are an expert agent operating in the ALFRED Embodied Environment. Your task is to: {task_description}
Prior to this step, you have already taken {step_count} step(s). Below are the most recent {history_length} observations and the corresponding actions you took: {action_history}
You are now at step {current_step} and your current observation is: {current_observation}
Your admissible actions of the current situation are: [{admissible_actions}].

Now it's your turn to take an action.
You should first reason step-by-step about the current situation. This reasoning process MUST be enclosed within <think> </think> tags.
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags.
"""


def _process_observation(observation: str) -> str:
    if observation.startswith("You arrive at loc "):
        observation = observation[observation.find(". ") + 2 :]
    return observation


class AlfWorldEnv(BaseEnv):
    """ALFWorld wrapper preserving the working QLASS ReAct behavior."""

    def __init__(self, task: AlfWorldTask, max_steps: int = 50, **kwargs) -> None:
        super().__init__(max_steps=max_steps, **kwargs)
        self.task = task
        self.env = self._load_single_task_env(task.game_file)

        self.current_admissible_commands: List[str] = []
        self.react_history: List[Tuple[str, str]] = []
        self.react_current_observation = task.observation

    def _load_single_task_env(self, game_file: str):
        with (ALFWORLD_DATA_DIR / "base_config.yaml").open() as f:
            config = yaml.safe_load(f)

        domain_randomization = config["env"]["domain_randomization"]
        if self.task.split != "train":
            domain_randomization = False

        wrappers = [
            AlfredDemangler(shuffle=domain_randomization),
            AlfredInfos,
        ]
        request_infos = textworld.EnvInfos(
            won=True,
            admissible_commands=True,
            extras=["gamefile"],
        )

        training_method = config["general"]["training_method"]
        if training_method == "dqn":
            max_env_steps = config["rl"]["training"]["max_nb_steps_per_episode"]
        elif training_method == "dagger":
            max_env_steps = config["dagger"]["training"]["max_nb_steps_per_episode"]
            if self.task.split == "train":
                wrappers.append(AlfredExpert(config["env"]["expert_type"]))
                request_infos.extras.append("expert_plan")
        else:
            raise ValueError(f"Unsupported ALFWorld training method: {training_method}")

        max_env_steps = max(int(max_env_steps), int(getattr(self, "max_steps", 50)))

        env_id = textworld.gym.register_games(
            [game_file],
            request_infos,
            batch_size=1,
            asynchronous=True,
            max_episode_steps=max_env_steps,
            wrappers=wrappers,
        )
        return textworld.gym.make(env_id)

    def get_task_text(self) -> str:
        marker = "Your task is to:"
        if marker in self.task.observation:
            return self.task.observation.split(marker, 1)[1].strip()
        return self.task.observation.strip()

    def get_current_observation(self) -> str:
        return str(self.react_current_observation or "").strip()

    def get_admissible_commands(self) -> List[str]:
        return [
            action
            for action in self.current_admissible_commands
            if action != "help"
        ]

    def build_agent_messages(self) -> List[Dict[str, str]]:
        """
        Build the actor prompt used by the reference QLASS ReAct setup.
        """
        history_length = max(int(self.history_length), 0)

        admissible_text = format_admissible_actions(self.get_admissible_commands())

        # First step: QLASS puts the complete initial ALFWorld observation into
        # `current_observation`. That observation already contains the task.
        if not self.react_history:
            prompt = ALFWORLD_REACT_TEMPLATE_NO_HIS.format(
                current_observation=self.react_current_observation,
                admissible_actions=admissible_text,
            )

        else:
            marker = "Your task is to:"
            initial_observation = self.task.observation

            if marker in initial_observation:
                task_description = initial_observation.split(marker, 1)[1].strip()
            else:
                task_description = initial_observation.strip()

            recent_history = (
                self.react_history[-history_length:] if history_length > 0 else []
            )

            first_step = len(self.react_history) - len(recent_history) + 1

            history_lines = []

            for offset, (observation, action) in enumerate(recent_history):
                step_num = first_step + offset

                history_lines.append(
                    f"[Observation {step_num}: '{observation}', "
                    f"Action {step_num}: '{action}']"
                )

            prompt = ALFWORLD_REACT_TEMPLATE.format(
                task_description=task_description,
                step_count=len(self.react_history),
                history_length=len(recent_history),
                action_history="\n".join(history_lines),
                current_step=len(self.react_history) + 1,
                current_observation=self.react_current_observation,
                admissible_actions=admissible_text,
            )

        # The reference Qwen setup uses admissible actions. We keep this flag only
        # so future experiments can disable them without changing the prompt code.
        if not self.include_admissible_actions:
            admissible_line = (
                "Your admissible actions of the current situation are: "
                f"[{admissible_text}].\n"
            )
            prompt = prompt.replace(admissible_line, "")

        return [
            {
                "role": "user",
                "content": prompt.strip(),
            }
        ]

    def _conduct_action(self, action: str):
        observation, _, done, info = self.env.step([action])

        self.current_admissible_commands = list(
            (info.get("admissible_commands") or [[]])[0]
        )
        clean_observation = _process_observation(observation[0])
        reward = info["won"][0]
        return clean_observation, reward, done[0]

    def step(self, llm_output: str) -> Tuple[str, State]:
        self.state.history.append(
            {"role": "assistant", "content": llm_output}
        )
        observation_before = self.get_current_observation()

        try:
            action = parse_react_action(llm_output)
            observation, reward, done = self._conduct_action(action)
            self.react_history.append((observation_before, action))
            self.react_current_observation = observation
        except Exception as exc:
            self.state.success = False
            self.state.reward = 0.0
            self.state.error = str(exc)

            observation = (
                "Observation: Error Input. Your response must contain an "
                "action inside <action>...</action> tags."
            )
            self.react_history.append(
                (observation_before, "__invalid_action__")
            )
            self.react_current_observation = observation

            self.state.history.append(
                {"role": "user", "content": observation}
            )
            self.state.steps += 1

            if self.state.steps >= self.max_steps:
                self.state.finished = True
                self.state.terminate_reason = "max_steps"

            return observation, self.state

        observation_message = f"Observation: {observation}"
        self.state.history.append(
            {"role": "user", "content": observation_message}
        )
        self.state.steps += 1

        if done:
            self.state.finished = True
            self.state.success = bool(reward)
            self.state.reward = float(reward)
            self.state.terminate_reason = (
                "success" if self.state.success else "env_done"
            )
        elif self.state.steps >= self.max_steps:
            self.state.finished = True
            self.state.success = False
            self.state.reward = float(reward)
            self.state.terminate_reason = "max_steps"

        return observation_message, self.state

    def reset(self) -> Tuple[str, State]:
        self.state = State()

        self.react_history = []
        self.react_current_observation = self.task.observation

        _, info = self.env.reset()
        self.current_admissible_commands = list(
            (info.get("admissible_commands") or [[]])[0]
        )

        return self.task.observation, self.state
