import argparse
import copy
import json
import logging
import random
import sys
from tqdm import tqdm
from pathlib import Path
from typing import Any, Dict

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agent_eval.agents import AGENT_REGISTRY
from agent_eval.compat import apply_sciworld_step_patch
from agent_eval.envs import AlfWorldEnv, SciWorldEnv, WebShopEnv
from agent_eval.envs.react import parse_react_action
from agent_eval.paths import SCIENCEWORLD_JAR
from agent_eval.tasks import AlfWorldTask, SciWorldTask, WebShopTask
#from agent_eval.critics import CRITIC_REGISTRY
from agent_eval.critics.base import BaseCritic
from agent_eval.critics.qnet import QwenQNetCritic


logger = logging.getLogger("agent_eval")


# Deliberately explicit registries. Adding a benchmark later should require
# adding its classes here, rather than introducing a complex plugin system.
TASK_REGISTRY = {
    "alfworld": AlfWorldTask,
    "sciworld": SciWorldTask,
    "webshop": WebShopTask,
}

ENV_REGISTRY = {
    "alfworld": AlfWorldEnv,
    "sciworld": SciWorldEnv,
    "webshop": WebShopEnv,
}

CRITIC_REGISTRY = {
    "qwen_qnet": QwenQNetCritic,
}


def _load_yaml(path: Path) -> Dict[str, Any]:
    """Load and validate a YAML config."""
    with path.open() as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Expected a mapping in YAML file: {path}")

    return data


def _repo_path(path_value: str) -> Path:
    """Resolve repository-relative config/output paths."""
    path = Path(path_value)

    if path.is_absolute():
        return path

    return REPO_ROOT / path


def _prepare_benchmark_runtime(
    benchmark: str,
    env_config: Dict[str, Any],
):
    """Create optional shared runtime objects for a benchmark.

    ALFWorld creates its own TextWorld environment for each task.

    ScienceWorld uses one Java-backed ScienceWorldEnv instance and loads a different task into it before each episode, matching current QLASS usage.
    """
    if benchmark == "webshop":
        from eval.webshop.web_agent_site.envs import WebAgentTextEnv

        # Exact runtime configuration used by QLASS.
        return WebAgentTextEnv(observation_mode="text", human_goals=True)

    if benchmark != "sciworld":
        return None

    if not SCIENCEWORLD_JAR.exists():
        raise FileNotFoundError(
            f"ScienceWorld JAR was not found: {SCIENCEWORLD_JAR}. "
            "Copy it from the QLASS repository."
        )

    from scienceworld import ScienceWorldEnv

    # Required for the reward/completion metadata expected by SciWorldEnv.
    apply_sciworld_step_patch()

    internal_step_limit = int(env_config.get("internal_env_step_limit", 200))

    return ScienceWorldEnv(
        "",
        serverPath=str(SCIENCEWORLD_JAR),
        envStepLimit=internal_step_limit,
    )


def _build_env(
    benchmark: str,
    task,
    env_config: Dict[str, Any],
    runtime,
):
    """Instantiate the benchmark wrapper for one episode."""
    env_cls = ENV_REGISTRY[benchmark]

    # These fields configure dataset loading/runtime creation and should not be forwarded to BaseEnv.
    wrapper_config = {
        key: value for key, value in env_config.items() if key
        not in {
            "name",
            "split",
            "part_num",
            "part_idx",
            "internal_env_step_limit",
        }
    }

    if benchmark in {"sciworld", "webshop"}:
        return env_cls(task=task, env=runtime, **wrapper_config)

    return env_cls(task=task, **wrapper_config)


def _task_metadata(benchmark: str, task) -> Dict[str, Any]:
    """Store benchmark-specific identifiers without leaking them into runner logic."""
    if benchmark == "alfworld":
        return {"game_file": task.game_file}

    if benchmark == "sciworld":
        return {
            "sub_task_name": task.sub_task_name,
            "variation_idx": task.variation_idx,
        }

    if benchmark == "webshop":
        return {"session_id": task.session_id}

    return {}


def _replay_episode_prefix(env, executed_outputs):
    """
    Reset the environment and replay only actions selected so far.

    Функция нужна, чтобы можно было понять, завершает ли данное действие траекторию или нет.
    То есть нам приходится сделать env.step, чтоб посмотреть что будет дальше и из-за этого
    для остальных действий приходится replay'ится
    """
    _, state = env.reset()

    for raw_output in executed_outputs:
        _, state = env.step(raw_output)

        if state.finished:
            raise RuntimeError(
                "Environment became terminal while replaying the already selected trajectory prefix."
            )

    return state


def _run_episode_with_critic(
    agent,
    critic,
    env,
    task,
    benchmark: str,
    split: str,
    attempt_id: int,
    experiment_name: str,
    num_candidates: int,
) -> Dict[str, Any]:
    """Run one actor trajectory with QNet best-of-N action selection."""

    _, state = env.reset()

    initial_observation = env.get_current_observation()
    task_text = env.get_task_text()

    step_records = []

    # Only outputs actually selected by the critic are stored here.
    # Temporary candidate actions must never enter the real trajectory
    executed_outputs = []

    while not state.finished:
        messages = env.build_agent_messages()

        observation_before = env.get_current_observation()
        admissible_actions = env.get_admissible_commands()

        # --------------------------------------------------------------
        # 1. Generate candidate actions from exactly the same actor state.
        # --------------------------------------------------------------
        candidate_outputs = []

        try:
            for candidate_id in range(num_candidates):
                raw_output = agent.act(messages)

                try:
                    parsed_action = parse_react_action(raw_output)
                except ValueError:
                    parsed_action = "__invalid_action__"

                candidate_outputs.append(
                    {
                        "candidate_id": candidate_id,
                        "raw_agent_output": raw_output,
                        "parsed_action": parsed_action,
                    }
                )

        except Exception as exc:
            state.finished = True
            state.success = False
            state.terminate_reason = "agent_error"
            state.error = str(exc)

            logger.exception(
                "Agent failed on task=%s attempt=%d",
                task.task_id, attempt_id,
            )

            break

        # --------------------------------------------------------------
        # 2. Evaluate every candidate from the same environment state.
        # --------------------------------------------------------------
        candidate_records = []

        for candidate in candidate_outputs:
            _replay_episode_prefix(
                env,
                executed_outputs,
            )

            candidate_observation, candidate_state = env.step(
                candidate["raw_agent_output"]
            )

            # Preserve QLASS terminal handling:
            # once the environment knows the outcome, use ground-truth
            # environment reward rather than asking QNet to predict it.
            if candidate_state.finished:
                critic_score = float(
                    candidate_state.reward
                    if candidate_state.reward is not None else 0.0
                )
                score_source = "environment_terminal_reward"

            else:
                critic_score = critic.score_candidate(
                    actor_messages=messages,
                    raw_action=candidate["raw_agent_output"],
                )
                score_source = "qnet"

            candidate_records.append(
                {
                    **candidate,
                    "critic_score": critic_score,
                    "score_source": score_source,
                    "next_observation": candidate_observation,
                    "reward": candidate_state.reward,
                    "done": candidate_state.finished,
                    "success": candidate_state.success,
                    "terminate_reason": candidate_state.terminate_reason,
                }
            )

        # --------------------------------------------------------------
        # 3. Greedily select the candidate with maximum Q-value.
        # --------------------------------------------------------------
        selected_idx = max(
            range(len(candidate_records)),
            key=lambda idx: candidate_records[idx]["critic_score"],
        )

        for idx, candidate in enumerate(candidate_records):
            candidate["selected"] = idx == selected_idx

        selected = candidate_records[selected_idx]

        # --------------------------------------------------------------
        # 4. Reconstruct the original state and execute only the winner.
        # --------------------------------------------------------------
        _replay_episode_prefix(
            env,
            executed_outputs,
        )

        observation_after, state = env.step(
            selected["raw_agent_output"]
        )

        executed_outputs.append(
            selected["raw_agent_output"]
        )

        step_records.append(
            {
                "step_id": state.steps,
                "observation": observation_before,
                "agent_messages": copy.deepcopy(messages),
                "admissible_actions": list(admissible_actions),

                # Keep the original trajectory fields unchanged.
                "raw_agent_output": selected["raw_agent_output"],
                "parsed_action": selected["parsed_action"],
                "next_observation": observation_after,
                "reward": state.reward,
                "done": state.finished,

                # Critic-specific diagnostics.
                "selected_candidate_id": selected["candidate_id"],
                "selection_reason": "qnet_argmax",
                "candidates": candidate_records,
            }
        )

    return {
        "experiment_name": experiment_name,
        "benchmark": benchmark,
        "split": split,
        "task_id": task.task_id,
        "attempt_id": attempt_id,
        "task_text": task_text,
        "initial_observation": initial_observation,
        "success": state.success,
        "reward": state.reward,
        "num_steps": state.steps,
        "terminate_reason": state.terminate_reason,
        "error": state.error,
        "task": _task_metadata(benchmark, task),
        "steps": step_records,
    }


def _run_episode(
    agent,
    env,
    task,
    benchmark: str,
    split: str,
    attempt_id: int,
    experiment_name: str,
) -> Dict[str, Any]:
    """Run one pure actor -> environment trajectory."""
    _, state = env.reset()

    initial_observation = env.get_current_observation()
    task_text = env.get_task_text()

    step_records = []

    # Для экспертных траекторий тут просто должен быть цикл for expert_step in expert_trajectory:
    while not state.finished:
        # Environment owns benchmark-specific prompt construction.
        messages = env.build_agent_messages()

        observation_before = env.get_current_observation()

        admissible_actions = env.get_admissible_commands()

        try:
            raw_output = agent.act(messages)

        except Exception as exc:
            # Preserve the failed episode instead of silently losing it.
            state.finished = True
            state.success = False
            state.terminate_reason = "agent_error"
            state.error = str(exc)

            logger.exception(
                "Agent failed on task=%s attempt=%d",
                task.task_id, attempt_id,
            )

            break

        # Save the action separately from the raw model output.
        try:
            # вот это действие после парсинга нам надо сравнить с экспертным действием - expert_step["action"]
            parsed_action = parse_react_action(raw_output)
        except ValueError:
            parsed_action = "__invalid_action__"

        # в случае экспертных траекторий переходить дальше нужно строго по expert trajectory
        observation_after, state = env.step(raw_output)

        step_records.append(
            {
                "step_id": state.steps,
                "observation": observation_before,
                "agent_messages": copy.deepcopy(messages),
                "admissible_actions": list(admissible_actions),
                "raw_agent_output": raw_output,
                "parsed_action": parsed_action,
                "next_observation": observation_after,
                "reward": state.reward,
                "done": state.finished,
            }
        )

    return {
        "experiment_name": experiment_name,
        "benchmark": benchmark,
        "split": split,
        "task_id": task.task_id,
        "attempt_id": attempt_id,
        "task_text": task_text,
        "initial_observation": initial_observation,
        "success": state.success,
        # Environment-level episode reward after executing this action.
        # For ScienceWorld this follows the current QLASS behavior and is the best raw_score observed so far,
        # not the immediate reward delta.
        "reward": state.reward,
        "num_steps": state.steps,
        "terminate_reason": state.terminate_reason,
        "error": state.error,
        "task": _task_metadata(benchmark, task),
        "steps": step_records,
    }


def _append_jsonl(
    path: Path,
    record: Dict[str, Any],
) -> None:
    """Append one completed trajectory immediately.

    Writing after every episode prevents losing all results if a long run is
    interrupted halfway through.
    """
    with path.open("a", encoding="utf-8") as f:
        f.write(
            json.dumps(record, ensure_ascii=False) + "\n"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an LLM-agent experiment, optionally with a critic.")

    parser.add_argument(
        "--config",
        required=True,
        help="Path to the experiment YAML config.",
    )

    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional output-directory override.",
    )

    parser.add_argument(
        "--max-tasks",
        type=int,
        default=None,
        help="Optional task limit for small/debug runs.",
    )

    parser.add_argument(
        "--server-address",
        default=None,
        help="Optional SGLang server address override.",
    )

    parser.add_argument(
        "--model-name",
        default=None,
        help="Optional model-name/path override for the actor config.",
    )

    parser.add_argument(
        "--critic-checkpoint",
        default=None,
        help="Optional QNet critic checkpoint override.",
    )
    parser.add_argument(
        "--critic-tokenizer-path",
        default=None,
        help="Optional tokenizer path used by the QNet critic.",
    )
    parser.add_argument(
        "--num-candidates",
        type=int,
        default=None,
        help="Number of actor candidates ranked by the critic at every step.",
    )

    parser.add_argument(
        "--human",
        action="store_true",
        help=(
            "Use an interactive human actor instead of the configured LLM actor"
        ),
    )
    parser.add_argument(
        "--task-seed",
        type=int,
        default=42,
        help=(
            "Random seed used to select tasks in human mode. Ignored for regular LLM experiments."
        ),
    )

    parser.add_argument(
        "--num-trajectories",
        type=int,
        default=None,
        help="Optional number of independent trajectories per task.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing trajectories.jsonl.",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s | "
            "%(levelname)s | "
            "%(message)s"
        ),
    )

    # ------------------------------------------------------------------
    # Load configs.
    # ------------------------------------------------------------------

    experiment_config = _load_yaml(_repo_path(args.config))

    agent_config_path = (
        REPO_ROOT / "configs/agents/human.yaml" if args.human 
        else _repo_path(experiment_config["agent_config"])
    )
    agent_config = _load_yaml(agent_config_path)

    env_config = _load_yaml(_repo_path(experiment_config["env_config"]))

    critic_config = None
    critic_type = None
    critic_benchmark = None

    critic_config_path = experiment_config.get("critic_config")
    if critic_config_path is not None:
        if args.human:
            raise ValueError("Critic mode is not supported together with --human.")

        critic_config = _load_yaml(_repo_path(critic_config_path))

        if args.critic_checkpoint:
            critic_config["checkpoint_path"] = args.critic_checkpoint
        if args.critic_tokenizer_path:
            critic_config["tokenizer_path"] = args.critic_tokenizer_path

        critic_type = str(critic_config["type"]).strip().lower()

        critic_benchmark = str(critic_config.get("benchmark", "")).strip().lower()

        if not critic_benchmark:
            raise ValueError(
                "Critic config must explicitly define the benchmark the critic was trained for."
            )

        if critic_type not in CRITIC_REGISTRY:
            raise ValueError(f"Unsupported critic type: {critic_type}")

    benchmark = str(env_config["name"]).strip().lower()

    if benchmark not in TASK_REGISTRY:
        raise ValueError(f"Unsupported benchmark: {benchmark}")

    if critic_config is not None and critic_benchmark != benchmark:
        raise ValueError(
            f"Critic/benchmark mismatch: critic is configured for {critic_benchmark!r}, "
            f"but the experiment uses {benchmark!r}. Use a critic trained specifically for this benchmark "
            "and its current ReAct prompt protocol."
        )

    agent_type = str(agent_config["type"]).strip().lower()

    if agent_type not in AGENT_REGISTRY:
        raise ValueError(
            f"Unsupported agent type: {agent_type}"
        )

    if not args.human:
        if args.server_address:
            agent_config["server_address"] = args.server_address
        if args.model_name:
            agent_config["model_name"] = args.model_name

    # ------------------------------------------------------------------
    # Dataset/run settings.
    # ------------------------------------------------------------------

    split = str(env_config.get("split", "test"))

    part_num = int(env_config.get("part_num", 1))

    part_idx = int(env_config.get("part_idx", -1))

    max_tasks = (
        args.max_tasks if args.max_tasks is not None else experiment_config.get("max_tasks")
    )
    # Human exploration is intentionally small by default.
    # --max-tasks can still be used to request, for example, 2 instead of 3 tasks.
    if args.human and max_tasks is None:
        max_tasks = 3

    if max_tasks is not None:
        max_tasks = int(max_tasks)

        if max_tasks <= 0:
            raise ValueError("max_tasks must be positive")

    if args.human and args.num_trajectories is None:
        # Do not make a person solve the same task several times just because
        # the corresponding LLM experiment uses best-of-N trajectories.
        num_trajectories = 1
    else:
        num_trajectories = int(
            args.num_trajectories
            if args.num_trajectories is not None else experiment_config.get("num_trajectories", 1)
        )

    if num_trajectories <= 0:
        raise ValueError("num_trajectories must be positive")

    num_candidates = int(
        args.num_candidates
        if args.num_candidates is not None
        else experiment_config.get("num_candidates", 1)
    )

    if num_candidates <= 0:
        raise ValueError("num_candidates must be positive")

    if critic_config is None and num_candidates != 1:
        raise ValueError("num_candidates > 1 requires a configured critic.")

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    output_dir = _repo_path(args.output_dir or experiment_config["output_dir"])

    output_dir.mkdir(parents=True, exist_ok=True)

    trajectories_path = output_dir / "trajectories.jsonl"
    metadata_path = output_dir / "run_metadata.json"

    if trajectories_path.exists():
        if args.overwrite:
            trajectories_path.unlink()

        else:
            raise FileExistsError(
                f"{trajectories_path} already exists. "
                "Use another output directory or pass --overwrite."
            )

    # ------------------------------------------------------------------
    # Initialize benchmark and actor.
    # ------------------------------------------------------------------

    task_cls = TASK_REGISTRY[benchmark]

    tasks, n_tasks = task_cls.load_tasks(split=split, part_num=part_num, part_idx=part_idx)

    logger.info(
        "Loaded benchmark=%s split=%s tasks=%d",
        benchmark, split, n_tasks,
    )

    if args.human:
        # Human mode is intended for inspecting a few representative tasks rather
        # than walking through the dataset from the beginning
        available_tasks = list(tasks)

        num_tasks_to_run = min(n_tasks, max_tasks)

        rng = random.Random(args.task_seed)
        selected_tasks = rng.sample(available_tasks, k=num_tasks_to_run)

        tasks = iter(selected_tasks)

        logger.info(
            "Human mode selected %d random task(s) with seed=%d: %s",
            num_tasks_to_run, args.task_seed,
            [task.task_id for task in selected_tasks],
        )
    else:
        # n_tasks is the number of tasks available after dataset slicing.
        # max_tasks additionally limits how many of them this concrete run executes.
        num_tasks_to_run = n_tasks if max_tasks is None else min(n_tasks, max_tasks)

    logger.info(
        "Running %d/%d available task(s), %d trajectory/trajectories per task",
        num_tasks_to_run, n_tasks, num_trajectories,
    )

    runtime = _prepare_benchmark_runtime(benchmark, env_config)

    agent_cls = AGENT_REGISTRY[agent_type]
    agent = agent_cls(agent_config)

    critic = None
    if critic_config is not None:
        critic_cls = CRITIC_REGISTRY[critic_type]
        critic = critic_cls(critic_config)

        logger.info(
            "Loaded critic type=%s checkpoint=%s candidates=%d",
            critic_type, critic_config["checkpoint_path"], num_candidates,
        )

    experiment_name = (
        f"human_{benchmark}"
        if args.human else str(experiment_config["name"])
    )

    processed_tasks = 0
    total_episodes = 0

    run_metadata = {
        "experiment_name": experiment_name,
        "actor_type": agent_type,
        "benchmark": benchmark,
        "split": split,
        "model_name": agent_config.get("model_name"),
        "num_trajectories": num_trajectories,
        "max_tasks": max_tasks,
        "task_seed": args.task_seed if args.human else None,
        "status": "running",

        "critic_type": critic_type,
        "num_candidates": num_candidates,
        "critic_benchmark": critic_benchmark,
    }

    if critic_config is not None:
        run_metadata["critic_checkpoint"] = critic_config["checkpoint_path"]

    # One progress-bar iteration corresponds to one complete benchmark task.
    progress_bar = tqdm(total=num_tasks_to_run, dynamic_ncols=True, disable=args.human)

    try:
        # --------------------------------------------------------------
        # Benchmark-agnostic experiment loop.
        # --------------------------------------------------------------

        for task in tasks:
            if processed_tasks >= num_tasks_to_run:
                break

            processed_tasks += 1

            for attempt_id in range(num_trajectories):
                env = _build_env(
                    benchmark=benchmark,
                    task=task,
                    env_config=env_config,
                    runtime=runtime,
                )

                if critic is None:
                    trajectory = _run_episode(
                        agent=agent,
                        env=env,
                        task=task,
                        benchmark=benchmark,
                        split=split,
                        attempt_id=attempt_id,
                        experiment_name=experiment_name,
                    )
                else:
                    trajectory = _run_episode_with_critic(
                        agent=agent,
                        critic=critic,
                        env=env,
                        task=task,
                        benchmark=benchmark,
                        split=split,
                        attempt_id=attempt_id,
                        experiment_name=experiment_name,
                        num_candidates=num_candidates,
                    )

                _append_jsonl(trajectories_path, trajectory)

                total_episodes += 1

                logger.info(
                    "task=%s attempt=%d success=%s reward=%s steps=%d reason=%s",
                    task.task_id, attempt_id, trajectory["success"], trajectory["reward"],
                    trajectory["num_steps"], trajectory["terminate_reason"],
                )
            
            # Update only after every trajectory for this task has finished.
            progress_bar.update(1)
    except Exception as exc:
        run_metadata["status"] = "failed"
        run_metadata["failure"] = repr(exc)
        raise

    else:
        run_metadata["status"] = "completed"

    finally:
        progress_bar.close()

        run_metadata["processed_tasks"] = processed_tasks
        run_metadata["total_episodes"] = total_episodes

        with metadata_path.open("w", encoding="utf-8") as f:
            json.dump(run_metadata, f, ensure_ascii=False, indent=2)

        if critic is not None:
            critic.close()

        agent.close()

    logger.info(
        "Finished %d episode(s) over %d task(s). Results: %s",
        total_episodes, processed_tasks, trajectories_path,
    )


if __name__ == "__main__":
    main()
