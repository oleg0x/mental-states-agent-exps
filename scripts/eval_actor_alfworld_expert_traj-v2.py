"""
Evaluate an ALFWorld actor on expert trajectories using teacher forcing.
For each expert trajectory: S0 --A0*--> S1 --A1*--> S2 --A2*--> S3
the script:
  1. asks the actor for an action in S_i;
  2. compares the actor action with expert action A_i*;
  3. executes A_i* in the environment, NOT the actor action;
  4. continues from the resulting expert state S_{i+1}.
"""

import argparse
import copy
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agent_eval.agents import AGENT_REGISTRY
from agent_eval.envs import AlfWorldEnv
from agent_eval.envs.react import parse_react_action
from agent_eval.paths import ALFWORLD_DATA_DIR
from agent_eval.tasks import AlfWorldTask

DEFAULT_EXPERT_DATA = REPO_ROOT / "data" / "expert" / "alfworld_sft.json"

INVALID_ACTION = "__invalid_action__"
AGENT_ERROR_ACTION = "__agent_error__"

logger = logging.getLogger("alfworld_expert_eval")


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open() as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a mapping in YAML file: {path}")
    return data


def _repo_path(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def _append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _close_env(env: Any) -> None:
    inner = getattr(env, "env", None)
    if inner is not None and hasattr(inner, "close"):
        try:
            inner.close()
        except Exception:
            pass


def _normalize_action(action: Any) -> str:
    return " ".join(str(action or "").strip().lower().split())


def _action_match(parsed_action: str, expert_action: str) -> int:
    if not parsed_action:
        return 0
    if parsed_action in {INVALID_ACTION, AGENT_ERROR_ACTION}:
        return 0
    if not expert_action:
        return 0
    return int(_normalize_action(parsed_action) == _normalize_action(expert_action))


def parse_thought(output: str) -> str:
    """
    Extract reasoning either from the current ReAct format:
        <think> ... </think>
        <action> ... </action>
    or from the legacy QLASS format:
        Thought: ...
        Action: ...
    """
    match = re.search(
        r"<think>\s*(.*?)\s*</think>",
        output,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if match:
        return match.group(1).strip()

    match = re.search(
        r"Thought:\s*(.*?)\s*\n\s*Action:",
        output,
        flags=re.DOTALL | re.IGNORECASE,
    )
    return match.group(1).strip() if match else ""


def to_react_format(output: str, action: Optional[str] = None) -> str:
    """
    Convert an expert output to the current ReAct format expected by
    AlfWorldEnv.step().
    """
    thought = parse_thought(output)
    return f"<think>{thought}</think>\n<action>{action}</action>"


def get_expert_steps(trajectory: dict) -> List[Dict[str, Any]]:
    """
    Convert one expert trajectory into a sequence of decision steps.
    Expected format:
        conversations[0] : human instruction
        conversations[1] : assistant OK
        conversations[2:] : alternating human observation / expert answer
    """
    messages = trajectory.get("conversations", [])
    if len(messages) <= 2:
        return []

    messages = messages[2:]
    steps: List[Dict[str, Any]] = []
    previous_actions: List[str] = []

    for i in range(0, len(messages) - 1, 2):
        state_message = messages[i]
        expert_message = messages[i + 1]

        state = str(state_message.get("value", "")).strip()
        expert_output = str(expert_message.get("value", "")).strip()

        # After the first step, observations usually have an "Observation:" prefix.
        if steps and state.lower().startswith("observation:"):
            state = state[len("Observation:"):].strip()

        try:
            action = parse_react_action(expert_output)
        except ValueError:
            action = ""

        steps.append(
            {
                "step_id": len(steps),
                # State BEFORE the expert action.
                "state": state,
                # Expert actions that led to this state.
                "previous_actions": previous_actions.copy(),
                "thought": parse_thought(expert_output),
                "action": action,
                # Convenient for env.step(...).
                "react_output": to_react_format(expert_output, action),
            }
        )

        previous_actions.append(action if action else INVALID_ACTION)

    return steps


def _resolve_game_file(game_file: str) -> Path:
    rel = Path(game_file)

    if rel.parts and rel.parts[0] == "data":
        rel = Path(*rel.parts[1:])

    candidate = ALFWORLD_DATA_DIR / rel / "game.tw-pddl"

    if candidate.is_file():
        return candidate
    return Path(game_file).expanduser()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate an ALFWorld actor on expert trajectories."
    )
    parser.add_argument("--actor-config", required=True,
        help="Path to the actor YAML config.",
    )
    parser.add_argument("--expert-data", default=str(DEFAULT_EXPERT_DATA),
        help="Path to the expert trajectories JSON file.",
    )
    parser.add_argument("--output-dir", required=True,
        help="Directory for trajectories.jsonl, metrics.json and run_metadata.json.",
    )
    parser.add_argument("--max-trajectories", type=int, default=None,
        help="Optional limit on the number of expert trajectories.",
    )
    parser.add_argument("--max-env-steps", type=int, default=50,
        help=("Base environment step limit. The real limit is max(max_env_steps, len(expert_steps) + 5)."),
    )
    parser.add_argument("--history-length", type=int, default=0,
        help="History length used by AlfWorldEnv.build_agent_messages().",
    )
    parser.add_argument("--server-address", default=None,
        help="Optional SGLang server address override.",
    )
    parser.add_argument("--model-name", default=None,
        help="Optional model-name/path override for the actor config.",
    )
    parser.add_argument("--omit-messages", action="store_true",
        help="Do not store full agent_messages in step records.",
    )
    parser.add_argument("--overwrite", action="store_true",
        help="Overwrite an existing trajectories.jsonl.",
    )
    return parser.parse_args()



def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    args = parse_args()
    agent_config = _load_yaml(_repo_path(args.actor_config))

    if args.server_address:
        agent_config["server_address"] = args.server_address
    if args.model_name:
        agent_config["model_name"] = args.model_name

    agent_type = str(agent_config["type"]).strip().lower()
    if agent_type not in AGENT_REGISTRY:
        raise ValueError(f"Unsupported agent type: {agent_type}")

    agent_cls = AGENT_REGISTRY[agent_type]
    agent = agent_cls(agent_config)

    expert_path = _repo_path(args.expert_data)
    if not expert_path.exists():
        raise FileNotFoundError(f"Expert trajectories not found: {expert_path}")

    with expert_path.open("r", encoding="utf-8") as f:
        trajectories = json.load(f)

    if args.max_trajectories is not None:
        trajectories = trajectories[: args.max_trajectories]

    output_dir = _repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    trajectories_path = output_dir / "trajectories.jsonl"
    metrics_path = output_dir / "metrics.json"
    metadata_path = output_dir / "run_metadata.json"

    if trajectories_path.exists():
        if args.overwrite:
            trajectories_path.unlink()
        else:
            raise FileExistsError(
                f"{trajectories_path} already exists. "
                "Use another output directory or pass --overwrite."
            )

    run_metadata = {
        "experiment_name": "alfworld_actor_on_expert_trajectories",
        "benchmark": "alfworld",
        "mode": "expert_trajectory_teacher_forcing",
        "actor_type": agent_type,
        "model_name": agent_config.get("model_name"),
        "expert_data": str(expert_path),
        "max_trajectories": args.max_trajectories,
        "history_length": args.history_length,
        "status": "running",
    }

    processed_trajectories = 0
    skipped_trajectories = 0
    total_evaluated_steps = 0
    total_matched_steps = 0
    total_format_errors = 0
    total_admissible_steps = 0
    expert_successes = 0
    per_trajectory_accuracies: List[float] = []
    exact_match_trajectories = 0
    trajectories_with_evaluated_steps = 0

    try:
        progress = tqdm(trajectories, desc="Processing trajectories", dynamic_ncols=True)

        for idx, trajectory in enumerate(progress):
            trajectory_id = trajectory.get("id", idx)

            # Parse expert steps.
            try:
                expert_steps = get_expert_steps(trajectory)
            except Exception:
                logger.exception("Failed to parse trajectory id=%s", trajectory_id)
                skipped_trajectories += 1
                continue

            if not expert_steps:
                logger.warning("Skipping trajectory id=%s: no expert steps", trajectory_id)
                skipped_trajectories += 1
                continue

            if any(not step["action"] for step in expert_steps):
                logger.warning(
                    "Skipping trajectory id=%s: found empty expert action",
                    trajectory_id)
                skipped_trajectories += 1
                continue

            # Identify and create ALFWorld environment.
            game_file_raw = trajectory.get("game_file")
            if not game_file_raw:
                logger.warning("Skipping trajectory id=%s: no game_file", trajectory_id)
                skipped_trajectories += 1
                continue

            game_file = _resolve_game_file(str(game_file_raw))
            if not game_file.exists():
                logger.warning(
                    "Skipping trajectory id=%s: game file not found: %s",
                    trajectory_id, game_file,
                )
                skipped_trajectories += 1
                continue

            initial_observation = expert_steps[0]["state"]
            if initial_observation.lower().startswith("observation:"):
                initial_observation = initial_observation[len("Observation:"):].strip()

            try:
                task = AlfWorldTask(
                    task_id=idx,
                    game_file=str(game_file),
                    env=None,
                    obs=initial_observation,
                    split="expert",
                )
                max_steps = max(args.max_env_steps, len(expert_steps) + 5)
                env = AlfWorldEnv(task=task, max_steps=max_steps)
                env.history_length = args.history_length
            except Exception:
                logger.exception(
                    "Failed to create ALFWorld environment for trajectory id=%s",
                    trajectory_id,
                )
                skipped_trajectories += 1
                continue

            try:
                _, state = env.reset()

                initial_observation = env.get_current_observation()
                task_text = env.get_task_text()

                step_records: List[Dict[str, Any]] = []
                matched_steps = 0
                evaluated_steps = 0

                for expert_step in expert_steps:
                    # If the environment has already finished, stop.
                    # Usually this happens after the final expert action.
                    if state.finished:
                        break

                    messages = env.build_agent_messages()
                    observation_before = env.get_current_observation()
                    admissible_actions = env.get_admissible_commands()

                    raw_output = ""
                    parsed_action = AGENT_ERROR_ACTION
                    actor_error = None

                    # 1. Ask the actor in the current expert state S_i.
                    try:
                        raw_output = agent.act(messages)
                        try:
                            parsed_action = parse_react_action(raw_output)
                        except ValueError:
                            parsed_action = INVALID_ACTION

                    except Exception as exc:
                        actor_error = repr(exc)
                        logger.exception("Actor failed on trajectory=%s step=%s",
                            trajectory_id, expert_step["step_id"])

                    # 2. Compare actor action with expert action A_i*.
                    actor_score = _action_match(parsed_action, expert_step["action"])

                    # 3. Execute the expert action, NOT the actor action.
                    try:
                        observation_after, state = env.step(expert_step["react_output"])
                    except Exception as exc:
                        logger.exception(
                            "Failed to execute expert action on trajectory=%s step=%s",
                            trajectory_id, expert_step["step_id"],
                        )
                        step_records.append({
                            "step_id": state.steps + 1,
                            "expert_step_id": expert_step["step_id"],
                            "observation": observation_before,
                            "agent_messages": None
                            if args.omit_messages
                            else copy.deepcopy(messages),
                            "admissible_actions": list(admissible_actions),
                            "raw_agent_output": raw_output,
                            "parsed_action": parsed_action,
                            "expert_action": expert_step["action"],
                            "expert_thought": expert_step.get("thought", ""),
                            "next_observation": None,
                            "reward": state.reward,
                            "actor_score": actor_score,
                            "actor_error": actor_error,
                            "expert_step_error": repr(exc),
                            "done": True,
                        })

                        state.finished = True
                        state.success = False
                        state.terminate_reason = "expert_step_error"
                        state.error = repr(exc)
                        break

                    step_records.append(
                        {
                            "step_id": state.steps,
                            "expert_step_id": expert_step["step_id"],
                            "observation": observation_before,
                            "agent_messages": None
                            if args.omit_messages
                            else copy.deepcopy(messages),
                            "admissible_actions": list(admissible_actions),
                            "raw_agent_output": raw_output,
                            "parsed_action": parsed_action,
                            "expert_action": expert_step["action"],
                            "expert_thought": expert_step.get("thought", ""),
                            "next_observation": observation_after,
                            "reward": state.reward,
                            "actor_score": actor_score,
                            "actor_error": actor_error,
                            "is_format_error": parsed_action == INVALID_ACTION,
                            "is_admissible": parsed_action in admissible_actions if parsed_action not in {INVALID_ACTION, AGENT_ERROR_ACTION} else False,
                            "done": state.finished,
                        }
                    )

                    evaluated_steps += 1
                    matched_steps += actor_score

                trajectory_accuracy = (
                    matched_steps / evaluated_steps if evaluated_steps else None
                )

                is_exact_match = (
                    evaluated_steps > 0 and matched_steps == evaluated_steps
                )
                if evaluated_steps > 0:
                    trajectories_with_evaluated_steps += 1
                    if is_exact_match:
                        exact_match_trajectories += 1

                format_errors = sum(1 for r in step_records if r.get("is_format_error"))
                admissible_count = sum(1 for r in step_records if r.get("is_admissible"))
                trajectory_format_error_rate = format_errors / evaluated_steps if evaluated_steps else None
                trajectory_admissibility_rate = admissible_count / evaluated_steps if evaluated_steps else None

                trajectory_record = {
                    "experiment_name": "alfworld_actor_on_expert_trajectories",
                    "benchmark": "alfworld",
                    "split": "expert_trajectories",
                    "trajectory_id": trajectory_id,
                    "task_id": idx,
                    "attempt_id": 0,
                    "task_text": task_text,
                    "initial_observation": initial_observation,
                    "success": state.success,
                    "reward": state.reward,
                    "num_steps": state.steps,
                    "expert_num_steps": len(expert_steps),
                    "evaluated_steps": evaluated_steps,
                    "matched_steps": matched_steps,
                    "actor_accuracy": trajectory_accuracy,
                    "exact_trajectory_match": bool(is_exact_match),
                    "format_error_rate": trajectory_format_error_rate,
                    "admissibility_rate": trajectory_admissibility_rate,
                    "terminate_reason": state.terminate_reason,
                    "error": state.error,
                    "task": {
                        "game_file": str(task.game_file),
                        "expert_trajectory_id": trajectory_id,
                    },
                    "steps": step_records,
                }

                _append_jsonl(trajectories_path, trajectory_record)

                processed_trajectories += 1
                total_evaluated_steps += evaluated_steps
                total_matched_steps += matched_steps
                total_format_errors += format_errors
                total_admissible_steps += admissible_count

                if trajectory_accuracy is not None:
                    per_trajectory_accuracies.append(trajectory_accuracy)

                if state.success:
                    expert_successes += 1

                print(f"\ntrajectory={trajectory_id}, reward={state.reward}, \
evaluated_steps={evaluated_steps}, matched_steps={matched_steps}, accuracy={trajectory_accuracy:.2f}")

            except Exception:
                logger.exception(
                    "Failed to evaluate trajectory id=%s",
                    trajectory_id,
                )
                skipped_trajectories += 1
            finally:
                _close_env(env)

        run_metadata["status"] = "completed"

    except Exception as exc:
        run_metadata["status"] = "failed"
        run_metadata["failure"] = repr(exc)
        raise

    finally:
        micro_actor_accuracy = (total_matched_steps / total_evaluated_steps)
        macro_actor_accuracy = (sum(per_trajectory_accuracies) / len(per_trajectory_accuracies))
        expert_success_rate = (expert_successes / processed_trajectories)
        exact_trajectory_match_rate = (exact_match_trajectories / processed_trajectories)
        global_format_error_rate = (total_format_errors / total_evaluated_steps)
        global_admissibility_rate = (total_admissible_steps / total_evaluated_steps)

        metrics = {
            "benchmark": "alfworld",
            "mode": "expert_trajectory_teacher_forcing",
            "processed_trajectories": processed_trajectories,
            "skipped_trajectories": skipped_trajectories,
            "total_evaluated_steps": total_evaluated_steps,
            "total_matched_steps": total_matched_steps,
            "micro_actor_accuracy": micro_actor_accuracy,
            "macro_actor_accuracy": macro_actor_accuracy,
            "expert_success_rate": expert_success_rate,
            "exact_match_trajectories": exact_match_trajectories,
            "trajectories_with_evaluated_steps": trajectories_with_evaluated_steps,
            "exact_trajectory_match_rate": exact_trajectory_match_rate,
            "format_error_rate": global_format_error_rate,
            "admissibility_rate": global_admissibility_rate,
        }

        run_metadata.update(
            {
                "processed_trajectories": processed_trajectories,
                "skipped_trajectories": skipped_trajectories,
                "total_evaluated_steps": total_evaluated_steps,
                "total_matched_steps": total_matched_steps,
                "exact_match_trajectories": exact_match_trajectories,
                "exact_trajectory_match_rate": exact_trajectory_match_rate,
            }
        )

        metrics_path.write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        metadata_path.write_text(
            json.dumps(run_metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        if hasattr(agent, "close"):
            agent.close()

        print('-'*40)
        print(f"Processed trajectories: {processed_trajectories} (of {len(trajectories)})")
        print(f"Total evaluated steps: {total_evaluated_steps}")
        print(f"Total matched steps: {total_matched_steps}")
        print(f"Micro accuracy: {micro_actor_accuracy:.2f}")
        print(f"Macro (mean) accuracy: {macro_actor_accuracy:.2f}")
        print(f"Exact match trajectory rate: {exact_trajectory_match_rate:.2f}")
        print(f"Format error rate (should be 0.0): {global_format_error_rate:.2f}")
        print(f"Admissibility rate (should be 1.0): {global_admissibility_rate:.2f}")

if __name__ == "__main__":
    main()
