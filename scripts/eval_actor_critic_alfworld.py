"""
Evaluate an ALFWorld actor AND critic on expert trajectories using teacher forcing.

Actor and Critic are evaluated INDEPENDENTLY at each step:
- Actor generates one action -> compared to expert action (actor_accuracy).
- Critic scores ALL admissible actions -> best action is compared to expert (critic_accuracy).
- Orchestrator (Best-of-2): Succeeds if EITHER actor OR critic matches the expert action.
- Environment always executes the EXPERT action (teacher forcing) to preserve trajectory validity.
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
from agent_eval.critics import CRITIC_REGISTRY
from agent_eval.envs import AlfWorldEnv
from agent_eval.envs.react import parse_react_action
from agent_eval.paths import ALFWORLD_DATA_DIR
from agent_eval.tasks import AlfWorldTask

DEFAULT_EXPERT_DATA = REPO_ROOT / "data" / "expert" / "alfworld_sft.json"
INVALID_ACTION = "invalid_action"
AGENT_ERROR_ACTION = "agent_error"

logger = logging.getLogger("alfworld_expert_eval_v4")


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
    if not parsed_action or parsed_action in {INVALID_ACTION, AGENT_ERROR_ACTION}:
        return 0
    if not expert_action:
        return 0
    return int(_normalize_action(parsed_action) == _normalize_action(expert_action))


def parse_thought(output: str) -> str:
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
    thought = parse_thought(output)
    return f"<think>{thought}</think>\n<action>{action}</action>"


def get_expert_steps(trajectory: dict) -> List[Dict[str, Any]]:
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

        if steps and state.lower().startswith("observation:"):
            state = state[len("Observation:"):].strip()

        try:
            action = parse_react_action(expert_output)
        except ValueError:
            action = ""

        steps.append({
            "step_id": len(steps),
            "state": state,
            "previous_actions": previous_actions.copy(),
            "thought": parse_thought(expert_output),
            "action": action,
            "react_output": to_react_format(expert_output, action),
        })
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


def _score_all_admissible(critic, messages, admissible_actions):
    """Score every admissible action independently. Returns list of (action, score)."""
    scored = []
    for action in admissible_actions:
        try:
            score = critic.score_candidate(
                actor_messages=messages,
                raw_action=action,
            )
            scored.append((action, score))
        except Exception as exc:
            logger.warning("Critic failed to score action %r: %s", action, exc)
            scored.append((action, float("-inf")))
    return scored


def _compute_expert_rank(scored_actions, expert_action):
    """Return 1-based rank of expert action in descending-score order. None if not found."""
    normalized_expert = _normalize_action(expert_action)
    sorted_actions = sorted(scored_actions, key=lambda x: x[1], reverse=True)
    for rank, (action, _) in enumerate(sorted_actions, start=1):
        if _normalize_action(action) == normalized_expert:
            return rank
    return None


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate ALFWorld actor AND critic on expert trajectories (independent, teacher-forced)."
    )
    parser.add_argument("--actor-config", required=True, help="Path to the actor YAML config.")
    parser.add_argument("--critic-config", required=True, help="Path to the critic YAML config.")
    parser.add_argument("--expert-data", default=str(DEFAULT_EXPERT_DATA), help="Path to expert trajectories JSON.")
    parser.add_argument("--output-dir", required=True, help="Directory for outputs.")
    parser.add_argument("--max-trajectories", type=int, default=None)
    parser.add_argument("--max-env-steps", type=int, default=50)
    parser.add_argument("--history-length", type=int, default=0)
    parser.add_argument("--server-address", default=None)
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--critic-checkpoint", default=None)
    parser.add_argument("--critic-tokenizer-path", default=None)
    parser.add_argument("--omit-messages", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args = parse_args()

    # --- Load actor ---
    agent_config = _load_yaml(_repo_path(args.actor_config))
    if args.server_address:
        agent_config["server_address"] = args.server_address
    if args.model_name:
        agent_config["model_name"] = args.model_name
    agent_type = str(agent_config["type"]).strip().lower()
    if agent_type not in AGENT_REGISTRY:
        raise ValueError(f"Unsupported agent type: {agent_type}")
    agent = AGENT_REGISTRY[agent_type](agent_config)

    # --- Load critic ---
    critic_config = _load_yaml(_repo_path(args.critic_config))
    if args.critic_checkpoint:
        critic_config["checkpoint_path"] = args.critic_checkpoint
    if args.critic_tokenizer_path:
        critic_config["tokenizer_path"] = args.critic_tokenizer_path
    critic_type = str(critic_config["type"]).strip().lower()
    if critic_type not in CRITIC_REGISTRY:
        raise ValueError(f"Unsupported critic type: {critic_type}")
    critic = CRITIC_REGISTRY[critic_type](critic_config)

    # --- Load expert data ---
    expert_path = _repo_path(args.expert_data)
    if not expert_path.exists():
        raise FileNotFoundError(f"Expert trajectories not found: {expert_path}")
    with expert_path.open("r", encoding="utf-8") as f:
        trajectories = json.load(f)
    if args.max_trajectories is not None:
        trajectories = trajectories[: args.max_trajectories]

    # --- Output setup ---
    output_dir = _repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    trajectories_path = output_dir / "trajectories.jsonl"
    metrics_path = output_dir / "metrics.json"
    metadata_path = output_dir / "run_metadata.json"

    if trajectories_path.exists():
        if args.overwrite:
            trajectories_path.unlink()
        else:
            raise FileExistsError(f"{trajectories_path} already exists. Use --overwrite.")

    run_metadata = {
        "experiment_name": "alfworld_actor_and_critic_on_expert_trajectories",
        "benchmark": "alfworld",
        "mode": "independent_evaluation_teacher_forcing",
        "actor_type": agent_type,
        "critic_type": critic_type,
        "model_name": agent_config.get("model_name"),
        "critic_checkpoint": critic_config.get("checkpoint_path"),
        "expert_data": str(expert_path),
        "max_trajectories": args.max_trajectories,
        "history_length": args.history_length,
        "status": "running",
    }

    # --- Aggregate counters ---
    processed = 0
    skipped = 0
    total_eval_steps = 0
    total_actor_matched = 0
    total_critic_matched = 0
    total_best_of_2_matched = 0
    total_format_errors = 0
    total_admissible_steps = 0
    per_traj_actor_acc: List[float] = []
    per_traj_critic_acc: List[float] = []
    per_traj_best_of_2_acc: List[float] = []
    exact_match_actor = 0
    exact_match_critic = 0
    exact_match_best_of_2 = 0
    trajs_with_steps = 0
    expert_ranks: List[int] = []

    try:
        progress = tqdm(trajectories, desc="Processing trajectories", dynamic_ncols=True)
        for idx, trajectory in enumerate(progress):
            traj_id = trajectory.get("id", idx)

            # Parse expert steps
            try:
                expert_steps = get_expert_steps(trajectory)
            except Exception:
                logger.exception("Failed to parse trajectory id=%s", traj_id)
                skipped += 1
                continue

            if not expert_steps or any(not s["action"] for s in expert_steps):
                logger.warning("Skipping trajectory id=%s: empty or invalid expert steps", traj_id)
                skipped += 1
                continue

            game_file_raw = trajectory.get("game_file")
            if not game_file_raw:
                logger.warning("Skipping trajectory id=%s: no game_file", traj_id)
                skipped += 1
                continue
            game_file = _resolve_game_file(str(game_file_raw))
            if not game_file.exists():
                logger.warning("Skipping trajectory id=%s: game file not found: %s", traj_id, game_file)
                skipped += 1
                continue

            initial_obs = expert_steps[0]["state"]
            if initial_obs.lower().startswith("observation:"):
                initial_obs = initial_obs[len("Observation:"):].strip()

            try:
                task = AlfWorldTask(task_id=idx, game_file=str(game_file), env=None, obs=initial_obs, split="expert")
                max_steps = max(args.max_env_steps, len(expert_steps) + 5)
                env = AlfWorldEnv(task=task, max_steps=max_steps)
                env.history_length = args.history_length
            except Exception:
                logger.exception("Failed to create env for trajectory id=%s", traj_id)
                skipped += 1
                continue

            try:
                _, state = env.reset()
                initial_observation = env.get_current_observation()
                task_text = env.get_task_text()
                step_records = []
                actor_matched = 0
                critic_matched = 0
                best_of_2_matched = 0
                evaluated_steps = 0

                for expert_step in expert_steps:
                    if state.finished:
                        break

                    messages = env.build_agent_messages()
                    observation_before = env.get_current_observation()
                    admissible_actions = env.get_admissible_commands()

                    # ===== ACTOR: generate one action =====
                    raw_output = ""
                    parsed_action = AGENT_ERROR_ACTION
                    actor_error = None
                    try:
                        raw_output = agent.act(messages)
                        try:
                            parsed_action = parse_react_action(raw_output)
                        except ValueError:
                            parsed_action = INVALID_ACTION
                    except Exception as exc:
                        actor_error = repr(exc)
                        logger.exception("Actor failed traj=%s step=%s", traj_id, expert_step["step_id"])

                    actor_score = _action_match(parsed_action, expert_step["action"])

                    # ===== CRITIC: score ALL admissible actions independently =====
                    critic_error = None
                    scored_actions = []
                    critic_best_action = ""
                    critic_best_score = float("-inf")
                    expert_rank = None
                    expert_q_value = None

                    try:
                        scored_actions = _score_all_admissible(critic, messages, admissible_actions)
                        if scored_actions:
                            critic_best_action, critic_best_score = max(scored_actions, key=lambda x: x[1])
                        expert_rank = _compute_expert_rank(scored_actions, expert_step["action"])
                        # Find expert Q-value
                        norm_expert = _normalize_action(expert_step["action"])
                        for act, sc in scored_actions:
                            if _normalize_action(act) == norm_expert:
                                expert_q_value = sc
                                break
                    except Exception as exc:
                        critic_error = repr(exc)
                        logger.exception("Critic failed traj=%s step=%s", traj_id, expert_step["step_id"])

                    critic_score_match = _action_match(critic_best_action, expert_step["action"])
                    best_of_2_score = int(bool(actor_score or critic_score_match))

                    # ===== Execute EXPERT action (teacher forcing) =====
                    try:
                        observation_after, state = env.step(expert_step["react_output"])
                        #observation_after, state = env.step(expert_step["action"])
                    except Exception as exc:
                        logger.exception("Expert step failed traj=%s step=%s", traj_id, expert_step["step_id"])
                        step_records.append({
                            "step_id": state.steps + 1,
                            "expert_step_id": expert_step["step_id"],
                            "observation": observation_before,
                            "agent_messages": None if args.omit_messages else copy.deepcopy(messages),
                            "admissible_actions": list(admissible_actions),
                            "raw_agent_output": raw_output,
                            "parsed_action": parsed_action,
                            "expert_action": expert_step["action"],
                            "expert_thought": expert_step.get("thought", ""),
                            "critic_best_action": critic_best_action,
                            "critic_best_score": critic_best_score,
                            "expert_rank": expert_rank,
                            "expert_q_value": expert_q_value,
                            "num_admissible": len(admissible_actions),
                            "next_observation": None,
                            "reward": state.reward,
                            "actor_score": actor_score,
                            "critic_score_match": critic_score_match,
                            "best_of_2_score": best_of_2_score,
                            "actor_error": actor_error,
                            "critic_error": critic_error,
                            "expert_step_error": repr(exc),
                            "done": True,
                        })
                        state.finished = True
                        state.success = False
                        state.terminate_reason = "expert_step_error"
                        state.error = repr(exc)
                        break

                    norm_admissible = {_normalize_action(a) for a in admissible_actions}
                    is_admissible = (
                        parsed_action not in {INVALID_ACTION, AGENT_ERROR_ACTION}
                        and _normalize_action(parsed_action) in norm_admissible
                    )

                    step_records.append({
                        "step_id": state.steps,
                        "expert_step_id": expert_step["step_id"],
                        "observation": observation_before,
                        "agent_messages": None if args.omit_messages else copy.deepcopy(messages),
                        "admissible_actions": list(admissible_actions),
                        "raw_agent_output": raw_output,
                        "parsed_action": parsed_action,
                        "expert_action": expert_step["action"],
                        "expert_thought": expert_step.get("thought", ""),
                        "critic_best_action": critic_best_action,
                        "critic_best_score": critic_best_score,
                        "expert_rank": expert_rank,
                        "expert_q_value": expert_q_value,
                        "num_admissible": len(admissible_actions),
                        "next_observation": observation_after,
                        "reward": state.reward,
                        "actor_score": actor_score,
                        "critic_score_match": critic_score_match,
                        "best_of_2_score": best_of_2_score,
                        "actor_error": actor_error,
                        "critic_error": critic_error,
                        "is_format_error": parsed_action == INVALID_ACTION,
                        "is_admissible": is_admissible,
                        "done": state.finished,
                    })

                    evaluated_steps += 1
                    actor_matched += actor_score
                    critic_matched += critic_score_match
                    best_of_2_matched += best_of_2_score
                    if expert_rank is not None:
                        expert_ranks.append(expert_rank)

                # Per-trajectory metrics
                actor_acc = actor_matched / evaluated_steps if evaluated_steps else None
                critic_acc = critic_matched / evaluated_steps if evaluated_steps else None
                best_of_2_acc = best_of_2_matched / evaluated_steps if evaluated_steps else None

                is_actor_exact = evaluated_steps == len(expert_steps) and actor_matched == evaluated_steps
                is_critic_exact = evaluated_steps == len(expert_steps) and critic_matched == evaluated_steps
                is_best_of_2_exact = evaluated_steps == len(expert_steps) and best_of_2_matched == evaluated_steps

                if evaluated_steps > 0:
                    trajs_with_steps += 1
                    if is_actor_exact:
                        exact_match_actor += 1
                    if is_critic_exact:
                        exact_match_critic += 1
                    if is_best_of_2_exact:
                        exact_match_best_of_2 += 1

                format_errors = sum(1 for r in step_records if r.get("is_format_error"))
                admissible_count = sum(1 for r in step_records if r.get("is_admissible"))

                trajectory_record = {
                    "experiment_name": "alfworld_actor_and_critic_on_expert_trajectories",
                    "benchmark": "alfworld",
                    "split": "expert_trajectories",
                    "trajectory_id": traj_id,
                    "task_id": idx,
                    "attempt_id": 0,
                    "task_text": task_text,
                    "initial_observation": initial_observation,
                    "success": state.success,
                    "reward": state.reward,
                    "num_steps": state.steps,
                    "expert_num_steps": len(expert_steps),
                    "evaluated_steps": evaluated_steps,
                    "actor_matched_steps": actor_matched,
                    "critic_matched_steps": critic_matched,
                    "actor_accuracy": actor_acc,
                    "critic_accuracy": critic_acc,
                    "best_of_2_accuracy": best_of_2_acc,
                    "actor_exact_match": bool(is_actor_exact),
                    "critic_exact_match": bool(is_critic_exact),
                    "best_of_2_exact_match": bool(is_best_of_2_exact),
                    "format_error_rate": format_errors / evaluated_steps if evaluated_steps else None,
                    "admissibility_rate": admissible_count / evaluated_steps if evaluated_steps else None,
                    "terminate_reason": state.terminate_reason,
                    "error": state.error,
                    "task": {"game_file": str(task.game_file), "expert_trajectory_id": traj_id},
                    "steps": step_records,
                }

                _append_jsonl(trajectories_path, trajectory_record)
                processed += 1
                total_eval_steps += evaluated_steps
                total_actor_matched += actor_matched
                total_critic_matched += critic_matched
                total_best_of_2_matched += best_of_2_matched
                total_format_errors += format_errors
                total_admissible_steps += admissible_count
                if actor_acc is not None:
                    per_traj_actor_acc.append(actor_acc)
                if critic_acc is not None:
                    per_traj_critic_acc.append(critic_acc)
                if best_of_2_acc is not None:
                    per_traj_best_of_2_acc.append(best_of_2_acc)

                print(
                    f"\ntraj={traj_id} | actor_acc={actor_acc:.3f} | critic_acc={critic_acc:.3f} | "
                    f"steps={evaluated_steps}"
                )

            except Exception:
                logger.exception("Failed to evaluate trajectory id=%s", traj_id)
                skipped += 1
            finally:
                _close_env(env)

        run_metadata["status"] = "completed"

    except Exception as exc:
        run_metadata["status"] = "failed"
        run_metadata["failure"] = repr(exc)
        raise

    finally:
        micro_actor_acc = total_actor_matched / total_eval_steps if total_eval_steps else 0.0
        macro_actor_acc = sum(per_traj_actor_acc) / len(per_traj_actor_acc) if per_traj_actor_acc else 0.0
        micro_critic_acc = total_critic_matched / total_eval_steps if total_eval_steps else 0.0
        macro_critic_acc = sum(per_traj_critic_acc) / len(per_traj_critic_acc) if per_traj_critic_acc else 0.0
        micro_best_of_2_acc = total_best_of_2_matched / total_eval_steps if total_eval_steps else 0.0  # <--- ДОБАВИТЬ
        macro_best_of_2_acc = sum(per_traj_best_of_2_acc) / len(per_traj_best_of_2_acc) if per_traj_best_of_2_acc else 0.0  # <--- ДОБАВИТЬ
        avg_expert_rank = sum(expert_ranks) / len(expert_ranks) if expert_ranks else None
        median_expert_rank = sorted(expert_ranks)[len(expert_ranks) // 2] if expert_ranks else None

        metrics = {
            "benchmark": "alfworld",
            "mode": "independent_evaluation_teacher_forcing",
            "processed_trajectories": processed,
            "skipped_trajectories": skipped,
            "total_evaluated_steps": total_eval_steps,
            "micro_actor_accuracy": micro_actor_acc,
            "macro_actor_accuracy": macro_actor_acc,
            "actor_exact_match_rate": exact_match_actor / trajs_with_steps if trajs_with_steps else 0.0,
            "micro_critic_accuracy": micro_critic_acc,
            "macro_critic_accuracy": macro_critic_acc,
            "critic_exact_match_rate": exact_match_critic / trajs_with_steps if trajs_with_steps else 0.0,
            "micro_best_of_2_accuracy": micro_best_of_2_acc,
            "macro_best_of_2_accuracy": macro_best_of_2_acc,
            "best_of_2_exact_match_rate": exact_match_best_of_2 / trajs_with_steps if trajs_with_steps else 0.0,
            "avg_expert_rank": avg_expert_rank,
            "median_expert_rank": median_expert_rank,
            "format_error_rate": total_format_errors / total_eval_steps if total_eval_steps else 0.0,
            "admissibility_rate": total_admissible_steps / total_eval_steps if total_eval_steps else 0.0,
        }

        run_metadata.update({
            "processed_trajectories": processed,
            "skipped_trajectories": skipped,
            "total_evaluated_steps": total_eval_steps,
        })

        metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        metadata_path.write_text(json.dumps(run_metadata, ensure_ascii=False, indent=2), encoding="utf-8")

        if hasattr(agent, "close"):
            agent.close()
        if hasattr(critic, "close"):
            critic.close()

        print("-" * 50)
        print(f"Processed: {processed} trajectories (of {len(trajectories)})")
        print(f"Total evaluated steps: {total_eval_steps}")
        avg_rank_str = f"{avg_expert_rank:.1f}" if avg_expert_rank is not None else "N/A"
        print(f"Expert Rank: avg={avg_rank_str}, median={median_expert_rank}")
        print(f"Format Error: {metrics['format_error_rate']:.3f}")
        print(f"Admissibility: {metrics['admissibility_rate']:.3f} \n")

        header = f"{'Method':<12} {'Micro':>10} {'Macro':>10} {'Exact':>10}"
        row_fmt = "{:<12} {:>10.3f} {:>10.3f} {:>10.3f}"
        print(header)
        print("-" * len(header))
        print(row_fmt.format("Actor", micro_actor_acc, macro_actor_acc, metrics['actor_exact_match_rate']))
        print(row_fmt.format("Critic", micro_critic_acc, macro_critic_acc, metrics['critic_exact_match_rate']))
        print(row_fmt.format("Best-of-2", micro_best_of_2_acc, macro_best_of_2_acc, metrics['best_of_2_exact_match_rate']))


if __name__ == "__main__":
    main()
