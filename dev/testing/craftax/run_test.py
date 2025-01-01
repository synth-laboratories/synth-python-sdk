# import json
# from pathlib import Path
# from typing import List, Literal

# import pytest
# import yaml
# from craftaxlm import CraftaxACI, CraftaxClassicACI
# from tqdm.asyncio import tqdm_asyncio
# from zyk import LM

# from dev.testing.craftax.agent import SimpleReActLanguageAgent
# from synth_sdk.tracing.abstractions import (
#     AgentComputeStep,
#     ArbitraryInputs,
#     ArbitraryOutputs,
#     Dataset,
#     EnvironmentComputeStep,
#     Event,
#     EventPartitionElement,
#     MessageInputs,
#     MessageOutputs,
#     RewardSignal,
#     TrainingQuestion,
# )
# from synth_sdk.tracing.upload import upload

# # Load config
# config_path = Path("dev/testing/craftax/react.yaml")
# with open(config_path, "r") as f:
#     react_config = yaml.safe_load(f)


# async def generate_single_episode(
#     agent: SimpleReActLanguageAgent,
#     mode: Literal["classic", "full"],
#     seed: int,
#     max_steps: int,
#     output_dir: Path = None,
# ):
#     """Generate a single episode"""
#     # Initialize environment
#     if mode == "classic":
#         env = CraftaxClassicACI(seed=seed, verbose=False)
#     else:
#         env = CraftaxACI(seed=seed, verbose=False)

#     # Store episode data
#     episode_data = {"mode": mode, "seed": seed, "steps": [], "final_achievements": None}

#     # Initial observation
#     initial_obs = {"state": env.starting_obs}
#     await agent.add_observations([initial_obs])
#     episode_data["steps"].append(
#         {
#             "observation": initial_obs,
#             "action": None,
#             "reward": 0.0,
#             "done": False,
#             "achievements": [],
#         }
#     )

#     # Run episode
#     for step in range(max_steps):
#         actions = await agent.get_actions()
#         step_infos = []
#         for action in actions:
#             step_info = env._step(env.map_action_string_to_int(action))
#             step_infos.append(step_info)
#             episode_data["steps"].append(
#                 {
#                     "observation": step_info,
#                     "action": action,
#                     "reward": step_info["reward"],
#                     "done": step_info["done"],
#                     "achievements": env.achievement_deltas[-1],
#                 }
#             )
#             if step_info["done"]:
#                 break

#         await agent.add_observations(step_infos)
#         if step_info["done"]:
#             break

#     # Store final achievements
#     raw_achievements = env.terminate()
#     episode_data["final_achievements"] = {
#         k: bool(v) for k, v in raw_achievements.items()
#     }
#     return episode_data


# async def generate_episodes(
#     agent: SimpleReActLanguageAgent,
#     mode: Literal["classic", "full"] = "classic",
#     seeds: List[int] = [0],
#     max_steps: int = 200,
#     output_dir: Path = None,
# ):
#     """Generate multiple episodes using the provided agent"""
#     # Create tasks for all episodes
#     tasks = [
#         generate_single_episode(agent, mode, seed, max_steps, output_dir)
#         for seed in seeds
#     ]

#     # Run episodes concurrently with progress bar
#     all_episodes_data = await tqdm_asyncio.gather(
#         *tasks, desc=f"Running {mode} episodes"
#     )

#     # Process results
#     total_achievements_list = []
#     for episode_data in all_episodes_data:
#         total_achievements = sum(
#             1 for v in episode_data["final_achievements"].values() if v
#         )
#         total_achievements_list.append(total_achievements)
#         print(
#             f"Final achievements for seed {episode_data['seed']}: {episode_data['final_achievements']} (Total: {total_achievements})"
#         )

#         # Save episode data if output directory provided
#         if output_dir:
#             output_dir = Path(output_dir)
#             output_dir.mkdir(parents=True, exist_ok=True)
#             output_file = output_dir / f"episode_{mode}_{episode_data['seed']}.json"
#             with open(output_file, "w") as f:
#                 json.dump(episode_data["final_achievements"], f, indent=2)

#     # Calculate and upload results
#     avg_achievements = sum(total_achievements_list) / len(total_achievements_list)
#     print(
#         f"\nAverage achievements across {len(seeds)} episodes: {avg_achievements:.2f}"
#     )

#     # Upload results with average reward
#     results = upload(
#         dataset=Dataset(
#             questions=[
#                 TrainingQuestion(
#                     id="default",
#                     intent=" ",
#                     criteria="Got as many achievements as possible across multiple episodes",
#                 )
#             ],
#             reward_signals=[
#                 RewardSignal(
#                     question_id="default",
#                     system_instance_id=agent.system_instance_id,
#                     reward=avg_achievements,
#                     annotation=f"Average achievements across {len(seeds)} episodes: {avg_achievements:.2f}",
#                 )
#             ],
#         )
#     )
#     print("Results: ", results)
#     print("Uploaded")
#     assert_compute_inputs_not_empty(results)
#     return results, all_episodes_data


# def assert_compute_inputs_not_empty(results):
#     """
#     Assert that compute inputs in traces are not empty.

#     Parameters:
#         results (tuple): A tuple containing response, questions_json, reward_signals_json, traces_json.

#     Raises:
#         AssertionError: If no compute steps are found in the traces or if compute inputs are empty.
#     """
#     response, questions_json, reward_signals_json, traces_json = results
#     found_compute_steps = False

#     # Iterate over each trace in traces_json
#     for trace in traces_json:
#         if "partition" not in trace:
#             continue

#         for partition in trace["partition"]:
#             if "events" not in partition:
#                 continue

#             for event_idx, event in enumerate(partition["events"]):
#                 # Check for agent_compute_steps
#                 if "agent_compute_steps" in event:
#                     found_compute_steps = True
#                     for step in event["agent_compute_steps"]:
#                         if not step.get("compute_input") and event_idx > 0:
#                             print("\nFound empty compute_input. Dumping all events:")
#                             for idx, e in enumerate(partition["events"]):
#                                 print(f"\nEvent {idx}:")
#                                 print(json.dumps(e, indent=2))
#                             assert False, f"Empty compute_input in agent step (event {event_idx}): {step}"
#                         if isinstance(step["compute_input"], list) and event_idx > 0:
#                             assert (
#                                 len(step["compute_input"]) > 0
#                             ), f"Agent compute_input is an empty list (event {event_idx})"

#                 # Check for environment_compute_steps
#                 if "environment_compute_steps" in event:
#                     found_compute_steps = True
#                     for step in event["environment_compute_steps"]:
#                         if not step.get("compute_input") and event_idx > 0:
#                             print("\nFound empty compute_input. Dumping all events:")
#                             for idx, e in enumerate(partition["events"]):
#                                 print(f"\nEvent {idx}:")
#                                 print(json.dumps(e, indent=2))
#                             assert False, f"Empty compute_input in environment step (event {event_idx}): {step}"
#                         if isinstance(step["compute_input"], list) and event_idx > 0:
#                             assert (
#                                 len(step["compute_input"]) > 0
#                             ), f"Environment compute_input is an empty list (event {event_idx})"

#     assert found_compute_steps, "No compute steps found in traces"
#     return True


# def validate_steps(steps):
#     """
#     Validate that compute steps have non-empty inputs and outputs.

#     Args:
#         steps: Either an EventPartitionElement object or a dictionary containing events

#     Returns:
#         bool: True if all compute steps are valid, False otherwise
#     """
#     # Handle both object and dict representations
#     if isinstance(steps, EventPartitionElement):
#         events = steps.events
#     else:
#         events = steps.get("events", [])

#     for event_idx, event in enumerate(events):
#         # Check agent compute steps
#         agent_steps = (
#             event.agent_compute_steps
#             if isinstance(event, Event)
#             else event.get("agent_compute_steps", [])
#         )

#         env_steps = (
#             event.environment_compute_steps
#             if isinstance(event, Event)
#             else event.get("environment_compute_steps", [])
#         )

#         # Validate both agent and environment steps
#         for compute_step in agent_steps + env_steps:
#             # Handle both object and dict representations
#             if isinstance(compute_step, (AgentComputeStep, EnvironmentComputeStep)):
#                 compute_input = compute_step.compute_input
#                 compute_output = compute_step.compute_output
#             else:
#                 compute_input = compute_step.get("compute_input")
#                 compute_output = compute_step.get("compute_output")

#             # Skip empty input validation for first event
#             if event_idx == 0:
#                 if not compute_output:
#                     return False
#             else:
#                 # Basic validation for non-first events
#                 if not compute_input or not compute_output:
#                     return False

#                 # Validate inputs (only for non-first events)
#                 for input_item in compute_input:
#                     if isinstance(input_item, (MessageInputs, ArbitraryInputs)):
#                         if not input_item.inputs and not getattr(
#                             input_item, "messages", None
#                         ):
#                             return False
#                     elif isinstance(input_item, dict):
#                         if not input_item.get("inputs") and not input_item.get(
#                             "messages"
#                         ):
#                             return False
#                     else:
#                         return False

#             # Validate outputs (for all events)
#             for output_item in compute_output:
#                 if isinstance(output_item, (MessageOutputs, ArbitraryOutputs)):
#                     if not output_item.outputs and not getattr(
#                         output_item, "messages", None
#                     ):
#                         return False
#                 elif isinstance(output_item, dict):
#                     if not output_item.get("outputs") and not output_item.get(
#                         "messages"
#                     ):
#                         return False
#                 else:
#                     return False

#     return True


# @pytest.mark.asyncio
# async def test_craftax_episode():
#     """Test generating multiple Craftax episodes"""
#     # Use config values
#     max_steps = react_config["agent"]["max_agent_steps"]
#     model_name = react_config["language_model"]["name"]

#     # Initialize LLM
#     lm = LM(
#         model_name=model_name,
#         formatting_model_name="gpt-4o-mini",
#         temperature=react_config["language_model"]["temperature"],
#         synth_logging=True,
#     )

#     # Setup test output directory
#     output_dir = Path("tests/iteration/craftax/generate_data/records")

#     # Create agent
#     agent = SimpleReActLanguageAgent(
#         lm=lm,
#         mode=react_config["agent"]["mode"],
#         config={
#             "max_history": react_config["agent"]["max_history"],
#             "max_agent_steps": react_config["agent"]["max_agent_steps"],
#         },
#     )

#     # Run episodes with multiple seeds
#     seeds = range(0, 3)  # You can modify this list to run more or fewer episodes
#     episodes_data = await generate_episodes(
#         agent=agent,
#         mode="classic",
#         seeds=seeds,
#         output_dir=output_dir,
#         max_steps=max_steps,
#     )

#     # Add assertions to verify the episodes
#     assert episodes_data is not None
#     assert len(episodes_data) == len(seeds)
#     for episode_data in episodes_data:
#         assert "final_achievements" in episode_data
#         assert isinstance(episode_data["final_achievements"], dict)
#         assert len(episode_data["steps"]) > 0
#         assert all(
#             isinstance(step["reward"], (int, float)) for step in episode_data["steps"]
#         )


# @pytest.mark.asyncio
# async def test_validate_steps():
#     """Test that all steps in traces have valid non-empty inputs and outputs"""
#     # Use config values
#     max_steps = react_config["agent"]["max_agent_steps"]
#     model_name = react_config["language_model"]["name"]

#     # Initialize LLM
#     lm = LM(
#         model_name=model_name,
#         formatting_model_name="gpt-4o-mini",
#         temperature=react_config["language_model"]["temperature"],
#         synth_logging=True,
#     )

#     # Create agent
#     agent = SimpleReActLanguageAgent(
#         lm=lm,
#         mode=react_config["agent"]["mode"],
#         config={
#             "max_history": react_config["agent"]["max_history"],
#             "max_agent_steps": react_config["agent"]["max_agent_steps"],
#         },
#     )

#     # Run a single episode to generate steps
#     results, episodes_data = await generate_episodes(
#         agent=agent,
#         mode="classic",
#         seeds=[0],  # Just test with one seed for speed
#         max_steps=max_steps,
#     )

#     # Validate each episode's steps
#     for episode in episodes_data:
#         assert len(episode["steps"]) > 0, "Episode has no steps"
#         for step in episode["steps"]:
#             assert "observation" in step, "Step missing observation"
#             assert "action" in step, "Step missing action"
#             assert "reward" in step, "Step missing reward"
#             assert "done" in step, "Step missing done flag"
#             assert "achievements" in step, "Step missing achievements"

#     # Extract traces from results
#     response, questions_json, reward_signals_json, traces_json = results

#     # Validate each trace
#     for trace in traces_json:
#         assert "partition" in trace, "Trace missing partition"
#         assert trace["partition"], "Empty partition in trace"

#         # Validate all steps in each partition
#         valid_steps = all(validate_steps(partition) for partition in trace["partition"])
#         assert valid_steps, "Invalid steps found in trace partitions"

#         # Additional validation for each partition
#         for partition in trace["partition"]:
#             assert "events" in partition, "Partition missing events"
#             for event in partition["events"]:
#                 # Validate agent compute steps
#                 if "agent_compute_steps" in event:
#                     for step in event["agent_compute_steps"]:
#                         assert step.get("compute_input"), "Empty agent compute input"
#                         assert step.get("compute_output"), "Empty agent compute output"

#                 # Validate environment compute steps
#                 if "environment_compute_steps" in event:
#                     for step in event["environment_compute_steps"]:
#                         assert step.get(
#                             "compute_input"
#                         ), "Empty environment compute input"
#                         assert step.get(
#                             "compute_output"
#                         ), "Empty environment compute output"


# if __name__ == "__main__":
#     import asyncio

#     # asyncio.run(test_craftax_episode())
#     #


#     #This is a hard test to run because ZYK import Synth-SDK
#     asyncio.run(test_validate_steps())
