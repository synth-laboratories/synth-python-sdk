# from pathlib import Path
# from typing import List, Literal

# import yaml
# from craftaxlm import CraftaxACI, CraftaxClassicACI
# from craftaxlm.src.recording import EpisodeRecorder
# from tqdm.asyncio import tqdm_asyncio
# from zyk import LM

# from dev.testing.craftax.agent import SimpleReActLanguageAgent
# from synth_sdk.tracing.abstractions import Dataset, RewardSignal, TrainingQuestion
# from synth_sdk.tracing.events.store import event_store
# from synth_sdk.tracing.upload import upload

# config_path = Path("dev/testing/craftax/react.yaml")
# with open(config_path, "r") as f:
#     react_config = yaml.safe_load(f)


# async def generate_single_episode(
#     agent: SimpleReActLanguageAgent,
#     mode: Literal["classic", "full"],
#     seed: int,
#     max_steps: int,
#     output_dir: Path = None,
#     record_video: bool = True,
# ):
#     """Generate a single episode"""
#     # Initialize environment
#     if mode == "classic":
#         env = CraftaxClassicACI(seed=seed, verbose=False)
#     else:
#         env = CraftaxACI(seed=seed, verbose=False)

#     # Initialize recorder
#     recorder = EpisodeRecorder(enabled=record_video, save_dir=output_dir)

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

#     # Record initial state
#     recorder.record_frame(env.state)

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

#         # Add after each step_info update:
#         recorder.record_frame(env.state)

#     # Store final achievements
#     raw_achievements = env.terminate()
#     episode_data["final_achievements"] = {
#         k: bool(v) for k, v in raw_achievements.items()
#     }

#     # Save video if recording was enabled
#     if record_video and output_dir:
#         print("Output dir: ", output_dir)  # output_dir /
#         video_filename = f"episode_{mode}_{seed}.mp4"
#         recorder.save_video(str(video_filename), fps=3)

#     return episode_data


# async def generate_episodes(
#     lm: LM,
#     mode: Literal["classic", "full"] = "classic",
#     seeds: List[int] = [0],
#     max_steps: int = 200,
#     output_dir: Path = None,
# ):
#     """Generate multiple episodes using the provided agent"""

#     # Run episodes sequentially to avoid trace overlap
#     all_episodes_data = []
#     full_episodes_data = []
#     async for seed in tqdm_asyncio(seeds, desc=f"Running {mode} episodes"):
#         agent = SimpleReActLanguageAgent(
#             lm=lm,
#             mode=react_config["agent"]["mode"],
#             config={
#                 "max_history": react_config["agent"]["max_history"],
#                 "max_agent_steps": react_config["agent"]["max_agent_steps"],
#             },
#         )
#         episode_data = await generate_single_episode(
#             agent, mode, seed, max_steps, output_dir
#         )
#         full_episodes_data.append(episode_data)
#         # Calculate score based on achievements
#         score = sum(1 for v in episode_data["final_achievements"].values() if v)
#         all_episodes_data.append(
#             {
#                 "system_instance_id": agent.system_instance_id,
#                 "score": score,
#             }
#         )

#         # Get traces after each episode
#         print("Trace Keys: ", event_store._traces.keys())
#         # traces = event_store.get_system_traces()
#         # all_traces.extend(traces)
#         # Clear event store for next episode
#         # event_store.clear()

#     # Process results
#     # total_achievements_list = []
#     # for episode_data in all_episodes_data:
#     #     total_achievements = sum(
#     #         1 for v in episode_data["final_achievements"].values() if v
#     #     )
#     #     total_achievements_list.append(total_achievements)

#     #     if output_dir:
#     #         output_dir = Path(output_dir)
#     #         output_dir.mkdir(parents=True, exist_ok=True)
#     #         # output_file = output_dir / f"episode_{mode}_{episode_data['seed']}.json"
#     #         # with open(output_file, "w") as f:
#     #         #     json.dump(episode_data["final_achievements"], f, indent=2)

#     # Calculate average achievements
#     # avg_achievements = sum(total_achievements_list) / len(total_achievements_list)
#     # print(
#     #     f"\nAverage achievements across {len(seeds)} episodes: {avg_achievements:.2f}"
#     # )

#     # Upload results with individual reward signals per episode
#     results = upload(
#         dataset=Dataset(
#             questions=[
#                 TrainingQuestion(
#                     intent=" ",
#                     criteria="Got as many achievements as possible in a single episode",
#                     question_id="default",
#                 )
#             ],
#             reward_signals=[
#                 RewardSignal(
#                     question_id="default",
#                     system_instance_id=episode["system_instance_id"],
#                     reward=episode["score"],
#                     annotation=f"Achieved {episode['score']} achievements in episode",
#                 )
#                 for episode in all_episodes_data
#             ],
#         ),
#         # traces=all_traces  # Pass accumulated traces
#     )
#     print("Results: ", len(results[2]))
#     print("Lens per result: ", [len(result) for result in results[3]])
#     print("Results[3]: ", len(results[3]))
#     print("Uploaded")
#     assert_compute_inputs_not_empty(results)
#     return full_episodes_data


# def assert_compute_inputs_not_empty(results):
#     """Assert that compute inputs in traces are not empty"""
#     response, questions_json, reward_signals_json, traces_json = results
#     found_compute_steps = False

#     for trace in traces_json:
#         if "partition" not in trace:
#             continue

#         for partition in trace["partition"]:
#             if "events" not in partition:
#                 continue

#             for event in partition["events"]:
#                 if "agent_compute_steps" in event:
#                     found_compute_steps = True
#                     for step in event["agent_compute_steps"]:
#                         assert step[
#                             "compute_input"
#                         ], f"Empty compute_input in agent step: {step}"
#                         assert (
#                             not isinstance(step["compute_input"], list)
#                             or len(step["compute_input"]) > 0
#                         ), "Agent compute_input is an empty list"
#                 if "environment_compute_steps" in event:
#                     found_compute_steps = True
#                     for step in event["environment_compute_steps"]:
#                         assert step[
#                             "compute_input"
#                         ], f"Empty compute_input in environment step: {step}"
#                         assert (
#                             not isinstance(step["compute_input"], list)
#                             or len(step["compute_input"]) > 0
#                         ), "Environment compute_input is an empty list"

#     assert found_compute_steps, "No compute steps found in traces"
#     return True


# async def run_craftax_episodes(
#     n_seeds: int = 10,
# ):
#     """Run multiple Craftax episodes"""
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
#     output_dir = Path("dev/testing/craftax/records")

#     # Create agent

#     # Run episodes with multiple seeds
#     seeds = range(0, n_seeds)  # You can modify this list to run more or fewer episodes
#     episodes_data = await generate_episodes(
#         lm=lm,
#         mode="classic",
#         seeds=seeds,
#         output_dir=output_dir,
#         max_steps=max_steps,
#     )

#     # Add assertions to verify the episodes
#     assert episodes_data is not None
#     assert len(episodes_data) == len(seeds)
#     for episode_data in episodes_data:
#         assert (
#             "final_achievements" in episode_data
#         ), "Final achievements not found - {}".format(episode_data)
#         assert isinstance(episode_data["final_achievements"], dict)
#         assert len(episode_data["steps"]) > 0
#         assert all(
#             isinstance(step["reward"], (int, float)) for step in episode_data["steps"]
#         )


# if __name__ == "__main__":
#     import asyncio

#     asyncio.run(run_craftax_episodes(n_seeds=2))
