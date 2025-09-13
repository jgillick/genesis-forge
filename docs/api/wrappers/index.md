# Wrappers

Environment wrappers are used to wrap your environment class to add functionality for training. This is similar to how wrappers work in Gymnasium and StableBaselines3.

For example, you might want to use the rsl_rl training framework and regularly capture videos during training:

```python
    # Your environment
    env = Go2SimpleEnv(num_envs=10, headless=True)

    # Record a video of every 5th episode
    env = VideoWrapper(
        env,
        video_length_sec=12,
        out_dir=os.path.join(log_dir, "videos"),
        episode_trigger=lambda episode_id: episode_id % 5 == 0,
    )

    # Make the environment compatible with rsl_rl
    env = RslRlWrapper(env)

    # Build the environment
    env.build()
    env.reset()

    # Train
    runner = OnPolicyRunner(env, training_cfg, log_dir, device=gs.device)
    runner.learn(num_learning_iterations=max_iterations)
    env.close()
```

```{toctree}
:maxdepth: 1

rsl_rl
skrl
video
```
