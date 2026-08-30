# Genesis Forge Examples

This directory contains a series of examples demonstrating how to use Genesis Forge. Each example builds on the previous one, in this order:

1. [Basic locomotion](./basic/)
2. [Command Direction](./command_direction/)
3. [Contact Sensors and foot air time](./contacts/)
4. [Domain Randomization](./domain_randomization/)
5. [Rough Terrain](./rough_terrain/)
6. [Gait Trainer](./gait_trainer/)

Wheeled robots:

1. [Wheeled robot](./wheeled_robot/) — drive a 4-wheeled car in a commanded direction
2. [Obstacle avoidance](./wheeled_robot_obstacles/) — add an ultrasonic range sensor and obstacles to avoid
3. [Goal navigation](./wheeled_robot_goal_nav/) — navigate to a goal position instead of following velocity commands

Advanced examples:

- [Multi-Agent RL (MASQ)](./multi_agent_rl) — use multiple agents (one agent per leg) to train a robot to walk.
- [Humanoid locomotion](./berkeley_humanoid/) — Berkeley humanoid robot
- [Stand up](./stand_up/) - Go2 learns to stand up from the ground
