"""
Run a trained Go2 locomotion policy using the deployment pipeline.

This script shows how to run a policy on a real robot using only
``genesis_forge.deploy`` -- no Genesis simulation, no RSL-RL, no training
dependencies required on the target hardware.

For demonstration purposes, Genesis is used here as a "mock robot" that
provides realistic sensor readings.  Every section that reads from the
simulation is clearly marked and would be replaced with calls to your
actual robot SDK (e.g. unitree_sdk2py, ROS2, etc.) when running on
hardware.

Usage::

    # Export artifacts first (from the examples/deploy directory):
    python export_config.py -e go2-simple

    # Run the deployment loop (simulation acting as the robot):
    python robot.py -e go2-simple

    # Run on CPU:
    python robot.py -e go2-simple -d cpu
"""

import os
import argparse

import torch
import genesis as gs

# -----------------------------------------------------------------------
# genesis_forge.deploy is the ONLY genesis_forge import you need on the
# real robot.  ObservationBuilder and ActionDecoder have no Genesis
# dependency -- they only require torch.
# -----------------------------------------------------------------------
from genesis_forge.deploy import DeploymentConfig, ObservationBuilder, ActionDecoder

# The following imports are only needed to drive the mock-robot simulation.
# On real hardware these would be replaced by your robot SDK.
from environment import Go2Env

EXPERIMENT_NAME = "go2-simple"

parser = argparse.ArgumentParser(add_help=True)
parser.add_argument("-d", "--device", type=str, default="gpu")
parser.add_argument("-e", "--exp_name", type=str, default=EXPERIMENT_NAME)
parser.add_argument("--headless", action="store_true", default=False)
args = parser.parse_args()


def main():
    log_path = f"./logs/{args.exp_name}"
    config_path = os.path.join(log_path, "deploy_config.json")
    policy_path = os.path.join(log_path, "policy.pt")

    for p in [config_path, policy_path]:
        if not os.path.exists(p):
            print(f"Missing: {p}")
            print("Run export_config.py first.")
            exit(1)

    # ------------------------------------------------------------------
    # Load deployment artifacts
    # These two files + torch are all you need on the robot.
    # ------------------------------------------------------------------
    config = DeploymentConfig.from_json(config_path)
    obs_builder = ObservationBuilder(config)
    action_decoder = ActionDecoder(config)

    print(
        f"Loaded deploy config: {config.num_observations} obs, {config.num_actions} actions"
    )
    print(
        f"Control dt: {config.dt}s  |  history: {config.observation_history_len} frame(s)"
    )

    # Load the TorchScript policy (no RSL-RL needed)
    policy = torch.jit.load(policy_path)
    policy.eval()
    print(f"Loaded policy from: {policy_path}")

    # ------------------------------------------------------------------
    # Mock robot: Genesis simulation
    # On real hardware, replace this entire block with your robot SDK
    # initialization (e.g. unitree_sdk2py channel factory setup).
    # ------------------------------------------------------------------
    backend = gs.gpu if args.device == "gpu" else gs.cpu
    if args.device == "cpu":
        torch.set_default_device("cpu")
    gs.init(logging_level="warning", backend=backend)

    sim = Go2Env(num_envs=1, headless=args.headless)
    sim.build()
    sim.reset()

    # ------------------------------------------------------------------
    # Deployment loop
    # ------------------------------------------------------------------
    print("\nRunning deployment loop. Press Ctrl+C to stop.\n")
    last_actions = torch.zeros(config.num_actions)
    obs_builder.reset()

    try:
        while True:
            # ==============================================================
            # SENSOR READS
            # On a real robot, replace each sim call below with the
            # equivalent call to your hardware SDK, for example:
            #   angle_velocity  = imu.get_gyroscope()         # shape (3,)
            #   linear_velocity = imu.get_linear_velocity()   # shape (3,)
            #   projected_gravity = imu.get_gravity_vector()  # shape (3,)
            #   dof_position    = robot.get_joint_positions() # shape (n,)
            #   dof_velocity    = robot.get_joint_velocities()# shape (n,)
            # ==============================================================
            robot = sim.robot
            robot_manager = sim.robot_manager
            action_manager = sim.action_manager

            sensor_values = {
                "angle_velocity": robot_manager.get_angular_velocity()[0],
                "linear_velocity": robot_manager.get_linear_velocity()[0],
                "projected_gravity": robot_manager.get_projected_gravity()[0],
                "dof_position": action_manager.get_dofs_position()[0],
                "dof_velocity": action_manager.get_dofs_velocity()[0],
                "actions": last_actions,
            }

            # ==============================================================
            # POLICY INFERENCE  (same on real robot and in simulation)
            # ==============================================================

            # 1. Assemble observation tensor from sensor readings
            obs = obs_builder.build_obs(sensor_values)  # shape: (num_observations,)

            # 2. Run the policy
            with torch.no_grad():
                raw_actions = policy(obs.unsqueeze(0)).squeeze(
                    0
                )  # shape: (num_actions,)

            # 3. Decode raw policy output into named joint position targets
            joint_commands = action_decoder.decode(raw_actions)
            # e.g. {"FL_hip_joint": 0.02, "FL_thigh_joint": 0.78, ...}

            last_actions = raw_actions.detach()

            # ==============================================================
            # ACTUATOR COMMANDS
            # On a real robot, replace the sim step below with your hardware
            # SDK's joint position command, for example:
            #   for name, pos in joint_commands.items():
            #       robot.set_joint_position(name, pos)
            # ==============================================================
            action_tensor = torch.tensor(
                list(joint_commands.values()), dtype=torch.float32
            ).unsqueeze(0)
            sim.action_manager.actuator_manager.control_dofs_position(
                action_tensor, sim.action_manager.dofs_idx
            )
            sim.scene.step()

    except KeyboardInterrupt:
        print("\nStopped.")
    except gs.GenesisException as e:
        if e.message != "Viewer closed.":
            raise e


if __name__ == "__main__":
    main()
