"""Drive the Freenove 4WD car with a trained policy, from a deployment bundle.

Runs on the Raspberry Pi. Nothing here imports torch or Genesis -- the bundle
carries the observation and action pipelines, `genesis_forge_runtime` replays
them, and onnxruntime runs the policy.

Everything it needs is in this directory, including the motor driver. Copy the
directory to the Pi; Raspberry Pi OS has no global pip, so install into a
virtual environment::

    sudo apt install libsdl2-2.0-0
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    python ./run.py

Left stick drives, right stick turns -- the same axes `eval.py` steers the
simulator with. The car only moves while a gamepad is connected, and only once
the driver has armed it by pressing X.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import onnxruntime
from gamepad import Gamepad
from motor_driver import MOTOR_MAX_VALUE, MotorDriver

from genesis_forge_runtime import load_bundle

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument(
    "--bundle",
    type=Path,
    default=Path("./trained_bundle.gfb"),
    help="The exported trained bundle: a directory or a .gfb archive.",
)
args = parser.parse_args()


def calculate_motor_pwm(value: float, clip_range: tuple[float, float]) -> int:
    """A linear range mapping target velocity to motor PWM."""
    low, high = clip_range
    value = max(low, min(high, value))
    motor_min = -MOTOR_MAX_VALUE
    return round(
        (value - low) * (MOTOR_MAX_VALUE - motor_min) / (high - low) + motor_min
    )


def main() -> None:
    bundle = load_bundle(args.bundle)
    print(bundle.describe(), "\n")
    if bundle.policy_path is None:
        print(f"Error: No policy found in {args.bundle}")
        return

    # Create observation and action handlers
    obs_assembler = bundle.create_observation_assembler()
    action_decoder = bundle.create_action_decoder()

    # Connect to the gamepad and motors
    gamepad = Gamepad.wait_for_connection()
    motors = MotorDriver()

    try:
        with bundle.unpacked() as directory:
            # Load the onnx policy
            policy_file = str(directory / bundle.policy_path)
            session = onnxruntime.InferenceSession(
                policy_file, providers=["CPUExecutionProvider"]
            )
            input_name = session.get_inputs()[0].name

            # Start the control loop
            print("Running. Ctrl-C to stop.")
            obs_assembler.reset()
            action_decoder.reset()
            while True:
                # If the gamepad disconnects, stop the motors and wait for reconnection.
                if not gamepad.connected:
                    motors.stop()
                    gamepad.reconnect()

                # Assemble observations
                observation = obs_assembler.assemble(
                    {
                        "velocity_cmd": gamepad.command(),
                        "actions": action_decoder.last_raw_actions,
                    }
                )

                # Get actions
                [raw_action] = session.run(
                    None, {input_name: observation[None, :].astype("float32")}
                )
                action_targets = action_decoder.decode(np.ravel(raw_action))

                # Send actions to the car
                pwm = (
                    calculate_motor_pwm(
                        action_targets.by_joint[name],
                        action_decoder.clip_range_by_joint[name],
                    )
                    for name in [
                        "TT_Motor-1_axel",
                        "TT_Motor-2_axel",
                        "TT_Motor-3_axel",
                        "TT_Motor-4_axel",
                    ]
                )
                motors.set_wheels_pwm(*pwm)

                # The sleep is mostly to keep from flooding the I2C bus.
                time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        motors.close()
        gamepad.close()
        print("\nStopped.")


if __name__ == "__main__":
    main()
