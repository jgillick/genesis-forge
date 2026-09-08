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
simulator with. The car only moves while a gamepad is connected.
"""

from __future__ import annotations

import argparse
import numpy as np
import onnxruntime
import time
from pathlib import Path

from gamepad import Gamepad
from motor_driver import MotorDriver, MOTOR_MAX_VALUE

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


def gamepad_connect(gamepad: Gamepad | None, motors: MotorDriver):
    """Wait until the gamepad is connected."""
    if gamepad is None or not gamepad.attached:
        print("Gamepad disconnected. Press any button to reconnect.")
        # car.set_motor_model(0, 0, 0, 0)
        motors.stop()
        if gamepad is not None:
            print("Gamepad disconnected.")
            gamepad.close()
        gamepad = Gamepad.wait_for_connection()
        print(f"{gamepad.name}: left stick drives, right stick turns")
    return gamepad


def main() -> None:
    bundle = load_bundle(args.bundle)
    print(bundle.describe(), "\n")

    # car = get_car()
    motors = MotorDriver()
    gamepad = None

    try:
        with bundle.unpacked() as directory:
            policy_file = str(directory / "policy" / bundle.policy_files[0])
            session = onnxruntime.InferenceSession(
                policy_file, providers=["CPUExecutionProvider"]
            )
            input_name = session.get_inputs()[0].name

            # Create observation and action handlers
            obs_assembler = bundle.create_observation_assembler()
            action_decoder = bundle.create_action_decoder()

            # The maximum velocity defined as the clip value on VelocityActionManager
            joint_clip = action_decoder.clip_range_by_joint

            print("Running. Ctrl-C to stop.")
            while True:
                # Wait for the gamepad controller before doing anything else.
                gamepad = gamepad_connect(gamepad, motors)

                # Assemble observations
                observation = obs_assembler.assemble(
                    {
                        "velocity_cmd": gamepad.command(),
                        "actions": action_decoder.last_raw_actions,
                    }
                )

                # Get actions
                raw_action = session.run(
                    None, {input_name: observation[None, :].astype("float32")}
                )[0]
                action_targets = action_decoder.decode(np.ravel(raw_action))

                # Send actions to the car
                pwm = (
                    calculate_motor_pwm(action_targets.by_joint[name], joint_clip[name])
                    for name in [
                        "TT_Motor-1_axel",
                        "TT_Motor-2_axel",
                        "TT_Motor-3_axel",
                        "TT_Motor-4_axel",
                    ]
                )
                motors.set_wheels_pwm(*pwm)

                # The sleep is mostly to keep from flooding the I2C bus.
                time.sleep(0.02)
    except KeyboardInterrupt:
        pass
    finally:
        motors.close()
        if gamepad is not None:
            gamepad.close()
        print("\nStopped.")


if __name__ == "__main__":
    main()
