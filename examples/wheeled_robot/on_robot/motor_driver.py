#!/usr/bin/python

import math
import time

import smbus

# ============================================================================
# Raspi PCA9685 16-Channel Motor Driver
# Copied and modified from
# https://github.com/Freenove/Freenove_4WD_Smart_Car_Kit_for_Raspberry_Pi/blob/master/Code/Server/pca9685.py
# ============================================================================

MOTOR_MAX_VALUE = 4095
STOP_DUTY = 0


class MotorDriver:
    __MODE1 = 0x00
    __PRESCALE = 0xFE
    __REG_ON_L = 0x06
    __REG_ON_H = 0x07
    __REG_OFF_L = 0x08
    __REG_OFF_H = 0x09

    MOTOR1_A = 6
    MOTOR1_B = 7
    MOTOR2_A = 5
    MOTOR2_B = 4
    MOTOR3_A = 0
    MOTOR3_B = 1
    MOTOR4_A = 2
    MOTOR4_B = 3

    def __init__(self, address: int = 0x40, debug: bool = False):
        self.bus = smbus.SMBus(1)
        self.address = address
        self.debug = debug
        self.write(self.__MODE1, 0x00)
        self.set_pwm_freq(50)

    def write(self, reg: int, value: int) -> None:
        """Writes an 8-bit value to the specified register/address."""
        self.bus.write_byte_data(self.address, reg, value)

    def read(self, reg: int) -> int:
        """Read an unsigned byte from the I2C device."""
        result = self.bus.read_byte_data(self.address, reg)
        return result

    def set_pwm_freq(self, freq: float) -> None:
        """Sets the PWM frequency."""
        prescaleval = 25000000.0  # 25MHz
        prescaleval /= 4096.0  # 12-bit
        prescaleval /= float(freq)
        prescaleval -= 1.0
        prescale = math.floor(prescaleval + 0.5)

        oldmode = self.read(self.__MODE1)
        newmode = (oldmode & 0x7F) | 0x10  # sleep
        self.write(self.__MODE1, newmode)  # go to sleep
        self.write(self.__PRESCALE, int(math.floor(prescale)))
        self.write(self.__MODE1, oldmode)
        time.sleep(0.005)
        self.write(self.__MODE1, oldmode | 0x80)

    def set_pwm(self, channel: int, on: int, off: int) -> None:
        """Sets a single PWM channel."""
        self.write(self.__REG_ON_L + 4 * channel, on & 0xFF)
        self.write(self.__REG_ON_H + 4 * channel, on >> 8)
        self.write(self.__REG_OFF_L + 4 * channel, off & 0xFF)
        self.write(self.__REG_OFF_H + 4 * channel, off >> 8)

    def set_motor_pwm(self, channel: int, duty: int) -> None:
        """Sets the PWM duty cycle for a motor."""
        self.set_pwm(channel, 0, duty)

    def close(self) -> None:
        """Close the I2C bus."""
        self.stop()
        self.bus.close()

    def stop(self) -> None:
        """Stop all motors."""
        self.set_wheels_pwm(0, 0, 0, 0)

    def set_wheels_pwm(self, duty1: int, duty2: int, duty3: int, duty4: int) -> None:
        """Set the PWM duty cycle for all four wheels."""
        self.wheel1(duty1)
        self.wheel2(duty2)
        self.wheel3(duty3)
        self.wheel4(duty4)

    def wheel1(self, duty):
        """Set the PWM duty cycle for the right/front wheel."""
        if duty > 0:
            self.set_motor_pwm(self.MOTOR1_A, 0)
            self.set_motor_pwm(self.MOTOR1_B, duty)
        elif duty < 0:
            self.set_motor_pwm(self.MOTOR1_A, abs(duty))
            self.set_motor_pwm(self.MOTOR1_B, 0)
        else:
            self.set_motor_pwm(self.MOTOR1_A, STOP_DUTY)
            self.set_motor_pwm(self.MOTOR1_B, STOP_DUTY)

    def wheel2(self, duty):
        """Set the PWM duty cycle for the right/rear wheel."""
        if duty > 0:
            self.set_motor_pwm(self.MOTOR2_A, 0)
            self.set_motor_pwm(self.MOTOR2_B, duty)
        elif duty < 0:
            self.set_motor_pwm(self.MOTOR2_A, abs(duty))
            self.set_motor_pwm(self.MOTOR2_B, 0)
        else:
            self.set_motor_pwm(self.MOTOR2_A, STOP_DUTY)
            self.set_motor_pwm(self.MOTOR2_B, STOP_DUTY)

    def wheel3(self, duty):
        """Set the PWM duty cycle for the left/front wheel."""
        if duty > 0:
            self.set_motor_pwm(self.MOTOR3_A, 0)
            self.set_motor_pwm(self.MOTOR3_B, duty)
        elif duty < 0:
            self.set_motor_pwm(self.MOTOR3_A, abs(duty))
            self.set_motor_pwm(self.MOTOR3_B, 0)
        else:
            self.set_motor_pwm(self.MOTOR3_A, STOP_DUTY)
            self.set_motor_pwm(self.MOTOR3_B, STOP_DUTY)

    def wheel4(self, duty):
        """Set the PWM duty cycle for the left/rear wheel."""
        if duty > 0:
            self.set_motor_pwm(self.MOTOR4_A, 0)
            self.set_motor_pwm(self.MOTOR4_B, duty)
        elif duty < 0:
            self.set_motor_pwm(self.MOTOR4_A, abs(duty))
            self.set_motor_pwm(self.MOTOR4_B, 0)
        else:
            self.set_motor_pwm(self.MOTOR4_B, STOP_DUTY)
            self.set_motor_pwm(self.MOTOR4_A, STOP_DUTY)


if __name__ == "__main__":
    pass
