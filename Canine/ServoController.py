import sys
import os
sys.path.append(os.path.dirname(__file__))
import ServoDriver as SD
import asyncio
import random

class ServoController:
    def __init__(self, Name, ServoType, GPIO_PIN, Current_Angle, Verbose=False):
        if(ServoType == "Leg" or ServoType == "Head"):
            self.servo = SD.Servo(Name, ServoType, GPIO_PIN, Current_Angle, Verbose)
        else:
            print("ServoController: Invalid 'ServoType' Received ", ServoType)
        
        self._task = None
        self.Verbose = Verbose
        self.ServoType = ServoType

    async def servo_control_receiver(self, command_name, angle1 = None, angle2 = None, period = None):
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass  
        self._task = asyncio.create_task(self._execute_servo_command(command_name, angle1, angle2, period))

    async def _execute_servo_command(self, command_name, angle1, angle2, period):
        if command_name == "Fixed":
            await self._Fixed_Command(angle1)
        elif command_name == "Oscillate":
            await self._Oscillate_Command(angle1, angle2, period)
        elif command_name == "Random":
            await self._Random_Command(angle1, angle2, period)
        else:
            print(f"ServoController: Unknown command {command_name}")

    async def _Fixed_Command(self, angle, max_speed=120):
        """ Sets servo motor position to [angle] capped at max_speed (deg/s) """
        if self.Verbose: print(f"ServoController: Received Fixed Command {angle}")

        angle_resolution = 1  # degrees per step
        if(self.ServoType == "Head"):
            max_speed = 180 # degrees per second

        step_time = angle_resolution / max_speed  # seconds per step

        # current = self.servo.Current_Angle or 0
        # direction = 1 if angle > current else -1

        # while abs(angle - self.servo.Current_Angle) > angle_resolution:
        #     next_angle = self.servo.Current_Angle + direction * angle_resolution
        #     self.servo.SetAngle(next_angle)
        #     await asyncio.sleep(step_time)

        self.servo.SetAngle(angle)

    async def _Oscillate_Command(self, angle1, angle2, period):
        """ Oscillates between angle1 and angle2 bounds every period seconds. Angle1 will be approached first """

        if self.Verbose: print(f"ServoController: Received Oscillate Command From {angle1} to {angle2} in {period} seconds")

        current_angle = self.servo.Current_Angle

        # Check if Servo Is In Bounds. If Not, Set To Angle1 and Wait 1 Second
        if(min(angle1, angle2) <= current_angle <= max(angle1,angle2)):
            self.servo.SetAngle(angle1)
            await asyncio.sleep(1)

        angle_resolution = 2  # degrees
        total_distance = abs(angle2 - angle1)
        num_steps = total_distance / angle_resolution
        step_time = (period / 2) / num_steps # Assuming negligible time between angle steps. Will need to change if longer than expected. 

        target_angle = angle1

        while True:
            current_angle = self.servo.Current_Angle
            distance = abs(target_angle - current_angle)

            if distance <= angle_resolution:
                target_angle = angle2 if target_angle == angle1 else angle1
            else:
                direction = 1 if target_angle > current_angle else -1
                self.servo.SetAngle(current_angle + direction * angle_resolution)

            await asyncio.sleep(step_time)

    async def _Random_Command(self, angle1, angle2, period):
        """ Selects random angle from [angle1 to angle2) every period seconds """

        if self.Verbose: print(f"ServoController: Received Random Command From {angle1} To {angle2} in {period} seconds")
        while True:
            random_angle = random.randint(angle1, angle2)
            self.servo.SetAngle(random_angle)
            await asyncio.sleep(period)