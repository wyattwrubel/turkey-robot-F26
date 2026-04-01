import sys
import os
sys.path.append(os.path.dirname(__file__))
import ServoDriver as SD
import asyncio
import random

class ServoController:

    def __init__(self, Name, ServoType, Channel_Pin, Offset, IsFlipped, Simulated, Verbose=False):
        if(ServoType == "Leg" or ServoType == "Head"):
            self.servo = SD.Servo(Name, ServoType, Channel_Pin, Offset, IsFlipped, Simulated, Verbose)
        else:
            print("ServoController: Invalid 'ServoType' Received ", ServoType)
        
        self._task = None
        self.Verbose = Verbose
        self.Name = Name
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
        self.servo.SetAngle(angle)

    async def _Oscillate_Command(self, angle1, angle2, period):
        if self.Verbose:
            print(f"ServoController: Oscillate {angle1} <-> {angle2} over {period}s")

        resolution = 1
        total_steps = abs(angle2 - angle1) / resolution
        step_time = (period / 2) / total_steps

        current = float(angle1)
        target = float(angle2)
        next_fire = asyncio.get_event_loop().time()

        while True:
            self.servo.SetAngle(int(current))

            current += resolution * (1 if target > current else -1)
            current = max(min(angle1, angle2), min(max(angle1, angle2), current))
            if int(current) == int(target):
                target = angle2 if target == angle1 else angle1

            next_fire += step_time
            sleep_dur = next_fire - asyncio.get_event_loop().time()
            if sleep_dur > 0:
                await asyncio.sleep(sleep_dur)

    async def _Random_Command(self, angle1, angle2, period):
        """ Selects random angle from [angle1 to angle2) every period seconds """

        if self.Verbose: print(f"ServoController: Received Random Command From {angle1} To {angle2} in {period} seconds")
        while True:
            random_angle = random.randint(angle1, angle2)
            self.servo.SetAngle(random_angle)
            await asyncio.sleep(period)