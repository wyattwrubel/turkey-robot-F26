import sys
import os
sys.path.append(os.path.dirname(__file__))
import ServoDriver as SD
import asyncio
import random

class ServoController:

    def __init__(self, Name, ServoType, GPIO_PIN, Offset, IsFlipped, Verbose=False):
        if(ServoType == "Leg" or ServoType == "Head"):
            self.servo = SD.Servo(Name, ServoType, GPIO_PIN, Offset, IsFlipped, Verbose)
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
        mid = (angle1 + angle2) // 2
        waypoints = [angle1, mid, angle2, mid]
        step_time = period / len(waypoints)

        while True:
            for waypoint in waypoints:
                self.servo.SetAngle(waypoint)
                await asyncio.sleep(step_time)

    async def _Random_Command(self, angle1, angle2, period):
        """ Selects random angle from [angle1 to angle2) every period seconds """

        if self.Verbose: print(f"ServoController: Received Random Command From {angle1} To {angle2} in {period} seconds")
        while True:
            random_angle = random.randint(angle1, angle2)
            self.servo.SetAngle(random_angle)
            await asyncio.sleep(period)