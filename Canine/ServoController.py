import asyncio

class ServoController:
    def __init__(self, servo, Verbose=False):
        self.servo = servo
        self._task = None
        self.Verbose = Verbose

    async def servo_control_receiver(self, command_name, data=None):
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            
        self._task = asyncio.create_task(self._execute_servo_command(command_name, data))

    async def _execute_servo_command(self, command_name, data):

        # TODO Add others
        if command_name == "Fixed":
            await self._Fixed_Command(data)
        elif command_name == "MoveWithSpeed":
            await self._MoveWithSpeed_Command()
        elif command_name == "Oscillate":
            await self._Oscillate_Command()
        else:
            print(f"ServoController: Unknown command {command_name}")

    # DEFINE ALL THE COMMANDS TO BE RUN TODO
    async def _Fixed_Command(self, angle):
        if self.Verbose: print("RECEIVED FIXED COMMAND")
        self.servo.SetAngle(angle)

    # DEFINE ALL THE COMMANDS TO BE RUN TODO
    async def _MoveWithSpeed_Command(self):
        if self.Verbose: print("RECEIVED MOVEWITHSPEED COMMAND")
        while True:
            self.servo.SetAngle(-60)
            await asyncio.sleep(0.4)
            self.servo.SetAngle(60)
            await asyncio.sleep(0.4)

    # DEFINE ALL THE COMMANDS TO BE RUN TODO
    async def _Oscillate_Command(self):
        if self.Verbose: print("RECEIVED OSCILLATION COMMAND")
        while True:
            self.servo.SetAngle(-60)
            await asyncio.sleep(0.4)
            self.servo.SetAngle(60)
            await asyncio.sleep(0.4)

    # TODO Add other commands as well