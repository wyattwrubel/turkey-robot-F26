import asyncio

class ServoController:
    
    def __init__(self, servo):
        self.servo = servo
        self._task = None

    async def run_pattern(self, command_name, data=None):
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

        self._task = asyncio.create_task(
            self._execute_pattern(command_name, data)
        )

#HI THERE

    async def _execute_pattern(self, command_name, data):
        if command_name == "swing":
            await self._pattern_swing()
        elif command_name == "static":
            await self._pattern_static(data)
        else:
            print(f"Unknown pattern: {command_name}")

    # DEFINE ALL THE ROUTINES TO BE RUN TODO
    async def _pattern_swing(self):
        print("SETTING SWINGING ANGLE")
        while True:
            self.servo.SetAngle(-60)
            await asyncio.sleep(0.4)
            self.servo.SetAngle(60)
            await asyncio.sleep(0.4)

    # DEFINE ALL THE ROUTINES TO BE RUN TODO
    async def _pattern_static(self, angle):
        print("SETTING STATIC ANGLE")
        self.servo.SetAngle(angle)