import ServoDriver
import ServoController
import asyncio

class PoseDriver():
    def __init__(self):
        self.servos = {}
        Verbose = True
        self.servos["FrontLeft"] = ServoController.ServoController(ServoDriver.Servo(None, None, Verbose))
        self.servos["FrontRight"] = ServoController.ServoController(ServoDriver.Servo(None, None, Verbose))
        self.servos["RearRight"] = ServoController.ServoController(ServoDriver.Servo(None, None, Verbose))
        self.servos["RearLeft"] = ServoController.ServoController(ServoDriver.Servo(None, None, Verbose))
        self.servos["Head"] = ServoController.ServoController(ServoDriver.Servo(None, None, Verbose))

    async def CanineMotor_Receiver(self, data):
        if data["Type"] == "Servo":
            await self._Process_ServoCommands(data)
        elif data["Type"] == "Pose":
            await self._Process_PoseCommands(data)
        else:
            print("INVALID POSEDRIVER TYPE RECEIVED")

    async def _Process_ServoCommands(self, data):
        try:
            name = data["Name"]
            pattern = data["Pattern"]
            payload = data.get("Data", None)
        except KeyError:
            print(f"Malformed command: {data}")
            return

        if name not in self.servos:
            print(f"Invalid servo name: {name}")
            return

        await self.servos[name].run_pattern(pattern, payload)

    async def _Process_PoseCommands(self, data):
        # Example: coordinated multi-servo pose
        await asyncio.gather(
            self.servos["FrontLeft"].run_pattern("static", 90),
            self.servos["FrontRight"].run_pattern("static", 90),
        )

async def main():
    driver = PoseDriver()

    await driver.CanineMotor_Receiver({
        "Type": "Servo",
        "Name": "FrontLeft",
        "Pattern": "swing"
    })

    await driver.CanineMotor_Receiver({
        "Type": "Servo",
        "Name": "FrontRight",
        "Pattern": "swing"
    })

    await driver.CanineMotor_Receiver({
        "Type": "Servo",
        "Name": "RearRight",
        "Pattern": "swing"
    })

    await driver.CanineMotor_Receiver({
        "Type": "Servo",
        "Name": "RearLeft",
        "Pattern": "swing"
    })

    await asyncio.sleep(0.5)

    await driver.CanineMotor_Receiver({
        "Type": "Servo",
        "Name": "FrontLeft",
        "Pattern": "static",
        "Data": 90
    })

    await asyncio.sleep(5)

if __name__ == "__main__":
    asyncio.run(main())