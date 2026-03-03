import sys
import os
sys.path.append(os.path.dirname(__file__))

import ServoDriver as SD
import ServoController as SC
import asyncio

class PoseDriver():
    def __init__(self):
        self.servos = {}
        Verbose = False
        self.servos["FrontLeft"] = SC.ServoController(SD.Servo(None, None, Verbose), Verbose)
        self.servos["FrontRight"] = SC.ServoController(SD.Servo(None, None, Verbose), Verbose)
        self.servos["RearRight"] = SC.ServoController(SD.Servo(None, None, Verbose), Verbose)
        self.servos["RearLeft"] = SC.ServoController(SD.Servo(None, None, Verbose), Verbose)
        self.servos["Head"] = SC.ServoController(SD.Servo(None, None, Verbose), Verbose)

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
            data = data["Data"]
        except KeyError:
            print(f"Malformed command: {data}")
            return

        if name not in self.servos:
            print(f"Invalid servo name: {name}")
            return

        await self.servos[name].servo_control_receiver("Fixed", data)

    async def _Process_PoseCommands(self, data):
        try: 
            data = data["Data"]
        except KeyError:
            print(f"Data not found in ", data)
            return
        
        if(data == "Pose1"):
            await self._Pose1_Protocol()
        elif(data == "Pose2"):
            await self._Pose2_Protocol()
        elif(data == "Pose3"):
            await self._Pose3_Protocol()
        elif(data == "Pose4"):
            await self._Pose4_Protocol()
        elif(data == "Pose5"):
            await self._Pose5_Protocol()
        else:
            print("Invalid pose received in ", data)

    #TODO Update Once Have ServoController Elements Defined < Team 19
    async def _Pose1_Protocol(self):
        await asyncio.gather(
            self.servos["FrontLeft"].servo_control_receiver("Fixed", 90),
            self.servos["FrontRight"].servo_control_receiver("Oscillate", 90),
        )

    async def _Pose2_Protocol(self):
        await asyncio.gather(
            self.servos["FrontLeft"].servo_control_receiver("Fixed", 90),
            self.servos["FrontRight"].servo_control_receiver("Fixed", 90),
        )

    async def _Pose3_Protocol(self):
        await asyncio.gather(
            self.servos["FrontLeft"].servo_control_receiver("Fixed", 90),
            self.servos["FrontRight"].servo_control_receiver("Fixed", 90),
        )

    async def _Pose4_Protocol(self):
        await asyncio.gather(
            self.servos["FrontLeft"].servo_control_receiver("static", 90),
            self.servos["FrontRight"].servo_control_receiver("static", 90),
        )

    async def _Pose5_Protocol(self):
        await asyncio.gather(
            self.servos["FrontLeft"].servo_control_receiver("static", 90),
            self.servos["FrontRight"].servo_control_receiver("static", 90),
        )

async def main():
    driver = PoseDriver()

    # await driver.CanineMotor_Receiver({
    #     "Type": "Servo",
    #     "Name": "FrontLeft",
    #     "Data": 90
    # })
    await driver.CanineMotor_Receiver({
            "Type": "Pose",
            "Data": "Pose1",
        })

    await asyncio.sleep(5)

    await driver.CanineMotor_Receiver({
            "Type": "Pose",
            "Data": "Pose2",
        })

    # await driver.CanineMotor_Receiver({
    #     "Type": "Servo",
    #     "Name": "FrontRight",
    #     "Pattern": "swing"
    # })

    # await driver.CanineMotor_Receiver({
    #     "Type": "Servo",
    #     "Name": "RearRight",
    #     "Pattern": "swing"
    # })

    # await driver.CanineMotor_Receiver({
    #     "Type": "Servo",
    #     "Name": "RearLeft",
    #     "Pattern": "swing"
    # })

    # await asyncio.sleep(0.5)

    # await driver.CanineMotor_Receiver({
    #     "Type": "Servo",
    #     "Name": "FrontLeft",
    #     "Pattern": "static",
    #     "Data": 90
    # })

    # await asyncio.sleep(5)

if __name__ == "__main__":
    asyncio.run(main())