import sys
import os
sys.path.append(os.path.dirname(__file__))

import ServoDriver as SD
import ServoController as SC
import asyncio

class PoseDriver():
    def __init__(self, Verbose):
        self.servos = {}
        self.Verbose = Verbose

        #TODO UPDATE WITH CORRECT PINOUTS
        #TODO ADD IN DIFFERENT MOTOR SPECS?

        self.servos["FrontLeft"] = SC.ServoController(SD.Servo(None, Verbose), Verbose)
        self.servos["FrontRight"] = SC.ServoController(SD.Servo(None, Verbose), Verbose)
        self.servos["RearRight"] = SC.ServoController(SD.Servo(None, Verbose), Verbose)
        self.servos["RearLeft"] = SC.ServoController(SD.Servo(None, Verbose), Verbose)
        self.servos["Head"] = SC.ServoController(SD.Servo(None, Verbose), Verbose)

    async def CanineMotor_Receiver(self, data):
        if data["Type"] == "Servo":
            await self._Process_ServoCommands(data)
        elif data["Type"] == "Pose":
            await self._Process_PoseCommands(data)
        else:
            print(f"PoseDriver: Invalid 'Type' in {data}") 

    async def _Process_ServoCommands(self, data):
        try:
            name = data["Name"]
            data = data["Data"]
        except KeyError:
            print(f"PoseDriver: Missing 'Name' or 'Data' in {data}") 
            return

        if name not in self.servos:
            print(f"PoseDriver: Invalid 'Data' in {data}") 
            return

        await self.servos[name].servo_control_receiver("Fixed", data)

    async def _Process_PoseCommands(self, data):
        try: 
            data = data["Data"]
        except KeyError:
            print(f"PoseDriver: Missing 'Data' in {data}") 
            return
        
        #TODO UPDATE THESE WITH REAL NEEDS FROM 19
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
    # INCLUDE ANY PRINTOUTS WITH VERBOSE DELAYS ARE DONE MANUALLY
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