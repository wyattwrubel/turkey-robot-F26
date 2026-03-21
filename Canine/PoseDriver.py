import sys
import os
sys.path.append(os.path.dirname(__file__))

import ServoDriver as SD
import ServoController as SC
import asyncio

class PoseDriver():
    # TODO Revise With RPi Wiring
    Front_Left_Pin = 12
    Front_Right_Pin = 13
    Rear_Left_Pin = 18
    Rear_Right_Pin = 19
    Head_Pin = 24

    Default_Angle_Delay = 0.5 # Seconds
 
    def __init__(self, ServoAnglePublisher, Verbose):
        self.servos = {}
        self.Verbose = Verbose
        self.servos["FrontLeft"] = SC.ServoController("FrontLeft", "Leg", self.Front_Left_Pin, 0, True)
        self.servos["FrontRight"] = SC.ServoController("FrontRight", "Leg", self.Front_Right_Pin, 0, True)
        self.servos["RearLeft"] = SC.ServoController("RearLeft", "Leg", self.Rear_Left_Pin, 0, True)
        self.servos["RearRight"] = SC.ServoController("RearRight", "Leg", self.Rear_Right_Pin, 0, True)
        self.servos["Head"] = SC.ServoController("Head", "Head", self.Head_Pin, 0, True)
        self.ServoAnglePublisher = ServoAnglePublisher

    async def CanineMotor_Receiver(self, data):
        if data["Type"] == "Servo":
            await self._Process_ServoCommands(data)
        elif data["Type"] == "Pose":
            await self._Process_PoseCommands(data)
        else:
            print(f"PoseDriver: Invalid 'Type' in {data}") 

    async def MotorPositions_Runner(self):

        while True:
            # Broadcasts Angle Positions at 10 HZ
            for name, servo in self.servos.items():
                await self.ServoAnglePublisher.publish({
                    "Name": name,
                    "Angle": servo.servo.Current_Angle,
                })            
            await asyncio.sleep(0.1)

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
        
        if(data == "IdlePose"):
            await self._IdlePose_Command()
        elif(data == "AlertPose"):
            await self._AlertPose_Command()
        elif(data == "AggressivePose"):
            await self._AggressivePose_Command()
        elif(data == "WalkGait"):
            await self._WalkGait_Command()
        elif(data == "ExcitedGait"):
            await self._ExcitedGait_Command()
        elif(data == "AlertStep"):
            await self._AlertStep_Command()
        elif(data == "HeadScan"):
            await self._HeadScan_Command()
        else:
            print("Invalid pose received in ", data)

    # ------------------------------------------------------------------
    # Fixed Command Protocols
    # ------------------------------------------------------------------

    async def _IdlePose_Command(self):
        if(self.Verbose): print("PoseDriver: Activating IdlePose")
        await asyncio.gather(
            self.servos["FrontLeft"].servo_control_receiver("Fixed", 0),
            self.servos["FrontRight"].servo_control_receiver("Fixed", 0),
            self.servos["RearLeft"].servo_control_receiver("Fixed", 0),
            self.servos["RearRight"].servo_control_receiver("Fixed", 0),
            self.servos["Head"].servo_control_receiver("Fixed", 0),
            asyncio.sleep(self.Default_Angle_Delay)
        )

    async def _AlertPose_Command(self):
        if(self.Verbose): print("PoseDriver: Activating AlertPose")
        await asyncio.gather(
            self.servos["FrontLeft"].servo_control_receiver("Fixed", 10),
            self.servos["FrontRight"].servo_control_receiver("Fixed", 10),
            self.servos["RearLeft"].servo_control_receiver("Fixed", 15),
            self.servos["RearRight"].servo_control_receiver("Fixed", 15),
            self.servos["Head"].servo_control_receiver("Fixed", 0),
            asyncio.sleep(self.Default_Angle_Delay)
        )

    async def _AggressivePose_Command(self):
        if(self.Verbose): print("PoseDriver: Activating AggressivePose")
        await asyncio.gather(
            self.servos["FrontLeft"].servo_control_receiver("Fixed", 30),
            self.servos["FrontRight"].servo_control_receiver("Fixed", 30),
            self.servos["RearLeft"].servo_control_receiver("Fixed", 0),
            self.servos["RearRight"].servo_control_receiver("Fixed", 0),
            self.servos["Head"].servo_control_receiver("Fixed", 0),
            asyncio.sleep(self.Default_Angle_Delay)
        )

    # ------------------------------------------------------------------
    # Oscillation Command Protocols
    # ------------------------------------------------------------------
    
    async def _WalkGait_Command(self):
        
        # Motor Initialization
        if(self.Verbose): print("PoseDriver: Activating WalkGait Initialization")
        await asyncio.gather(
            self.servos["FrontLeft"].servo_control_receiver("Fixed", 15),
            self.servos["FrontRight"].servo_control_receiver("Fixed", -15),
            self.servos["RearLeft"].servo_control_receiver("Fixed", 15),
            self.servos["RearRight"].servo_control_receiver("Fixed", -15),
            asyncio.sleep(self.Default_Angle_Delay)
        )

        # Motor Oscillation
        if(self.Verbose): print("PoseDriver: Activating WalkGait Oscillation")
        await asyncio.gather(
            self.servos["FrontLeft"].servo_control_receiver("Oscillate", 25, -25, 2),
            self.servos["FrontRight"].servo_control_receiver("Oscillate", -25, 25, 2),
            self.servos["RearLeft"].servo_control_receiver("Oscillate", 25, -25, 2),
            self.servos["RearRight"].servo_control_receiver("Oscillate", -25, 25, 2),
            self.servos["Head"].servo_control_receiver("Random", -20, 20, 3),
        )

    async def _ExcitedGait_Command(self):
        
        # Motor Initialization
        if(self.Verbose): print("PoseDriver: Activating ExcitedGait Initialization")
        await asyncio.gather(
            self.servos["FrontLeft"].servo_control_receiver("Fixed", 0),
            self.servos["FrontRight"].servo_control_receiver("Fixed", 0),
            self.servos["RearLeft"].servo_control_receiver("Fixed", 0),
            self.servos["RearRight"].servo_control_receiver("Fixed", 0),
            asyncio.sleep(self.Default_Angle_Delay)
        )

        # Motor Oscillation
        if(self.Verbose): print("PoseDriver: Activating ExcitedGait Oscillation")
        await asyncio.gather(
            self.servos["FrontLeft"].servo_control_receiver("Oscillate", 40, -40, 1),
            self.servos["FrontRight"].servo_control_receiver("Oscillate", -40, 40, 1),
            self.servos["RearLeft"].servo_control_receiver("Oscillate", -40, 40, 1),
            self.servos["RearRight"].servo_control_receiver("Oscillate", 40, -40, 1),
            self.servos["Head"].servo_control_receiver("Random", -45, 45, 1),
        )

    async def _AlertStep_Command(self):
        
        # Motor Initialization
        if(self.Verbose): print("PoseDriver: Activating AlertStep Initialization")
        await asyncio.gather(
            self.servos["FrontLeft"].servo_control_receiver("Fixed", 0),
            self.servos["FrontRight"].servo_control_receiver("Fixed", 0),
            self.servos["RearLeft"].servo_control_receiver("Fixed", 0),
            self.servos["RearRight"].servo_control_receiver("Fixed", 0),
            self.servos["Head"].servo_control_receiver("Fixed", 0),
            asyncio.sleep(self.Default_Angle_Delay)
        )

        # Motor Oscillation
        if(self.Verbose): print("PoseDriver: Activating AlertStep Oscillation")
        await asyncio.gather(
            self.servos["FrontLeft"].servo_control_receiver("Oscillate", 20, -10, 3),
            self.servos["FrontRight"].servo_control_receiver("Oscillate", -10, 20, 3),
            self.servos["RearLeft"].servo_control_receiver("Oscillate", -10, 20, 3),
            self.servos["RearRight"].servo_control_receiver("Oscillate", 20, -10, 3),
            self.servos["Head"].servo_control_receiver("Oscillate", -40, 40, 5),
        )

    async def _HeadScan_Command(self):

        # Motor Initialization
        if(self.Verbose): print("PoseDriver: Activating HeadScan Initialization")
        await asyncio.gather(
            self.servos["Head"].servo_control_receiver("Fixed", 0),
            asyncio.sleep(self.Default_Angle_Delay)
        )

        # Motor Oscillation
        if(self.Verbose): print("PoseDriver: Activating HeadScan Oscillation")        
        await asyncio.gather(
            self.servos["Head"].servo_control_receiver("Oscillate", -60, 60, 4),
        )
