import sys
import os
sys.path.append(os.path.dirname(__file__))

import PoseDriver
import AudioDriver

class CanineDriver:
    
    UI_lock = False

    def __init__(self):
        self.Audio = AudioDriver.AsyncAudioDriver()
        self.PoseDriver = PoseDriver.PoseDriver()
    
    async def CanineCommand_Receiver(self, data):
        try:
            cmd_id = data["Id"]
        except KeyError:
           print(f"Bad Motor Command ID Received: 'Id' key missing in {data}")

        ## Check ID if UI or Not
        if(cmd_id == "UnlockUI"):
            self.UI_lock = False

        elif(cmd_id == "UI"):
            self.UI_lock = True
            # HANDLE AUDIO, RAW MOTOR, MOTOR POSES
            await self.Process_CanineData(data)

        elif(cmd_id == "System"):
            if(self.UI_lock == False):
                # HANDLE AUDIO, RAW MOTOR, MOTOR POSES
                await self.Process_CanineData(data)
        else:
            print("Canine Receiver: Invalid Data Received", data)

    async def Process_CanineData(self, data):
        try:
            cmd_type = data["Type"]
        except KeyError:
           print(f"Bad Canine Command Type Received: 'Type' key missing in {data}") 

        try:
            cmd_data = data["Data"]
        except KeyError:
           print(f"Bad Canine Command Data Received: 'Data' key missing in {data}") 

        if(cmd_type == "Audio" and cmd_data in self.Audio.sounds):
            self.Audio.play(cmd_data)
        elif(cmd_type == "Servo" or cmd_type == "Pose"):
            await self.PoseDriver.CanineMotor_Receiver(data)
        else: 
            print(f"Bad Canine Command Type Received: Invalid 'Type' missing in {data}") 