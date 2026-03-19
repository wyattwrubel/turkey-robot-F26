import sys
import os
sys.path.append(os.path.dirname(__file__))

import PoseDriver
import AudioDriver

class CanineDriver:
    
    UI_lock = False

    def __init__(self, ServoAnglePublisher, Verbose=False):
        self.Audio = AudioDriver.AsyncAudioDriver(Verbose)
        self.PoseDriver = PoseDriver.PoseDriver(ServoAnglePublisher, Verbose)

    async def CanineDriver_Receiver(self, data):
        """ Receiver For All Canine Related Commands. See Protocol #2 For More Details """
    
        try:
            cmd_id = data["Id"]
        except KeyError:
           print(f"CanineDriver: 'Id' key missing in {data}")

        if(cmd_id == "UnlockUI"):
            self.UI_lock = False
        elif(cmd_id == "UI"):
            self.UI_lock = True
            await self._Process_CanineData(data)
        elif(cmd_id == "System"):
            if(self.UI_lock == False):
                await self._Process_CanineData(data)
        else:
            print(f"CanineDriver: Invalid 'Id' in {data}") 

    async def _Process_CanineData(self, data):
        """  Handles audio, poses, and raw motor commands """

        try:
            cmd_type = data["Type"]
            cmd_data = data["Data"]
        except KeyError:
           print(f"CanineDriver: 'Type' or 'Data' key missing in {data}") 

        if(cmd_type == "Audio" and cmd_data in self.Audio.sounds):
            self.Audio.play(cmd_data)
        elif(cmd_type == "Servo" or cmd_type == "Pose"):
            await self.PoseDriver.CanineMotor_Receiver(data)
        else: 
            print(f"CanineDriver: Invalid 'Type' or 'Data' in {data}") 