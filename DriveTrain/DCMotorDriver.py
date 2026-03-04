import numpy as np
import asyncio

class DC_Driver:
    
    def __init__(self):
        self.MotorSpeeds = np.array([0,0], dtype=np.float32)
        self.CurrentState = np.array([2, 2, 0], dtype=np.float32)
        # self.RC = RoboClaw(port_name="/dev/ttyUSB0", baud_rate=115200, timeout=20, retries=128, address=128) #UPDATE THIS TODO
        # self.RC.Open()

        self.Motors = {}
        self.Motors["FrontLeft"] = {0x81,0} # UPDATE THIS TODO
        self.Motors["FrontRight"] = {0x82,0} # UPDATE THIS TODO
        self.Motors["RearRight"] = {0x83,0} # UPDATE THIS TODO
        self.Motors["RearLeft"] = {0x84,0} # UPDATE THIS TODO
    
        self.MaxMotorSpeed = 3 # m/s UPDATE WITH REAL TODO

    async def _ReadEncoderValues(self):
        for Motors in self.Motors:2

async def main():
    myDC_Driver = DC_Driver()
    await myDC_Driver._ReadEncoderValues()


if __name__ == "__main__":
    asyncio.run(main())