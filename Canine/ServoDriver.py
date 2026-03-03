import asyncio

class Servo:
    Default_Max_Speed = 5 #UPDATE WITH SPECS TODO

    Name = None
    Address = None
    Min_Angle_Allowed = None
    Max_Angle_Allowed = None
    Max_Speed_Allowed = None

    Current_Angle = None
    Current_Speed = None

    Path_Desired_Angle = None
    Path_Angle_Upper = None
    Path_Angle_Lower = None
    Path_Positive_Direction = True

    SEQUENCE_MODE = False
    OSCILLATION_MODE = False
    RUNNING = False
    DEFAULT_DELAY = 0.2 # s CHANGE THIS TODO
    SEQUENCE_TIME_STEP = 0.01 # s CHANGE THIS TODO

    Angle_Resolution = 2 #CHANGE THIS TODO

    def __init__(self, Name, Address, MinAngle, MaxAngle, MaxSpeed=Default_Max_Speed):
        self.Name = Name
        self.Address = Address
        self.Min_Angle_Allowed = MinAngle
        self.Max_Angle_Allowed = MaxAngle
        self.Max_Speed_Allowed = MaxSpeed
        self.OSCILLATION_MODE = False
        self.SEQUENCE_MODE = False

    def SetAngle(self, Angle):
        self.Current_Angle = Angle
        self.RUNNING = True
        self.SEQUENCE_MODE = False
        self.OSCILLATION_MODE = False

    def Freeze(self):
        self.RUNNING = False
        self.OSCILLATION_MODE = False

    def MoveWithSpeed(self, Angle, Speed):
        self.Current_Speed = Speed
        self.Current_Angle = Angle
        self.RUNNING = True
        self.SEQUENCE_MODE = True
        self.OSCILLATION_MODE = False

    def OscillateWithSpeed(self, AngleUpper, AngleLower, Speed=Default_Max_Speed, Positive_Direction=True):
        self.Path_Angle_Upper = AngleUpper
        self.Path_Angle_Lower = AngleLower
        self.Current_Speed = Speed 
        self.RUNNING = True
        self.SEQUENCE_MODE = True
        self.OSCILLATION_MODE = True
        self.Path_Positive_Direction = Positive_Direction

    async def ServiceMotors(self):
        while True:
            if(self.RUNNING == True):
                if(self.SEQUENCE_MODE):
                    await self._SequenceRoutine()
                else: 
                    await self._ServoAngle()
                    self.Freeze()
            await asyncio.sleep(self.DEFAULT_DELAY)

    # TAKES INTO ACCOUNT CUSTOM SPEEDS TODO
    async def _SequenceRoutine(self):
        while self.RUNNING and self.SEQUENCE_MODE:
            if(self.OSCILLATION_MODE == True):
                print("OSCILLATING ROUTINE", self.Current_Angle)
            elif(self.OSCILLATION_MODE == False):
                print("ANGLE SEQUENCE ROUTINE", self.Current_Angle)
            await asyncio.sleep(self.SEQUENCE_TIME_STEP)

    def _HandleOscillatingRoutine(self):
        print()
    
    def _HandleSequenceRoutine(self):
        print()

    # FUNCTION TO DETERMINE APPROPRIATE DELAY
    async def _ComputeTimeStep(self):
        print()

    # DRIVE PWM PINS TO OUTPUT TODO
    async def _ServoAngle(self):
        print("SENDING ANGLE ", self.Current_Angle, "OUTPUT TO", self.Name)
        # SPEED COMES IN DELAY + INTERVALS?

async def main():
    # Create a servo instance
    servo1 = Servo("Servo1", 0x01, 0, 180)

    # Test: simple angle command

    # Start the service loop as a background task
    service_task = asyncio.create_task(servo1.ServiceMotors())

    while(True):
        # Let it run for a few iterations
        servo1.SetAngle(90)
        await asyncio.sleep(1)
        servo1.SetAngle(45)
        await asyncio.sleep(1)

        # Test: move with sequence mode
        servo1.MoveWithSpeed(120, 3)

        # Let it run the sequence/oscillation routine for a few iterations
        await asyncio.sleep(1)

        # Test: oscillation mode
        servo1.OscillateWithSpeed(150, 30, Speed=2)

        # Run for a few seconds to observe printouts
        await asyncio.sleep(1)

        servo1.Freeze()

        await asyncio.sleep(5)

    # Stop the servo
    servo1.Freeze()

    # Cancel background task
    service_task.cancel()
    try:
        await service_task
    except asyncio.CancelledError:
        print("Service loop stopped")

# Run the main function
if __name__ == "__main__":
    asyncio.run(main())