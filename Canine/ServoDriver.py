from gpiozero import PWMOutputDevice
from gpiozero.pins.lgpio import LGPIOFactory

class Servo:

    #TODO REVISE THESE SPEEDS WITH TESTING
    # ------------------------------------------------------------------
    # Leg Specifications
    # ------------------------------------------------------------------
    Leg_Motor_Max_Angle = 74 # degrees 
    Leg_Motor_Min_Angle = -74 # degrees 
    Leg_Motor_Max_Pulse = 2350 # microseconds 
    Leg_Motor_Min_Pulse = 850 # microseconds 
    Max_Leg_Command = 45 # degrees
    Min_Leg_Command = -45 # degrees

    # ------------------------------------------------------------------
    # Head Specifications
    # ------------------------------------------------------------------
    Head_Motor_Max_Angle = 74 # degrees 
    Head_Motor_Min_Angle = -74 # degrees
    Head_Motor_Max_Pulse = 2350 # microseconds 
    Head_Motor_Min_Pulse = 850 # microseconds 
    Max_Head_Command = 60 # degrees
    Min_Head_Command = -60 # degrees

    Name = None
    Current_Angle = None
    GPIO_PIN = None
    Verbose = False

    def __init__(self, Name, ServoType, GPIO_PIN, Current_Angle, Verbose=False):
        """
        Parameters
        ----------
        Name             : Servo Name, One of 'FrontLeft' 'FrontRight' 'RearRight' 'RearLeft' or 'Head'
        ServoType        : 'Leg' For One of Four Leg Servos or 'Head' For Head Servo
        GPIO_PIN         : Hardware Servo Enable Pin on Raspberry Pin
        Max_Command_Angle: Max Allowed Angle Command. If Above Will Send This Command
        Min_Command_Angle: Min Allowed Angle Command If Below Will Send This Command
        Verbose          : Optional Parameters to Enable Debugging Printouts
        """

        self.Name = Name
        self.GPIO_PIN = GPIO_PIN
        self.Current_Angle = Current_Angle
        self.Verbose = Verbose
        factory = LGPIOFactory()
        self.pwm = PWMOutputDevice(GPIO_PIN, frequency=50, pin_factory=factory)

        if(ServoType == "Leg"):
            self.Motor_Max_Angle = self.Leg_Motor_Max_Angle
            self.Motor_Min_Angle = self.Leg_Motor_Min_Angle
            self.Motor_Max_Pulse = self.Leg_Motor_Max_Pulse
            self.Motor_Min_Pulse = self.Leg_Motor_Min_Pulse
            self.Max_Command_Angle = self.Max_Leg_Command
            self.Min_Command_Angle = self.Min_Leg_Command
        elif(ServoType == "Head"):
            self.Motor_Max_Angle = self.Head_Motor_Max_Angle
            self.Motor_Min_Angle = self.Head_Motor_Min_Angle
            self.Motor_Max_Pulse = self.Head_Motor_Max_Pulse
            self.Motor_Min_Pulse = self.Head_Motor_Min_Pulse
            self.Max_Command_Angle = self.Max_Head_Command
            self.Min_Command_Angle = self.Min_Head_Command
        else:
            print("ServoDriver: Invalid 'ServoType' Received ", ServoType)

    def SetAngle(self, Angle):
        # TODO Flip orientation of Left or Right motors
        
        motor_angle = max(self.Min_Command_Angle, min(self.Max_Command_Angle, Angle))
        self.Current_Angle = motor_angle
        t = (motor_angle - self.Motor_Min_Angle) / (self.Motor_Max_Angle - self.Motor_Min_Angle)
        pulse_width = t * (self.Motor_Max_Pulse - self.Motor_Min_Pulse) + self.Motor_Min_Pulse
        self._SendPulse(pulse_width)

    def _SendPulse(self,pulse_width):
        if self.Verbose: print("ServoDriver: ", self.Name, " Sending Pulsewidth: ", pulse_width)
        duty_cycle = pulse_width / 20000
        self.pwm.value = duty_cycle