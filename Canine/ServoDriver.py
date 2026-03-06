# import pigpio

class Servo:

    # TODO Revise With Team 19
    # Specifications for Head and Leg Servo Motors
    Leg_Max_Motor_Speed = 498 # degree/s 
    Leg_Motor_Max_Angle = 74 # degrees 
    Leg_Motor_Min_Angle = -74 # degrees 
    Leg_Motor_Max_Pulse = 2350 # microseconds 
    Leg_Motor_Min_Pulse = 850 # microseconds 

    Head_Max_Motor_Speed = 498 # degree/s 
    Head_Motor_Max_Angle = 74 # degrees 
    Head_Motor_Min_Angle = -74 # degrees
    Head_Motor_Max_Pulse = 2350 # microseconds 
    Head_Motor_Min_Pulse = 850 # microseconds 

    Name = None
    Current_Angle = None
    GPIO_PIN = None
    Verbose = False

    # TODO change this interfacce to account for min and max angles if Team 19 needs
    def __init__(self, Name, ServoType, GPIO_PIN, Verbose=False):
        """
        Parameters
        ----------
        Name      : Servo Name, One of 'FrontLeft' 'FrontRight' 'RearRight' 'RearLeft' or 'Head'
        ServoType : 'Leg' For One of Four Leg Servos or 'Head' For Head Servo
        GPIO_PIN  : Hardware Servo Enable Pin on Raspberry Pin
        Verbose   : Optional Parameters to Enable Debugging Printouts
        """

        self.Name = Name
        self.GPIO_PIN = GPIO_PIN
        self.Verbose = Verbose

        if(ServoType == "Leg"):
            self.Max_Motor_Speed = self.Leg_Max_Motor_Speed
            self.Motor_Max_Angle = self.Leg_Motor_Max_Angle
            self.Motor_Min_Angle = self.Leg_Motor_Min_Angle
            self.Motor_Max_Pulse = self.Leg_Motor_Max_Pulse
            self.Motor_Min_Pulse = self.Leg_Motor_Min_Pulse
        elif(ServoType == "Head"):
            self.Max_Motor_Speed = self.Head_Max_Motor_Speed
            self.Motor_Max_Angle = self.Head_Motor_Max_Angle
            self.Motor_Min_Angle = self.Head_Motor_Min_Angle
            self.Motor_Max_Pulse = self.Head_Motor_Max_Pulse
            self.Motor_Min_Pulse = self.Head_Motor_Min_Pulse
        else:
            print("ServoDriver: Invalid 'ServoType' Received ", ServoType)

    # TODO validate that this works
    def SetAngle(self, Angle):
        motor_angle = max(self.Motor_Min_Angle, min(self.Motor_Max_Angle, Angle))
        pulse_width = (motor_angle + self.Motor_Max_Angle) / (self.Motor_Max_Angle - self.Motor_Min_Angle)
        pulse_width = pulse_width * (self.Motor_Max_Pulse - self.Motor_Min_Pulse) + self.Motor_Min_Pulse
        self._SendPulse(pulse_width)

    # TODO On raspberry pi | Establish PWMs and I2C connections
    def _SendPulse(self,pulse_width):
        if self.Verbose: print("ServoDriver: ", self.Name, " Sending Pulsewidth: ", pulse_width)
        # self._pi.set_servo_pulsewidth(self.gpio_pin, pulse_width)
            