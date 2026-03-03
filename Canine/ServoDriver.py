# import pigpio

class Servo:
    # MOTOR SPECIFICATIONS
    Motor_Speed = 498 #degree/s TODO Revise based on inertia calculations
    Motor_Max_Angle = 74 #degrees TODO Revise based on Team 19 Needs
    Motor_Min_Angle = -74 #degrees TODO Revise based on Team 19 Needs
    Motor_Min_Pulse = 850 # microseconds TODO Team 19
    Motor_Max_Pulse = 2350 #microseconds TODO Team 19

    Current_Angle = None
    Address = None # Address for I2C TODO figure out hardware needs of PWM expander
    I2C_Pin = None # TODO once wiring finalized

    Verbose = False

    # TODO change this interfacce to account for min and max angles if Team 19 needs
    def __init__(self, I2C_Pin,  Address, Verbose=False):
        self.I2C_Pin = I2C_Pin
        self.Address = Address
        self.Verbose = Verbose

    # TODO validate that this works
    def SetAngle(self, Angle):
        motor_angle = max(self.Motor_Min_Angle, min(self.Motor_Max_Angle, Angle))
        pulse_width = (motor_angle + self.Motor_Max_Angle) / (self.Motor_Max_Angle - self.Motor_Min_Angle)
        pulse_width = pulse_width * (self.Motor_Max_Pulse - self.Motor_Min_Pulse) + self.Motor_Min_Pulse
        self._SendPulse(pulse_width)

    # TODO On raspberry pi | Establish PWMs and I2C connections
    def _SendPulse(self,pulse_width):
        if self.Verbose: print("SENDING PULSE OF WIDTH IN MICROSECONDS", pulse_width)
        # self._pi.set_servo_pulsewidth(self.gpio_pin, pulse_width)
            