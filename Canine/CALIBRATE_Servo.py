import lgpio
import time

class Servo:

    # ------------------------------------------------------------------
    # Leg Specifications
    # ------------------------------------------------------------------
    Leg_Motor_Max_Angle = 74
    Leg_Motor_Min_Angle = -74
    Leg_Motor_Max_Pulse = 2350  # microseconds
    Leg_Motor_Min_Pulse = 850   # microseconds
    Max_Leg_Command = 45
    Min_Leg_Command = -45

    # ------------------------------------------------------------------
    # Head Specifications
    # ------------------------------------------------------------------
    Head_Motor_Max_Angle = 74
    Head_Motor_Min_Angle = -74
    Head_Motor_Max_Pulse = 2350
    Head_Motor_Min_Pulse = 850
    Max_Head_Command = 60
    Min_Head_Command = -60

    Name = None
    Current_Angle = None
    GPIO_PIN = None
    Verbose = False

    def __init__(self, Name, ServoType, GPIO_PIN, Current_Angle, Verbose=False):
        self.Name = Name
        self.GPIO_PIN = GPIO_PIN
        self.Current_Angle = Current_Angle
        self.Verbose = Verbose

        self.handle = lgpio.gpiochip_open(0)
        lgpio.gpio_claim_output(self.handle, self.GPIO_PIN)

        if ServoType == "Leg":
            self.Motor_Max_Angle = self.Leg_Motor_Max_Angle
            self.Motor_Min_Angle = self.Leg_Motor_Min_Angle
            self.Motor_Max_Pulse = self.Leg_Motor_Max_Pulse
            self.Motor_Min_Pulse = self.Leg_Motor_Min_Pulse
            self.Max_Command_Angle = self.Max_Leg_Command
            self.Min_Command_Angle = self.Min_Leg_Command
        elif ServoType == "Head":
            self.Motor_Max_Angle = self.Head_Motor_Max_Angle
            self.Motor_Min_Angle = self.Head_Motor_Min_Angle
            self.Motor_Max_Pulse = self.Head_Motor_Max_Pulse
            self.Motor_Min_Pulse = self.Head_Motor_Min_Pulse
            self.Max_Command_Angle = self.Max_Head_Command
            self.Min_Command_Angle = self.Min_Head_Command
        else:
            print("ServoDriver: Invalid 'ServoType' Received", ServoType)

        self.SetAngle(self.Current_Angle)

    def SetAngle(self, Angle):
        motor_angle = max(self.Min_Command_Angle, min(self.Max_Command_Angle, Angle))
        self.Current_Angle = motor_angle
        t = (motor_angle - self.Motor_Min_Angle) / (self.Motor_Max_Angle - self.Motor_Min_Angle)
        pulse_width = t * (self.Motor_Max_Pulse - self.Motor_Min_Pulse) + self.Motor_Min_Pulse
        self._SendPulse(pulse_width)

    def _SendPulse(self, pulse_width_us):
        """Send a single PWM pulse of the given width in microseconds."""
        if self.Verbose:
            print(f"ServoDriver: {self.Name} Sending Pulsewidth: {pulse_width_us:.1f} µs")

        lgpio.gpio_write(self.handle, self.GPIO_PIN, 1)
        time.sleep(pulse_width_us / 1_000_000)
        lgpio.gpio_write(self.handle, self.GPIO_PIN, 0)

    def close(self):
        lgpio.gpio_write(self.handle, self.GPIO_PIN, 0)
        lgpio.gpiochip_close(self.handle)