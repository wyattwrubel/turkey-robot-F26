from datetime import datetime

# Only import and initialize the kit if not in simulation mode
_kit = None

def _get_kit():
    global _kit
    if _kit is None:
        from adafruit_servokit import ServoKit
        _kit = ServoKit(channels=16)
    return _kit


class Servo:

    # ------------------------------------------------------------------
    # Leg Specifications
    # ------------------------------------------------------------------
    Leg_Motor_Max_Angle = 135
    Leg_Motor_Min_Angle = -135
    Leg_Motor_Max_Pulse = 2500
    Leg_Motor_Min_Pulse = 500

    # Command limits = the safe envelope every source is clamped to (UI sliders, poses, planner).
    # Legs were +-180 here, so a UI slider could send a leg to the motor's end stop (+-135) and
    # past the leg's mechanical range: on 11-17 Rear Right at -135 stalled and the servo rail
    # sagged until no servo responded. W26's poses never go past +-40 (PoseDriver), and W26's
    # original slider range was +-45, so +-45 is known-safe. Widen only after measuring each leg's
    # real mechanical range on the robot. UI sliders mirror these (script.js SERVO_DEFS);
    # tests/check_ui.py fails if the two drift apart or a pose exceeds them.
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
    Channel_Pin = None
    Verbose = False

    def __init__(self, Name, ServoType, Channel_Pin, Offset, IsFlipped, Simulated, Verbose=False):
        self.Name = Name
        self.Channel_Pin = Channel_Pin
        self.Current_Angle = 0
        self.Verbose = Verbose
        self.IsFlipped = IsFlipped
        self.Offset = Offset
        self.Simulated = Simulated

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
            return

        if not self.Simulated:
            kit = _get_kit()
            kit.servo[self.Channel_Pin].set_pulse_width_range(self.Motor_Min_Pulse, self.Motor_Max_Pulse)
            kit.servo[self.Channel_Pin].actuation_range = self.Motor_Max_Angle - self.Motor_Min_Angle

        self.SetAngle(0)

    def SetAngle(self, Angle):
        clamped_angle = max(self.Min_Command_Angle, min(self.Max_Command_Angle, Angle))
        self.Current_Angle = clamped_angle

        if self.Simulated:
            if self.Verbose:
                print(f"[SIM] {self.Name}: angle set to {clamped_angle:.1f}°")
            return

        if self.IsFlipped:
            clamped_angle = -clamped_angle

        effective_angle = clamped_angle + self.Offset
        effective_angle = max(self.Motor_Min_Angle, min(self.Motor_Max_Angle, effective_angle))

        kit_angle = effective_angle - self.Motor_Min_Angle
        _get_kit().servo[self.Channel_Pin].angle = kit_angle

    def close(self):
        if self.Simulated:
            if self.Verbose:
                print(f"[SIM] {self.Name}: close called")
            return
        _get_kit().servo[self.Channel_Pin].angle = 0