import lgpio
import time
from datetime import datetime

class Servo:

    # ------------------------------------------------------------------
    # Leg Specifications
    # ------------------------------------------------------------------
    Leg_Motor_Max_Angle = 130
    Leg_Motor_Min_Angle = -130
    Leg_Motor_Max_Pulse = 2500
    Leg_Motor_Min_Pulse = 500

    Max_Leg_Command = 180
    Min_Leg_Command = -180

    # ------------------------------------------------------------------
    # Head Specifications
    # ------------------------------------------------------------------
    Head_Motor_Max_Angle = 74
    Head_Motor_Min_Angle = -74
    Head_Motor_Max_Pulse = 2350
    Head_Motor_Min_Pulse = 850
    Max_Head_Command = 60
    Min_Head_Command = -60

    SERVO_PERIOD_US = 20_000

    Name = None
    Current_Angle = None
    GPIO_PIN = None
    Verbose = False

    def __init__(self, Name, ServoType, GPIO_PIN, Offset, IsFlipped, Verbose=False, Debug=False):
        self.Name = Name
        self.GPIO_PIN = GPIO_PIN
        self.Current_Angle = 0
        self.Verbose = Verbose
        self.IsFlipped = IsFlipped
        self.Offset = Offset
        self.Debug = Debug

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

        # ------------------------------------------------------------------
        # Debug monitor: claim a second handle on the same pin as input
        # Wire a jumper from GPIO_PIN to DEBUG_MONITOR_PIN to verify pulses
        # ------------------------------------------------------------------
        self.DEBUG_MONITOR_PIN = 16
        self._last_rise_tick = None
        self._debug_pulse_count = 0

        if self.Debug:
            try:
                self.monitor_handle = lgpio.gpiochip_open(0)
                lgpio.gpio_claim_alert(self.monitor_handle, self.DEBUG_MONITOR_PIN, lgpio.BOTH_EDGES)
                lgpio.callback(self.monitor_handle, self.DEBUG_MONITOR_PIN, lgpio.BOTH_EDGES, self._debug_edge_callback)
                print(f"[DEBUG] {self.Name}: Monitoring GPIO{self.DEBUG_MONITOR_PIN} — wire a jumper from GPIO{self.GPIO_PIN} to GPIO{self.DEBUG_MONITOR_PIN}")
            except Exception as e:
                print(f"[DEBUG] {self.Name}: Failed to set up monitor pin: {e}")
                self.Debug = False

        self.SetAngle(0)

    def _debug_edge_callback(self, chip, gpio, level, tick):
        if level == 1:
            self._last_rise_tick = tick
        elif level == 0 and self._last_rise_tick is not None:
            pulse_us = (tick - self._last_rise_tick) & 0xFFFFFFFF
            self._debug_pulse_count += 1
            expected_us = getattr(self, '_last_sent_pulse_us', None)
            error_us = f"  ERROR delta={pulse_us - expected_us:.1f}µs" if expected_us else ""
            print(f"[DEBUG] {self.Name} #{self._debug_pulse_count} received={pulse_us}µs expected={expected_us:.1f}µs{error_us}")
            self._last_rise_tick = None

    def SetAngle(self, Angle):
        clamped_angle = max(self.Min_Command_Angle, min(self.Max_Command_Angle, Angle))
        self.Current_Angle = clamped_angle

        t = (clamped_angle - self.Motor_Min_Angle) / (self.Motor_Max_Angle - self.Motor_Min_Angle)
        pulse_width = t * (self.Motor_Max_Pulse - self.Motor_Min_Pulse) + self.Motor_Min_Pulse

        offset_pulse = self.Offset * (self.Motor_Max_Pulse - self.Motor_Min_Pulse) / (self.Motor_Max_Angle - self.Motor_Min_Angle)
        pulse_width = max(self.Motor_Min_Pulse, min(self.Motor_Max_Pulse, pulse_width + offset_pulse))

        if self.Verbose:
            print(f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] {self.Name} CMD={Angle} t={t:.3f} pw={pulse_width:.1f}µs")

        self._SendPulse(pulse_width)

    def _SendPulse(self, pulse_width_us):
        self._last_sent_pulse_us = pulse_width_us

        t_start = time.perf_counter()
        lgpio.tx_pulse(self.handle, self.GPIO_PIN, int(pulse_width_us), 0, 0, 1)
        time.sleep(pulse_width_us / 1_000_000)
        t_end = time.perf_counter()

        if self.Verbose:
            block_ms = (t_end - t_start) * 1000
            print(f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] {self.Name} pw={pulse_width_us:.1f}µs blocked={block_ms:.2f}ms")

    def close(self):
        lgpio.tx_pulse(self.handle, self.GPIO_PIN, 0, 0, 0, 0)
        lgpio.gpio_write(self.handle, self.GPIO_PIN, 0)
        lgpio.gpiochip_close(self.handle)
        if self.Debug:
            lgpio.gpiochip_close(self.monitor_handle)