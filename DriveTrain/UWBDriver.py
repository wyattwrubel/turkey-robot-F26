import numpy as np
from scipy.optimize import least_squares
import serial

class UWB:
    
    #TODO Update These Locations Inside Zone
    anchors = [
        [12.0, 15.0],
        [50.0, 17.0],
        [0.0, 50.0],
        [50.0, 50.0]
    ]
    
    # TODO Update These With ESP32 Code
    # port = "/dev/ttyACM0"
    port = "/tmp/vserial1"
    baudrate = 9600

    def __init__(self):
        self.UWBSerial = serial.Serial(port = self.port, baudrate=self.baudrate, timeout=1)

    def estimate_UWB_position(self, distances, current_state=None):
        anchor_positions = np.array(self.anchors)

        distances = np.array(distances)

        if current_state is None:
            initial_guess =  np.mean(anchor_positions, axis=0)
        else:
            initial_guess = current_state[0:2]

        # Define residual function
        def residuals(pos):
            x, y = pos
            # Euclidean distance to each anchor minus measured distance
            return np.sqrt((x - anchor_positions[:,0])**2 + (y - anchor_positions[:,1])**2) - distances
        
        # Solve using nonlinear least squares
        result = least_squares(residuals, initial_guess)
        return result.x
    
    def UpdatePosition(self, current_state):
        line = self.UWBSerial.readline().decode('utf-8').strip()  # read one line
        if not line:
            return

        try:
            # Split by commas and convert to float
            parts = line.split(',')
            if len(parts) != 4:
                print("UWBDriver: Unexpected number of distances:", line)
                return
            
            distances = [float(p) for p in parts]

            x, y = self.estimate_UWB_position(distances, current_state)
            return np.array([x, y, current_state[2]], dtype=np.float32)

        except ValueError:
            # happens if the line cannot be converted to floats
            print("UWBDriver: Error parsing line:", line)
            return

            
