from vision import VisionObject
import time

vision = VisionObject()
vision.start()

while True:
    for target in vision.outputs:
        print(target.class_name, target.x_m, target.y_m)
    time.sleep(0.05)