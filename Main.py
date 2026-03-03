import DriveTrain.DriveTrainDriver as Drive
import asyncio
import numpy as np
import matplotlib.pyplot as plt
import Navigation.TrajectoryTracker as Nav
import Canine.CanineDriver as K9
import time

# ----------------- Async Publisher -----------------
class AsyncPublisher:
    def __init__(self):
        self._subscribers = []  # fixed typo
    def subscribe(self, callback):
        self._subscribers.append(callback)

    async def publish(self, data):
        for callback in self._subscribers:
            try:
                await callback(data)
            except Exception as e: 
                print(f"Subscriber error: {e}")


# ----------------- Global Data -----------------
output = [np.array([0,0,0], dtype=np.float32)]
time = []
totaltime = 0
input = []


# ----------------- Receiver -----------------
async def Receiver(data):
    global totaltime
    ArrayData = np.array(data, dtype=np.float32)
    # print("RECEIVED: ", [f"{x:.3f}" for x in ArrayData])
    output.append(ArrayData)


# ----------------- Transmit Orders -----------------
async def Transmit_Orders(command_publisher):
    global totaltime
    while True:
        CMDData = np.array([2, 2])
        payload = {"Id": "System", "Cmd": CMDData.tolist()}
        await command_publisher.publish(payload)
        await asyncio.sleep(2)
        input.append(CMDData)
        time.append(totaltime)
        totaltime += 2
        
        CMDData = np.array([2, 1.7])
        payload = {"Id": "System", "Cmd": CMDData.tolist()}
        await command_publisher.publish(payload)
        input.append(CMDData)
        time.append(totaltime)
        totaltime += 4
        await asyncio.sleep(4)

        CMDData = np.array([-2, 2])
        payload = {"Id": "System", "Cmd": CMDData.tolist()}
        await command_publisher.publish(payload)
        input.append(CMDData)
        time.append(totaltime)
        totaltime += 0.3
        await asyncio.sleep(0.3)

        CMDData = np.array([1, 1])
        payload = {"Id": "System", "Cmd": CMDData.tolist()}
        await command_publisher.publish(payload)
        input.append(CMDData)
        time.append(totaltime)
        totaltime += 1.7
        await asyncio.sleep(1.7)


# ----------------- Plot after delay -----------------
async def plot_after_delay(delay=10):
    await asyncio.sleep(delay)
    
    output_array = np.array(output)
    input_array = np.array(input)  # convert input list to array

    plt.figure(figsize=(10, 4))
    
    # Trajectory
    plt.subplot(1, 2, 1)
    plt.plot(output_array[:, 0], output_array[:, 1], marker='o')
    plt.xlabel("X position [m]")
    plt.ylabel("Y position [m]")
    plt.title("Robot Trajectory")
    plt.axis("equal")
    
    # Motor inputs over time
    plt.subplot(1, 2, 2)
    plt.plot(time, input_array[:, 0], marker='x', label="Left Motor")
    plt.plot(time, input_array[:, 1], marker='o', label="Right Motor")
    plt.xlabel("Time [s]")
    plt.ylabel("Motor Command")
    plt.title("Motor Commands over Time")
    plt.legend()
    
    plt.tight_layout()
    plt.show()

async def TurkeyChase(VisionObject_Publisher):
        
        await asyncio.sleep(0.01)
        TURKEYDATA = np.array([[4, 4, 0]])
        HUMANDATA = np.empty((0, 3))
        
        payload = {"Id": "DynamicObjects", "Turkeys": TURKEYDATA.tolist(), "Humans": HUMANDATA.tolist()}
        await VisionObject_Publisher.publish(payload)
        await asyncio.sleep(0.5)
      
        TURKEYDATA = np.array([[4, 5, 0]])
        HUMANDATA = np.empty((0, 3))
        
        payload = {"Id": "DynamicObjects", "Turkeys": TURKEYDATA.tolist(), "Humans": HUMANDATA.tolist()}
        await VisionObject_Publisher.publish(payload)
        await asyncio.sleep(0.5)
        TURKEYDATA = np.array([[5, 5, 0]])
        HUMANDATA = np.empty((0, 3))
        
        payload = {"Id": "DynamicObjects", "Turkeys": TURKEYDATA.tolist(), "Humans": HUMANDATA.tolist()}
        await VisionObject_Publisher.publish(payload)
        await asyncio.sleep(0.5)

        TURKEYDATA = np.array([[6, 5, 0]])
        HUMANDATA = np.empty((0, 3))
        payload = {"Id": "DynamicObjects", "Turkeys": TURKEYDATA.tolist(), "Humans": HUMANDATA.tolist()}
        await VisionObject_Publisher.publish(payload)
        await asyncio.sleep(0.5)
        
        for i in range(6,16):
            TURKEYDATA = np.array([[6, i, 0]])
            HUMANDATA = np.empty((0, 3))
            payload = {"Id": "DynamicObjects", "Turkeys": TURKEYDATA.tolist(), "Humans": HUMANDATA.tolist()}
            await VisionObject_Publisher.publish(payload)
            await asyncio.sleep(0.5)

        await asyncio.sleep(0.5)

async def FakeVision(VisionObject_Publisher):
        TURKEYDATA = np.array([[5, 5, 0], [10, 4, 0]])
        HUMANDATA = np.array([[7, 7, 0], [12, 9, 0]])
        
        payload = {"Id": "DynamicObjects", "Turkeys": TURKEYDATA.tolist(), "Humans": HUMANDATA.tolist()}
        await VisionObject_Publisher.publish(payload)
    
    # while True:
    #     await TurkeyChase(VisionObject_Publisher)
        # CMDData = np.array([5, 5, 0])
        # payload = {"Id": "Turkey", "Cmd": CMDData.tolist()}
        # await VisionObject_Publisher.publish(payload)
        # await asyncio.sleep(2)

        ################
        # TURKEYDATA = np.array([[5, 5, 0], [10, 4, 0]])
        # HUMANDATA = np.array([[7, 7, 0], [10, 13, 0]])
        
        # payload = {"Id": "Clear"}
        # await VisionObject_Publisher.publish(payload)
        # payload = {"Id": "DynamicObjects", "Turkeys": TURKEYDATA.tolist(), "Humans": HUMANDATA.tolist()}
        # await VisionObject_Publisher.publish(payload)

        # await asyncio.sleep(2)

        # payload = {"Id": "Clear"}
        # await VisionObject_Publisher.publish(payload)
        # TURKEYDATA = np.array([[5, 5, 0]])
        # HUMANDATA = np.empty((0, 3))
        # payload = {"Id": "DynamicObjects", "Turkeys": TURKEYDATA.tolist(), "Humans": HUMANDATA.tolist()}
        # await VisionObject_Publisher.publish(payload)
        ##############

async def FakeHighLevel(TrajectoryCMD_Publisher):
    while True:
        await asyncio.sleep(1)
        DESIRED_LOCATION = np.array([17, 5])
        payload = {"Id": "Move", "Cmd": DESIRED_LOCATION.tolist()}
        await TrajectoryCMD_Publisher.publish(payload)

        await asyncio.sleep(1)
        DESIRED_LOCATION = np.array([17, 5, np.pi/2])
        payload = {"Id": "Orient", "Cmd": DESIRED_LOCATION.tolist()}
        await TrajectoryCMD_Publisher.publish(payload)

async def K9_tester(K9_pub):
    # payload = {"Id": "System", "Type": "Audio", "Data": "howl"}
    # await K9_pub.publish(payload)
    # await asyncio.sleep(2)

    payload = {"Id": "System", "Type": "Servo", "Name": "FrontLeft", "Data": 75}
    await K9_pub.publish(payload)
    await asyncio.sleep(2)

    payload = {"Id": "System", "Type": "Pose", "Data": "Pose2"}
    await K9_pub.publish(payload)
    await asyncio.sleep(2)

    print("WAVE /////////////////////////")
    payload = {"Id": "System", "Type": "Servo", "Name": "FrontLeft", "Data": 0}
    await K9_pub.publish(payload)
    payload = {"Id": "System", "Type": "Servo", "Name": "FrontRight", "Data": 10}
    await K9_pub.publish(payload)
    payload = {"Id": "System", "Type": "Servo", "Name": "RearLeft", "Data": 20}
    await K9_pub.publish(payload)
    payload = {"Id": "System", "Type": "Servo", "Name": "RearRight", "Data": 30}
    await K9_pub.publish(payload)
    
    await asyncio.sleep(2)


    payload = {"Id": "System", "Type": "Pose", "Data": "Pose1"}
    await K9_pub.publish(payload)
    await asyncio.sleep(2)

    # print("System HOWL")
    # payload = {"Id": "System", "Type": "Audio", "Data": "howl"}
    # await K9_pub.publish(payload)
    # await asyncio.sleep(2)

    # print("UI HOWL")
    # payload = {"Id": "UI", "Type": "Audio", "Data": "howl"}
    # await K9_pub.publish(payload)
    # await asyncio.sleep(2)

    # print("System HOWL")
    # payload = {"Id": "System", "Type": "Audio", "Data": "howl"}
    # await K9_pub.publish(payload)
    # await asyncio.sleep(2)

    # print("UI HOWL")
    # payload = {"Id": "UI", "Type": "Audio", "Data": "howl"}
    # await K9_pub.publish(payload)
    # await asyncio.sleep(2)

    # print("System UNLOCK")
    # payload = {"Id": "UnlockUI"}
    # await K9_pub.publish(payload)
    # await asyncio.sleep(2)

    # print("System HOWL")
    # payload = {"Id": "System", "Type": "Audio", "Data": "howl"}
    # await K9_pub.publish(payload)
    # await asyncio.sleep(2)

# ----------------- Main -----------------
async def main():

    pose_publisher = AsyncPublisher()
    command_publisher = AsyncPublisher()
    visionobject_publisher = AsyncPublisher()
    TrajectoryCMD_Publisher = AsyncPublisher()

    driver = Drive.MotorDriver(pose_publisher, command_publisher)
    pose_publisher.subscribe(Receiver)
    
    navigator = Nav.TrajectoryTracker(Verbose=False)
    pose_publisher.subscribe(navigator.RobotPose_Receiver)
    visionobject_publisher.subscribe(navigator.VisionObject_Receiver)
    TrajectoryCMD_Publisher.subscribe(navigator.TrajectoryCMD_Receiver)

    navigator.Map.Zone_Map = navigator.Map.Zone_Map[0:15, 0:20]

    Canine = K9.CanineDriver()
    K9_publisher = AsyncPublisher()
    K9_publisher.subscribe(Canine.CanineCommand_Receiver)

    await asyncio.gather(
        driver.robotpose_transmitter(),
        driver.increment_dynamics(),
        Transmit_Orders(command_publisher),
        FakeVision(visionobject_publisher),
        FakeHighLevel(TrajectoryCMD_Publisher),
        navigator.MPC.DriveTrain_Sender(),
        K9_tester(K9_publisher),
        # plot_after_delay(10)
    )


if __name__ == "__main__":
    asyncio.run(main())