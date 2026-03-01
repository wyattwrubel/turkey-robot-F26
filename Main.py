import DriveTrain.DriveTrain_Commander as Drive
import asyncio
import numpy as np
import matplotlib.pyplot as plt
import Navigation.TrajectoryTracker as Nav

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
        payload = {"Id": "Nav", "Cmd": CMDData.tolist()}
        await command_publisher.publish(payload)
        await asyncio.sleep(2)
        input.append(CMDData)
        time.append(totaltime)
        totaltime += 2
        
        CMDData = np.array([2, 1.7])
        payload = {"Id": "Nav", "Cmd": CMDData.tolist()}
        await command_publisher.publish(payload)
        input.append(CMDData)
        time.append(totaltime)
        totaltime += 4
        await asyncio.sleep(4)

        CMDData = np.array([-2, 2])
        payload = {"Id": "Nav", "Cmd": CMDData.tolist()}
        await command_publisher.publish(payload)
        input.append(CMDData)
        time.append(totaltime)
        totaltime += 0.3
        await asyncio.sleep(0.3)

        CMDData = np.array([1, 1])
        payload = {"Id": "Nav", "Cmd": CMDData.tolist()}
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
        HUMANDATA = np.array([[7, 7, 0], [12, 13, 0]])
        
        payload = {"Id": "Clear"}
        await VisionObject_Publisher.publish(payload)
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
    await asyncio.sleep(1)
    DESIRED_LOCATION = np.array([17, 5])
    payload = {"Id": "Move", "Cmd": DESIRED_LOCATION.tolist()}
    await TrajectoryCMD_Publisher.publish(payload)

# ----------------- Main -----------------
async def main():
    pose_publisher = AsyncPublisher()
    command_publisher = AsyncPublisher()
    visionobject_publisher = AsyncPublisher()

    TrajectoryCMD_Publisher = AsyncPublisher()

    driver = Drive.MotorDriver(pose_publisher, command_publisher)
    pose_publisher.subscribe(Receiver)
    
    navigator = Nav.TrajectoryTracker(Verbose=True)
    pose_publisher.subscribe(navigator.RobotPose_Receiver)
    visionobject_publisher.subscribe(navigator.VisionObject_Receiver)
    TrajectoryCMD_Publisher.subscribe(navigator.TrajectoryCMD_Receiver)

    navigator.Map.Zone_Map = navigator.Map.Zone_Map[0:15, 0:20]

    await asyncio.gather(
        driver.robotpose_transmitter(),
        driver.increment_dynamics(),
        # Transmit_Orders(command_publisher),
        FakeVision(visionobject_publisher),
        FakeHighLevel(TrajectoryCMD_Publisher),

        # plot_after_delay(10)
    )


if __name__ == "__main__":
    asyncio.run(main())