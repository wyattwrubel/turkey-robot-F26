import DriveTrain.DriveTrain_Commander as Drive
import asyncio
import numpy as np
import matplotlib.pyplot as plt


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
    print("RECEIVED: ", [f"{x:.3f}" for x in ArrayData])
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


async def FakeVision(VisionObject_Publisher):
    while True:
        CMDData = np.array([[5, 5, 0]])
        payload = {"Id": "Turkey", "Cmd": CMDData.tolist()}
        await VisionObject_Publisher.publish(payload)
        # input.append(CMDData)
        # time.append(totaltime)
        # totaltime += 0.3
        await asyncio.sleep(2)

# ----------------- Main -----------------
async def main():
    pose_publisher = AsyncPublisher()
    command_publisher = AsyncPublisher()

    driver = Drive.MotorDriver(pose_publisher, command_publisher)
    pose_publisher.subscribe(Receiver)
    
    await asyncio.gather(
        driver.robotpose_transmitter(),
        driver.increment_dynamics(),
        Transmit_Orders(command_publisher),
        # plot_after_delay(10)
    )


if __name__ == "__main__":
    asyncio.run(main())