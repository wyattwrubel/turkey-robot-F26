import asyncio
import can
import struct


class AsyncCANInterface:
    def __init__(self, channel="vcan0"):
        self.bus = can.Bus(channel=channel, interface="socketcan")
        self.reader = can.AsyncBufferedReader()
        self.loop = asyncio.get_running_loop()
        self.notifier = can.Notifier(self.bus, [self.reader], loop=self.loop)
        self._running = True

    async def send_u32(self, arbitration_id: int, value: int):
        payload = value.to_bytes(4, byteorder="little", signed=False)

        msg = can.Message(
            arbitration_id=arbitration_id,
            data=payload,
            is_extended_id=True
        )

        self.bus.send(msg)   # non-blocking in asyncio

    def send_raw(self, arbitration_id: int, data: bytes):
        msg = can.Message(
            arbitration_id=arbitration_id,
            data=data,
            is_extended_id=True
        )

        self.bus.send(msg)

    async def receive_loop(self):
        while self._running:
            msg = await self.reader.get_message()
            if msg is None:
                continue

            self.handle_message(msg)

    async def handle_message(self, msg: can.Message):
        await print(f"RX ID={hex(msg.arbitration_id)} DATA={msg.data.hex()}")

    async def shutdown(self):
        self._running = False
        self.notifier.stop()
        self.bus.shutdown()


async def main():
    can_interface = AsyncCANInterface()

    receive_task = asyncio.create_task(can_interface.receive_loop())

    await can_interface.send_u32(0x205, 123456)

    await asyncio.sleep(1)

    for i in range(5):
        await asyncio.sleep(1)
        await can_interface.send_u32(0x205, 123456)
    
    await can_interface.shutdown()
    await receive_task

if __name__ == "__main__":
    asyncio.run(main())