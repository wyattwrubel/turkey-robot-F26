# Pi system configuration (copied off the robot 2026-11-05)

These files live outside this repo on the robot's Pi. They're copied here so they're versioned
next to the code that depends on them. Changing a copy here changes nothing on the Pi; installing
one is a deliberate step.

| File here | Lives on the Pi at | Why it matters |
|---|---|---|
| `udev/99-motor-drivers.rules` | `/etc/udev/rules.d/` | Names the RoboClaws `ttyMotor0`–`3` by **physical hub port** (`3-1.1` … `3-1.4`). Swapping two RoboClaw cables on the hub swaps which wheel gets which command. A different hub changes the paths. |
| `udev/99-usb-serial.rules` | `/etc/udev/rules.d/` | `/dev/ttyGPS` = the Prolific PL2303 GPS puck (067b:23a3, serial `A;AUb144J08`) |
| `udev/99-usb-devices.rules` | `/etc/udev/rules.d/` | `/dev/video_camera` = the webcam, only in Pi USB port `3-2` |
| `udev/99-hailo.rules` | `/etc/udev/rules.d/` | Keeps the Hailo AI Hat's PCIe link powered |
| `udev/99-rpi-keyboard.rules` | `/etc/udev/rules.d/` | Stock Raspberry Pi OS file |
| `config.txt` | `/boot/firmware/` | I2C bus 1 on (PCA9685 servo driver, 0x40) and a second I2C bus on GPIO 14/15 (`i2c3`, BNO055 IMU, 0x28); PCIe Gen 2 for the AI Hat |
| `gpsd` | `/etc/default/` | `USBAUTO="true"`: gpsd grabs USB GPS receivers on plug-in. Turn it off before adding the u-blox F9P |
| `bashrc` | `~/.bashrc` | Starts a virtual display (`Xvfb :99`) on SSH logins |
| `pip-freeze-pi.txt` | (from `UserInterface/.venv`) | Exact Python packages on the Pi, Python 3.13.5. Includes ~20 `nvidia-*` CUDA packages that do nothing on a Pi |

Pi facts: Raspberry Pi 5 Model B Rev 1.1, Debian 13 (trixie), kernel 6.12.75, hostname
`ME450TurkeySquad`, user `turkeysquad`, Ethernet 192.168.10.2.
