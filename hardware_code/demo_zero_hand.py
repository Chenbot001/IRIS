from libgx.libgx11 import SSRSurgery
from libgx.utils import search_ports
import time

hand = SSRSurgery(port="COM3")  # COM* for Windows, ttyACM* or ttyUSB* for Linux
hand.connect()
hand.off()
hand.set_all_zero()
time.sleep(2)
print(hand.get_angle())
print("Set Whole Hand Zero Success！！")

