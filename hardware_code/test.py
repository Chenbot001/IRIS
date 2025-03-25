from libgx.libgx11 import SSRHand
from libgx.utils import search_ports
import time

hand = SSRHand(port="COM5")  # COM* for Windows, ttyACM* or ttyUSB* for Linux
hand.connect()

angle = [0, 0, 0,
          0, 0, 0,
            0, 0, 0,
              0,0, 0, 
              0, 0, 0]

# angle = [0, 0, 80,
#           80, 80, 80,
#             0, 0, 0,
#               0,0, 0, 
#               40, 40, 0]

hand.Safe_control(angle)
time.sleep(2)

print(hand.get_angle())

