from libgx.libgx11 import SSRHand
from libgx.utils import search_ports
import time

hand = SSRHand(port="COM65")  # COM* for Windows, ttyACM* or ttyUSB* for Linux
hand.connect()
a = [0] * 11
hand.setj(a)
time.sleep(2)
print(hand.getj())

"""
0 75
-10 90
0 105
0 90

0 -30
0 70
0 -130

0 -70
0 180

0 -70 
0 180

"""
hand.hand_demo()
