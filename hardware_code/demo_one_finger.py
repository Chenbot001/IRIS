from libgx.libgx11 import Hand
from libgx.utils import search_ports
import time


port = search_ports()[0]
 
hand = Hand(port="COM65") # COM* for Windows, ttyACM* or ttyUSB* for Linux
hand.connect()
a=hand.getj()
print(a)

# hand.move_demo(3)
pip = 60
dip = 80
a = [-pip] + [pip+dip] + [0]*9
hand.setj(a)
# hand.grasp()
time.sleep(2)
a=hand.getj()
print(a)
a = [-pip] + [pip] + [0]*9
hand.setj(a)
time.sleep(2)
a=hand.getj()
print(a)
a = [-pip] + [pip+dip+60] + [0]*9
hand.setj(a)
time.sleep(2)
a = [0] + [dip]  + [0]*9
hand.setj(a)
hand.release()



