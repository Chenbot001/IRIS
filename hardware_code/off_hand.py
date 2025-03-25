from libgx.libgx11 import SSRHand
from libgx.utils import search_ports
import time

hand = SSRHand(port="COM5")  # COM* for Windows, ttyACM* or ttyUSB* for Linux
hand.connect()


hand.off()


