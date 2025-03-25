from libgx.libgx11 import SSRHand
from libgx.utils import search_ports
import time

hand = SSRHand(port="COM5")  # COM* for Windows, ttyACM* or ttyUSB* for Linux
hand.connect()

# dof_angle = [0, 0, 0, 0, 0, 0, 0, 80, 0, 0, 80, 0, 0, 80, 0]
             
dof_angle = [0, 60, 0,    # 小指侧摆（左摆为正，右摆为负）、小指PIP内收、小指MCP内收
             0, 60, 0,    # 无名指侧摆（左摆为正，右摆为负）、无名指PIP内收、无名指MCP内收
             0, 60, 0,    # 中指侧摆（左摆为正，右摆为负）、中指PIP内收、中指MCP内收
             0, 60, 0,    # 食指侧摆（左摆为正，右摆为负）、食指PIP内收、食指MCP内收
             0, 0, 0]  # 拇指侧摆（左摆为正，右摆为负）、拇指PIP内收、拇指MCP内收

'''
dof_angles 是一个包含15个自由度的角度数组，顺序是：

小指侧摆（左摆为正，右摆为负）
小指PIP内收
小指MCP内收
无名指侧摆（左摆为正，右摆为负）
无名指PIP内收
无名指MCP内收
中指侧摆（左摆为正，右摆为负）
中指PIP内收
中指MCP内收
食指侧摆（左摆为正，右摆为负）
食指PIP内收
食指MCP内收
拇指侧摆（左摆为正，右摆为负）
拇指PIP内收
拇指MCP内收
'''

angle = hand.convert_dof_to_motor_angle(dof_angle)
print(angle)

hand.Safe_control(angle)
time.sleep(2)

print(hand.get_angle())

