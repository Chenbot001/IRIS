import time
from .dynamixel_sdk import PortHandler, PacketHandler
from .motor import Motor
import sys
import numpy as np
from .config import BAUDRATE, PROTOCOL_VERSION
from .gx11 import kinematics

class SSRSurgery:

    def __init__(self, port) -> None:
        self.is_connected = False
        self.port = port
        self.name = 'SSR-Surgery'
        # self.kin = kinematics.KinGX11()

    def connect(self):
        """
        连接Hand，并且使能每个电机为默认的力控位置模式
        """

        portHandler = PortHandler(self.port)
        packetHandler = PacketHandler(PROTOCOL_VERSION)

        if portHandler.openPort() and portHandler.setBaudRate(BAUDRATE):
            print(f'Open {self.port} Success...')
            self.is_connected = True
        else:
            print(f'Failed...')
            self.is_connected = False
            sys.exit(0)

        self.portHandler = portHandler
        self.packetHandler = packetHandler

        self.motors = [Motor(i + 1, portHandler, packetHandler) for i in range(2)]

        for m in self.motors:
            m.init_config()

        print(f'{self.name} init done...')

    def off(self):
        """
        失能所有电机
        """
        for m in self.motors:
            m.torq_off()

    def on(self):
        """
        使能所有电机
        """
        for m in self.motors:
            m.torq_on()

    def getj(self):
        """
        获取ISR关节角度，单位度
        """
        js = [m.get_pos() for m in self.motors]
        return js

    def setj(self, js):
        """
        设置ISR关节角度，单位度
        """
        for m, j in zip(self.motors, js):
            m.set_pos(j)

    def set_all_zero(self):
        """
        将所有电机的角度设置为零
        """
        for m in self.motors:
            m.set_zero()

    def Safe_control(self, angle):
        """
        直接设置电机角度，不进行安全限制
        """
        self.setj(angle)

        # # 定义安全范围
        # valid_ranges = [
        #     (0, 180),  # 第1个电机的角度范围
        #     (0, 180),  # 第2个电机的角度范围
        # ]

        # # 初始化上一个有效角度的存储（与电机数量相同）
        # last_valid_angles = [None] * len(self.motors)

        # # 获取当前电机的角度
        # current_angles = self.getj()

        # # 更新角度并检查是否在有效范围内
        # for i, current_angle in enumerate(current_angles):
        #     # 检查角度是否在预定义的有效范围内
        #     if valid_ranges[i][0] <= angle[i] <= valid_ranges[i][1]:
        #         # 如果在有效范围内，更新上一次有效角度，并设置当前角度
        #         last_valid_angles[i] = angle[i]
        #         current_angles[i] = angle[i]
        #     else:
        #         # 如果不在有效范围内，使用上一次有效角度，如果没有则使用当前角度
        #         current_angles[i] = last_valid_angles[i] if last_valid_angles[i] is not None else current_angle

        # # 设置更新后的角度
        # self.setj(current_angles)

    def get_angle(self):
      
        js = [m.get_pos() for m in self.motors]

        return js

    def hand_demo(self):
        # 初始位置（全零位置）
        a = [0] * 2
        
        # 第一个动作：两个电机分别转到40度和90度
        angle = [40, 90]
        self.setj(angle)
        time.sleep(2)
        self.setj(a)  # 回到零位
        
        # 第二个动作：第一个电机不动，第二个电机转到40度
        angle = [0, 40]
        self.setj(angle)
        time.sleep(2)
        self.setj(a)  # 回到零位
        
        # 第三个动作：两个电机分别转到-40度和130度
        angle = [-40, 130]
        self.setj(angle)
        time.sleep(2)
        self.setj(a)  # 回到零位
        time.sleep(1)

   

    # def convert_dof_to_motor_angle(self, dof_angles):
    #     """
    #     将自由度的期望角度转换为电机驱动的角度。
    #     :param dof_angles: 各自由度期望的角度变化，长度为15的列表
    #     :return: 对应电机的角度列表，用于控制电机驱动绳子
    #     """
        
    #     motor_angles = [0] * 15  # 初始化电机角度列表
        
    #     # 小指部分
    #         # 侧摆
    #     motor_angles[0] = dof_angles[0]  # 小指左摆
    #     motor_angles[1] = -dof_angles[0]  # 小指右摆（与左摆角度相反）
    #         # PIP内收
    #     motor_angles[14] = dof_angles[1]  # 小指弯曲
    #         # MCP内收
    #     motor_angles[0] = motor_angles[0] + dof_angles[2]
    #     motor_angles[1] = motor_angles[1] + dof_angles[2]

    #     # 无名指部分
    #         #侧摆
    #     motor_angles[2] = dof_angles[3]  # 无名指左摆
    #     motor_angles[3] = -dof_angles[3]  # 无名指右摆（与左摆角度相反）
    #         #PIP内收
    #     motor_angles[13] = dof_angles[4]  # 无名指弯曲
    #         #MCP内收
    #     motor_angles[2] = motor_angles[2] + dof_angles[5]
    #     motor_angles[3] = motor_angles[3] + dof_angles[5]

    #     # 中指部分
    #         #侧摆
    #     motor_angles[4] = dof_angles[6]  # 中指左摆
    #     motor_angles[5] = -dof_angles[6]  # 中指右摆（与左摆角度相反）
    #         #PIP内收
    #     motor_angles[12] = dof_angles[7]  # 中指弯曲
    #         #MCP内收
    #     motor_angles[4] = motor_angles[4] + dof_angles[8]
    #     motor_angles[5] = motor_angles[5] + dof_angles[8]

    #     # 食指部分
    #         #侧摆
    #     motor_angles[7] = dof_angles[9]  # 食指左摆
    #     motor_angles[8] = -dof_angles[9]  # 食指右摆（与左摆角度相反）
    #         #PIP内收
    #     motor_angles[11] = dof_angles[10]  # 食指弯曲
    #         #MCP内收
    #     motor_angles[7] = motor_angles[7] + dof_angles[11]
    #     motor_angles[8] = motor_angles[8] + dof_angles[11]

    #     # 拇指部分
    #         #侧摆
    #     motor_angles[9] = dof_angles[12]  # 拇指左摆
    #     motor_angles[6] = -dof_angles[12]  # 拇指右摆（与左摆角度相反）
    #         #PIP内收
    #     motor_angles[10] = dof_angles[13]  # 拇指弯曲
    #         #MCP内收
    #     motor_angles[9] = motor_angles[9] + dof_angles[14]
    #     motor_angles[6] = motor_angles[6] + dof_angles[14]


    #     return motor_angles

