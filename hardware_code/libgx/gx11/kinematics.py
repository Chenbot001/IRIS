import ikpy.chain
import os.path as osp
import numpy as np
abs_path = osp.dirname(osp.abspath(__file__))



class KinGX11:
    def __init__(self) -> None:
        self.name = 'GX11'
        self.chain_finger2 = ikpy.chain.Chain.from_urdf_file(osp.join(abs_path, 'urdf/finger2.urdf'))

        self.direction_finger2 = [1, -1, -1, -1]

    def fk_finger2(self, q=[0]*4):
        """
        finger2 正运动学，4自由度
        """
        # 匹配关节方向，degree to rad
        q = [q_*d*np.pi/180 for q_, d in zip(q, self.direction_finger2)]

        ee_frame = self.chain_finger2.forward_kinematics([0]+q+[0])
        ee_pos = ee_frame[:3, -1] # 笛卡尔坐标

        return ee_pos
    
    def ik_finger2(self, xyz):
        q = self.chain_finger2.inverse_kinematics(xyz)

        q = q[1:-1]
        # 匹配关节方向，rad to degree
        q = [q_*d*180/np.pi for q_, d in zip(q, self.direction_finger2)]
        
        return q

if __name__ == "__main__":
    kin = KinGX11()
