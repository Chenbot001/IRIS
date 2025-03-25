from libgx.libgx11 import SSRSurgery
import keyboard
import time

# 初始化设备
hand = SSRSurgery(port="COM3")  # 根据实际端口修改
hand.connect()

# 初始化角度和步进值
current_angles = [0, 0]  # 两个电机的当前角度
step = 1  # 每次按键移动的角度

print("q使用方向键控制电机:")
print("上/下键: 控制第一个电机")
print("左/右键: 控制第二个电机")
print("按 'q' 退出程序")

try:
    while True:
        if keyboard.is_pressed('up'):
            current_angles[0] += step
            hand.Safe_control(current_angles)
            time.sleep(0.01)
        
        if keyboard.is_pressed('down'):
            current_angles[0] -= step
            hand.Safe_control(current_angles)
            time.sleep(0.01)

        if keyboard.is_pressed('right'):
            current_angles[1] += step
            hand.Safe_control(current_angles)
            time.sleep(0.01)
            
        if keyboard.is_pressed('left'):
            current_angles[1] -= step
            hand.Safe_control(current_angles)
            time.sleep(0.01)
            
        if keyboard.is_pressed('q'):
            break
            
except KeyboardInterrupt:
    pass
finally:
    # 程序结束前关闭电机
    hand.off()
    print("\n程序已退出")