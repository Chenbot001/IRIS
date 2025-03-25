import sys
import pyqtgraph as pg
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QObject, pyqtSignal, QThread
import serial
import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter
from scipy.interpolate import griddata

# 串口与数据帧头帧尾
serialPort = 'COM69'
baudRate = 921600
FRAME_START = b'\xAA\xBB'
FRAME_END = b'\xCC\xDD'

def read_data_array(mSerial):
    frame_data = bytearray()
    '''循环读取帧头'''
    while True:
        print("帧头读取中...")
        if mSerial.read(2) == FRAME_START:
            # print("帧头读取到")
            break
    ''' 读取固定长度的帧数据'''
    frame_data.extend(mSerial.read(200))  # 减去帧头的长度

    '''读取帧尾'''
    frame_end = mSerial.read(2)
    # print("帧尾是：",frame_end)
    if frame_end != FRAME_END:
        # print("帧尾错误！")
        return None
    '''数据提取'''
    data_array = np.zeros(100, dtype=np.uint16)
    # 迭代原始数据帧，每两个字节为一组，组合成一个新的数据
    for i in range(0, len(frame_data), 2):
        # 将每两个字节组合成一个数据，并将其添加到新的数据帧中
        new_data = (frame_data[i] << 8) + frame_data[i + 1]
        data_array[i // 2] = new_data
    '''数据整形'''
    # print("数据最大值：")
    # print(max(data_array))
    # print("数据最小值：")
    # print(min(data_array))

    return data_array

def data_interpolated(data):
    # 创建4x5的原始网格
    x = np.linspace(0, data.shape[0]-1, data.shape[0])  # 对应数据矩阵的行数
    y = np.linspace(0, data.shape[1]-1, data.shape[1])  # 对应数据矩阵的列数

    # 创建512x512的目标网格
    x_new = np.linspace(0, data.shape[0]-1, 50)
    y_new = np.linspace(0, data.shape[1]-1, 50)

    # 创建插值函数
    f = RegularGridInterpolator((x, y),data, method='linear')

    # 对目标网格进行插值
    xv, yv = np.meshgrid(x_new, y_new, indexing='ij')
    interpolated_matrix = f((xv, yv))
    return interpolated_matrix

class Worker(QObject):
    dataReceived = pyqtSignal(np.ndarray)

    def __init__(self):
        super().__init__()

    def run(self):
        try:
            mSerial = serial.Serial(serialPort, baudRate)
            print("串口打开成功！")
            while True:
                # 数据接收
                self.dataReceived.emit(read_data_array(mSerial))
                # print("ing")
        except serial.SerialException as e:
            print("串口打开失败...")


class MainWindow(pg.GraphicsLayoutWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle('Real-time Data Visualization')
        self.resize(2000, 800)

        self.p1 = self.addPlot(title="Finger 1")
        self.img1 = pg.ImageItem()
        self.p1.addItem(self.img1)

        self.nextColumn()
        self.p2 = self.addPlot(title="Finger 2")
        self.img2 = pg.ImageItem()
        self.p2.addItem(self.img2)

        self.nextColumn()
        self.p3 = self.addPlot(title="Finger 3")
        self.img3 = pg.ImageItem()
        self.p3.addItem(self.img3)

        self.nextColumn()
        self.p4 = self.addPlot(title="Finger 4")
        self.img4 = pg.ImageItem()
        self.p4.addItem(self.img4)

        self.nextColumn()
        self.p5 = self.addPlot(title="Finger 5")
        self.img5 = pg.ImageItem()
        self.p5.addItem(self.img5)

        # 初始化用于存储前10次循环的初始值
        self.initial_values = {
            'img1_array': np.zeros((10, 5, 4)),
            'img2_array': np.zeros((10, 5, 4)),
            'img3_array': np.zeros((10, 5, 4)),
            'img4_array': np.zeros((10, 5, 4)),
            'img5_array': np.zeros((10, 5, 4))
        }
        self.avg_initial_values = {}
        self.loop_count = 0
        self.initial_values_computed = False

        self.worker = Worker()
        self.workerThread = QThread()
        self.worker.moveToThread(self.workerThread)
        self.workerThread.started.connect(self.worker.run)
        self.worker.dataReceived.connect(self.updateImage)

        self.workerThread.start()

    def updateImage(self, new_array):
        if new_array is None:
            print("返回无效数据")
            return

        img_shape = (5, 4)
        img1_array = new_array[:20].reshape(img_shape, order='C').astype(np.float64)
        img2_array = new_array[20:40].reshape(img_shape, order='C').astype(np.float64)
        img3_array = new_array[40:60].reshape(img_shape, order='C').astype(np.float64)
        img4_array = new_array[60:80].reshape(img_shape, order='C').astype(np.float64)
        img5_array = new_array[80:100].reshape(img_shape, order='C').astype(np.float64)

        if self.loop_count < 10:
            # 存储前10次的值
            self.initial_values['img1_array'][self.loop_count] = img1_array
            self.initial_values['img2_array'][self.loop_count] = img2_array
            self.initial_values['img3_array'][self.loop_count] = img3_array
            self.initial_values['img4_array'][self.loop_count] = img4_array
            self.initial_values['img5_array'][self.loop_count] = img5_array
        else:
            if not self.initial_values_computed:
                # 计算前10次的平均值，并将其转换为 float64
                self.avg_initial_values['img1_array'] = np.mean(self.initial_values['img1_array'], axis=0).astype(
                    np.float64)
                self.avg_initial_values['img2_array'] = np.mean(self.initial_values['img2_array'], axis=0).astype(
                    np.float64)
                self.avg_initial_values['img3_array'] = np.mean(self.initial_values['img3_array'], axis=0).astype(
                    np.float64)
                self.avg_initial_values['img4_array'] = np.mean(self.initial_values['img4_array'], axis=0).astype(
                    np.float64)
                self.avg_initial_values['img5_array'] = np.mean(self.initial_values['img5_array'], axis=0).astype(
                    np.float64)
                self.initial_values_computed = True
                print(self.avg_initial_values)

            # 减去平均初始值，并进行裁剪
            img1_array -= self.avg_initial_values['img1_array']
            img2_array -= self.avg_initial_values['img2_array']
            img3_array -= self.avg_initial_values['img3_array']
            img4_array -= self.avg_initial_values['img4_array']
            img5_array -= self.avg_initial_values['img5_array']

            # 将结果裁剪到 uint16 的范围，并转换为 uint16
            img1_array = np.clip(img1_array, 0, 65535).astype(np.uint16)
            img2_array = np.clip(img2_array, 0, 65535).astype(np.uint16)
            img3_array = np.clip(img3_array, 0, 65535).astype(np.uint16)
            img4_array = np.clip(img4_array, 0, 65535).astype(np.uint16)
            img5_array = np.clip(img5_array, 0, 65535).astype(np.uint16)

            # 调用 data_interpolated 函数（假设这个函数存在）
            img1_array_interpolated = data_interpolated(img1_array)
            img2_array_interpolated = data_interpolated(img2_array)
            img3_array_interpolated = data_interpolated(img3_array)
            img4_array_interpolated = data_interpolated(img4_array)
            img5_array_interpolated = data_interpolated(img5_array)

            # 更新图像
            self.img1.setImage(img1_array_interpolated)
            self.img2.setImage(img2_array_interpolated)
            self.img3.setImage(img3_array_interpolated)
            self.img4.setImage(img4_array_interpolated)
            self.img5.setImage(img5_array_interpolated)

            # 使用 matplotlib 的 jet 颜色映射
            cmap = plt.get_cmap('jet')
            lut = cmap(np.linspace(0, 1, 256)) * 255
            lut = lut.astype(np.ubyte)

            # 设置图像的颜色映射
            self.img1.setLookupTable(lut)
            self.img2.setLookupTable(lut)
            self.img3.setLookupTable(lut)
            self.img4.setLookupTable(lut)
            self.img5.setLookupTable(lut)

            # 设置图像的显示范围
            self.img1.setLevels([0, 800])
            self.img2.setLevels([0, 500])
            self.img3.setLevels([0, 500])
            self.img4.setLevels([0, 800])
            self.img5.setLevels([0, 800])

        # 增加循环计数
        self.loop_count += 1


if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainWindow = MainWindow()
    mainWindow.show()
    sys.exit(app.exec_())

