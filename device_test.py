import torch
import sys

def check_cuda():
    print(f"Python版本: {sys.version}")
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA是否可用: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA版本: {torch.version.cuda}")
        print(f"GPU设备数量: {torch.cuda.device_count()}")
        
        # 显示所有可用的GPU设备信息
        for i in range(torch.cuda.device_count()):
            print(f"\nGPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"GPU {i} 显存总量: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")

def test_cuda_operation():
    if not torch.cuda.is_available():
        print("\nCUDA不可用，跳过GPU运算测试")
        return
        
    print("\n执行简单的GPU张量运算测试...")
    try:
        # 创建一个GPU张量并执行简单运算
        x = torch.randn(1000, 1000).cuda()
        y = torch.randn(1000, 1000).cuda()
        
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        
        start_time.record()
        z = torch.matmul(x, y)
        end_time.record()
        
        # 等待GPU操作完成
        torch.cuda.synchronize()
        
        print(f"矩阵乘法耗时: {start_time.elapsed_time(end_time):.2f} ms")
        print("GPU运算测试完成，一切正常！")
        
    except Exception as e:
        print(f"GPU运算测试出错: {str(e)}")

if __name__ == '__main__':
    print("开始检查CUDA环境...\n")
    check_cuda()
    test_cuda_operation()