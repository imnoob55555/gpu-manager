"""GPU Manager: 实验室共享 GPU 服务器上的个人空闲提醒 + 训练任务自动调度工具。

硬性安全边界（详见 README.md 「硬性安全边界」一节）:
  - 不 kill / pause / signal 其他用户的进程
  - 不清理其他用户显存、不 reset GPU、不使用 sudo
  - 不抢占已被其他用户使用的 GPU
  - 只观察 GPU、启动自己的进程、监控/停止自己的进程
"""

__version__ = "1.0.0"
__all__ = ["__version__"]
