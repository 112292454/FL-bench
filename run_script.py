import random
import subprocess
import sys
from pathlib import Path
import asyncio
import torch
from typing import Dict, Set, List
from dataclasses import dataclass
from collections import defaultdict
import time
import os
import logging

#### see in https://github.com/uglyghost/FedOBP/blob/master/run_script.py


# 设置日志配置
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

@dataclass
class GPUResource:
    """GPU资源管理类"""
    id: int
    semaphore: asyncio.Semaphore  # 控制并发任务数
    running_tasks: int = 0  # 当前运行的任务数

class TaskStats:
    """任务统计类"""
    def __init__(self):
        self.completed_tasks = 0
        self.total_time = 0.0
        self.last_report_time = time.time()
        self.start_time = time.time()
        self.last_report_count = 0
    
    def add_task_time(self, duration: float):
        self.completed_tasks += 1
        self.total_time += duration
        
        # 每10个任务报告一次
        if self.completed_tasks % 10 == 0:
            current_time = time.time()
            tasks_since_last = self.completed_tasks - self.last_report_count
            time_since_last = current_time - self.last_report_time
            
            logger.info("--- 任务统计 ---")
            logger.info(f"已完成任务数: {self.completed_tasks}")
            logger.info(f"最近 {tasks_since_last} 个任务平均耗时: {time_since_last/tasks_since_last:.2f} 秒")
            logger.info(f"总体平均耗时: {(current_time-self.start_time)/self.completed_tasks:.2f} 秒")
            
            self.last_report_time = current_time
            self.last_report_count = self.completed_tasks
    
    def final_report(self):
        if self.completed_tasks > 0:
            logger.info("=== 最终统计 ===")
            logger.info(f"总任务数: {self.completed_tasks}")
            logger.info(f"平均耗时: {self.total_time/self.completed_tasks:.2f} 秒")
            logger.info(f"总耗时: {self.total_time:.2f} 秒")

class GPUManager:
    """GPU资源管理器"""
    def __init__(self, max_tasks_per_gpu: int = 1):
        self.gpu_count = torch.cuda.device_count()
        self.gpus = {
            i: GPUResource(
                id=i,
                semaphore=asyncio.Semaphore(max_tasks_per_gpu)
            ) for i in range(self.gpu_count)
        }
        logger.info(f"初始化 {self.gpu_count} 个 GPU，每个GPU最多运行 {max_tasks_per_gpu} 个任务")
    
    async def acquire_gpu(self, task_slots: int = 1) -> GPUResource:
        """获取负载最小的GPU
        Args:
            task_slots: 需要申请的任务槽数量
        """
        while True:
            # 对所有GPU尝试获取信号量，倒序遍历以优先使用编号较大的GPU
            for gpu in sorted(self.gpus.values(), key=lambda x: x.running_tasks, reverse=True):
                # 检查GPU是否有足够的剩余槽位
                if gpu.running_tasks + task_slots > gpu.semaphore._value:
                    continue
                    
                try:
                    # 尝试获取信号量，设置超时时间为0.1秒
                    acquired_all = True
                    for _ in range(task_slots):
                        if not await asyncio.wait_for(gpu.semaphore.acquire(), timeout=0.1):
                            acquired_all = False
                            break
                            
                    if acquired_all:
                        gpu.running_tasks += task_slots
                        return gpu
                    else:
                        # 如果未能获取所有需要的槽位，释放已获取的槽位
                        for _ in range(task_slots):
                            gpu.semaphore.release()
                except asyncio.TimeoutError:
                    continue  # 如果超时，尝试下一个GPU
            
            # 如果所有GPU都尝试失败，等待一段时间后重试
            await asyncio.sleep(5)
    
    def release_gpu(self, gpu: GPUResource, task_slots: int = 1):
        """释放GPU资源
        Args:
            gpu: 要释放的GPU资源
            task_slots: 要释放的任务槽数量
        """
        gpu.running_tasks -= task_slots
        for _ in range(task_slots):
            gpu.semaphore.release()

async def run_command_async(command: List[str], log_file: Path, gpu: GPUResource, gpu_manager: GPUManager, stats: TaskStats):
    """异步运行命令并将输出写入日志文件"""
    start_time = time.time()
    # 获取数据集名称
    dataset_name = next((arg.split('=')[1] for arg in command if 'dataset.name=' in arg), '')
    task_slots = 2 if dataset_name == 'emnist' else 1
    
    try:
        logger.info(f"[GPU {gpu.id}] 开始运行: {' '.join(command)}")
        with open(log_file, 'w', encoding='utf-8') as f:
            env = dict(os.environ)
            env['CUDA_VISIBLE_DEVICES'] = str(gpu.id)
            
            process = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                env=env
            )

            while True:
                line = await process.stdout.readline()
                if not line:
                    break
                try:
                    decoded_line = line.decode('utf-8')
                except UnicodeDecodeError:
                    decoded_line = line.decode('utf-8', errors='ignore')
                f.write(decoded_line)

            await process.wait()
            if process.returncode != 0:
                logger.error(f"[GPU {gpu.id}] 命令失败，查看日志文件: {log_file}")
            else:
                logger.info(f"[GPU {gpu.id}] 命令成功完成: {' '.join(command)}")
    finally:
        duration = time.time() - start_time
        stats.add_task_time(duration)
        # 检查是否是从GPU管理器获取的资源
        if gpu in gpu_manager.gpus.values():
            gpu_manager.release_gpu(gpu, task_slots)
            logger.info(f"[GPU {gpu.id}] 释放了 {task_slots} 个任务槽 (当前运行: {gpu.running_tasks})")

async def main():
    # 定义参数
    datasets_name = ['emnist','cifar10','svhn','cifar100','medmnistA','medmnistC','fmnist','mnist']
    # datasets_name = ['emnist']

    # datasets_name = ['cifar100']
    # methods = ['fedavg', 'local']  # baselines
    # methods = ['fedper', 'apfl', 'lgfedavg', 'fedrep', 'pfedla', 'feddpa', 'flute', 'floco']  SOTA solutions
    # methods = ['FedMD', 'pfedsim', 'perFedAvg', 'pFedHN', 'pFedMe', 'FedEM', 'Floco','FedAP', 'kNNPer', 'FedProto', 'FedPAC', 'MetaFed', 'FedALA', 'FedAS', 'pFedFDA', 'FedFomo', 'FedBN', 'CFL','pFedLA', 'FedBabu', 'PeFLL', 'Ditto']

    # split to:
    # methods=[
    # 'pFedHN',
    # 'MetaFed',
    
    # 'pFedLA',
    # 'FedAP',
    # 'FedMD',
    # ] # serial
    # methods=['FedMD','FedBabu','PeFLL'] # 数据集特殊设置/需要fintune/doesn't support global buffers


    # methods=['FedMD','pfedsim','perFedAvg','pFedHN','pFedMe','FedEM','Floco',]  # 4090*4
    # methods=['FedAP','kNNPer','FedProto','FedPAC','MetaFed','FedALA','FedAS','pFedFDA','FedFomo','FedBN','CFL',]  # 3090*8
    methods=['pFedLA','FedBabu','PeFLL','Ditto',]   # a800*3

    methods = [m.lower() for m in methods]

    # 创建日志目录
    log_dir = Path("experiment_logs")
    log_dir.mkdir(exist_ok=True)

    # 初始化GPU管理器和任务统计
    gpu_manager = GPUManager(max_tasks_per_gpu=2)
    stats = TaskStats()
    tasks = []

    # 遍历每个方法和数据集的组合
    for dataset in datasets_name:
        for method in methods:
            # 构建命令
            command = [
                sys.executable,
                'main.py',
                f'method={method}',
                f'dataset.name={dataset}',
            ]

            # 添加特殊设置
            if method in ['fedbabu']:
                command.append('common.test.client.finetune_epoch=10')
            if method in ['pefll','pfedhn']:
                command.append('common.buffers=local')
            if method in ['fedmd']:
                if dataset in ['cifar100']:
                    command.append('+fedmd.public_dataset=cifar10')
                elif dataset in ['emnist','femnist']:
                    command.append('+fedmd.public_dataset=emnist')
                else:
                    command.append(f'+fedmd.public_dataset={dataset}')
            if method in ['pfedhn','pfedla','metafed']:
                command.append('mode=serial')

            # 准备日志文件
            log_filename = f"{method}_{dataset}.log"
            log_path = log_dir / log_filename

            # 获取GPU资源
            if method in ['pfedhn', 'metafed', 'pfedla', 'fedap', 'fedmd'] or 'mode=serial' in command:
                gpu = GPUResource(id=random.randint(0, torch.cuda.device_count() - 1), semaphore=asyncio.Semaphore(1))
            else:
                task_slots = 2 if dataset == 'emnist' else 1
                gpu = await gpu_manager.acquire_gpu(task_slots)

            # 创建新的异步任务
            task = asyncio.create_task(
                run_command_async(command, log_path, gpu, gpu_manager, stats)
            )
            tasks.append(task)
            time.sleep(3 if torch.cuda.device_count() <= 4 else 20)

    # 等待所有任务完成
    await asyncio.gather(*tasks)
    stats.final_report()
    logger.info("所有实验已完成。")

if __name__ == '__main__':
    asyncio.run(main())
