"""
训练监控模块
提供实时终端打印和 SwanLab 实验追踪
"""

import time
from typing import Optional, Dict, Any
from pathlib import Path
import json
from transformers import TrainerCallback, TrainerState, TrainerControl
from transformers.trainer_utils import IntervalStrategy

try:
    import swanlab
    SWANLAB_AVAILABLE = True
except ImportError:
    SWANLAB_AVAILABLE = False
    print("⚠️ SwanLab 未安装，将跳过实验追踪功能")


class TrainingMonitor(TrainerCallback):
    """训练监控回调 - 终端打印 + SwanLab"""
    
    def __init__(
        self,
        use_swanlab: bool = True,
        swanlab_project: str = "protected-lora",
        swanlab_experiment: Optional[str] = None,
        swanlab_config: Optional[Dict[str, Any]] = None,
        print_interval: int = 10,  # 每 N 步打印一次
        enable_console_logging: bool = True,
    ):
        """
        Args:
            use_swanlab: 是否使用 SwanLab
            swanlab_project: SwanLab 项目名称
            swanlab_experiment: 实验名称（可选，自动生成）
            swanlab_config: 实验配置字典
            print_interval: 终端打印间隔（步数）
            enable_console_logging: 是否启用终端日志
        """
        self.use_swanlab = use_swanlab and SWANLAB_AVAILABLE
        self.swanlab_project = swanlab_project
        self.swanlab_experiment = swanlab_experiment
        self.swanlab_config = swanlab_config or {}
        self.print_interval = print_interval
        self.enable_console_logging = enable_console_logging
        
        # 状态记录
        self.start_time = None
        self.last_print_step = 0
        self.last_print_time = None
        self.step_times = []
        
        # SwanLab 实例
        self.swanlab_run = None
        
    def on_train_begin(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        """训练开始时初始化"""
        self.start_time = time.time()
        self.last_print_time = self.start_time
        
        # 初始化 SwanLab
        if self.use_swanlab:
            try:
                self.swanlab_run = swanlab.init(
                    project=self.swanlab_project,
                    experiment_name=self.swanlab_experiment,
                    config=self.swanlab_config,
                )
                print(f"✅ SwanLab 实验已初始化: {self.swanlab_project}")
                print(f"   查看地址: {self.swanlab_run.url if hasattr(self.swanlab_run, 'url') else 'https://swanlab.cn'}")
            except Exception as e:
                print(f"⚠️ SwanLab 初始化失败: {e}")
                self.use_swanlab = False
        
        if self.enable_console_logging:
            print("\n" + "="*80)
            print("🚀 训练开始")
            print("="*80)
            print(f"📊 总步数: {state.max_steps}")
            print(f"📈 打印间隔: 每 {self.print_interval} 步")
            print("="*80 + "\n")
    
    def on_log(self, args, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        """每次日志记录时调用"""
        if logs is None:
            return
        
        current_step = state.global_step
        
        # 记录到 SwanLab
        if self.use_swanlab and self.swanlab_run is not None:
            try:
                # 过滤并记录指标
                filtered_logs = {}
                for key, value in logs.items():
                    if isinstance(value, (int, float)):
                        filtered_logs[key] = value
                
                if filtered_logs:
                    swanlab.log(filtered_logs, step=current_step)
            except Exception as e:
                print(f"⚠️ SwanLab 日志记录失败: {e}")
        
        # 终端打印（按间隔）
        if self.enable_console_logging and (
            current_step % self.print_interval == 0 or 
            current_step == state.max_steps
        ):
            self._print_progress(logs, state)
    
    def _print_progress(self, logs: Dict, state: TrainerState):
        """打印训练进度"""
        current_step = state.global_step
        current_time = time.time()
        
        # 计算速度
        if self.last_print_step > 0:
            steps_done = current_step - self.last_print_step
            time_elapsed = current_time - self.last_print_time
            speed = steps_done / time_elapsed if time_elapsed > 0 else 0
            self.step_times.append(speed)
        else:
            speed = 0
        
        # 计算进度
        progress = (current_step / state.max_steps * 100) if state.max_steps > 0 else 0
        
        # 提取关键指标
        loss = logs.get('loss', None)
        learning_rate = logs.get('learning_rate', None)
        epoch = logs.get('epoch', None)
        
        # 格式化输出
        print(f"📍 Step {current_step:>5}/{state.max_steps} ({progress:>5.1f}%) | ", end="")
        
        if loss is not None:
            print(f"Loss: {loss:>7.4f} | ", end="")
        
        if learning_rate is not None:
            print(f"LR: {learning_rate:.2e} | ", end="")
        
        if speed > 0:
            print(f"Speed: {speed:>5.2f} it/s | ", end="")
        
        if epoch is not None:
            print(f"Epoch: {epoch:>5.2f}", end="")
        
        print()  # 换行
        
        # 更新记录
        self.last_print_step = current_step
        self.last_print_time = current_time
    
    def on_train_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        """训练结束时总结"""
        if self.enable_console_logging:
            total_time = time.time() - self.start_time
            avg_speed = sum(self.step_times) / len(self.step_times) if self.step_times else 0
            
            print("\n" + "="*80)
            print("✅ 训练完成!")
            print("="*80)
            print(f"⏱️  总时间: {total_time:.2f} 秒 ({total_time/60:.2f} 分钟)")
            print(f"📈 平均速度: {avg_speed:.2f} it/s")
            print(f"📊 总步数: {state.global_step}")
            print("="*80 + "\n")
        
        # 完成 SwanLab 实验
        if self.use_swanlab and self.swanlab_run is not None:
            try:
                swanlab.finish()
                print("✅ SwanLab 实验已保存")
            except Exception as e:
                print(f"⚠️ SwanLab 结束时出错: {e}")


class OrthogonalLossMonitor(TrainerCallback):
    """正交损失专用监控"""
    
    def __init__(
        self,
        trainer_with_orth_loss,  # OrthogonalLoRATrainer 实例
        print_interval: int = 10,
        enable_console_logging: bool = True,
        use_swanlab: bool = True,
    ):
        """
        Args:
            trainer_with_orth_loss: 包含 orth_loss_history 的 Trainer
            print_interval: 打印间隔
            enable_console_logging: 是否启用终端打印
            use_swanlab: 是否使用 SwanLab
        """
        self.trainer = trainer_with_orth_loss
        self.print_interval = print_interval
        self.enable_console_logging = enable_console_logging
        self.use_swanlab = use_swanlab and SWANLAB_AVAILABLE
        self.last_logged_step = -1
    
    def on_step_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        """每步结束后检查并记录正交损失"""
        if not hasattr(self.trainer, 'orth_loss_history'):
            return
        
        current_step = state.global_step
        
        # 获取最新的正交损失记录
        if len(self.trainer.orth_loss_history) > 0:
            latest_entry = self.trainer.orth_loss_history[-1]
            
            # 避免重复记录
            if latest_entry['step'] != self.last_logged_step:
                self.last_logged_step = latest_entry['step']
                
                # 记录到 SwanLab
                if self.use_swanlab:
                    try:
                        swanlab_data = {
                            'orth_loss/total': latest_entry['orth_loss'],
                            'orth_loss/task_loss': latest_entry['task_loss'],
                        }
                        
                        # 记录各维度的正交损失
                        for dim, value in latest_entry['details'].items():
                            swanlab_data[f'orth_loss/{dim}'] = value
                        
                        swanlab.log(swanlab_data, step=current_step)
                    except Exception as e:
                        print(f"⚠️ 记录正交损失到 SwanLab 失败: {e}")
                
                # 终端打印
                if self.enable_console_logging and current_step % self.print_interval == 0:
                    orth_loss = latest_entry['orth_loss']
                    task_loss = latest_entry['task_loss']
                    
                    print(f"  ┣━ Orth Loss: {orth_loss:>9.6f} | Task Loss: {task_loss:>7.4f}", end="")
                    
                    # 打印各维度
                    dims_str = " | ".join([f"{dim}={val:.6f}" for dim, val in latest_entry['details'].items()])
                    if dims_str:
                        print(f" | {dims_str}")
                    else:
                        print()


def create_training_callbacks(
    trainer,
    use_swanlab: bool = True,
    swanlab_project: str = "protected-lora",
    swanlab_experiment: Optional[str] = None,
    swanlab_config: Optional[Dict[str, Any]] = None,
    print_interval: int = 10,
    enable_console_logging: bool = True,
    monitor_orth_loss: bool = False,
) -> list:
    """
    创建训练回调列表
    
    Args:
        trainer: Trainer 实例
        use_swanlab: 是否使用 SwanLab
        swanlab_project: SwanLab 项目名
        swanlab_experiment: 实验名称
        swanlab_config: 配置字典
        print_interval: 打印间隔
        enable_console_logging: 启用终端日志
        monitor_orth_loss: 是否监控正交损失
    
    Returns:
        callbacks: 回调列表
    """
    callbacks = []
    
    # 添加基础训练监控
    callbacks.append(
        TrainingMonitor(
            use_swanlab=use_swanlab,
            swanlab_project=swanlab_project,
            swanlab_experiment=swanlab_experiment,
            swanlab_config=swanlab_config,
            print_interval=print_interval,
            enable_console_logging=enable_console_logging,
        )
    )
    
    # 如果需要，添加正交损失监控
    if monitor_orth_loss and hasattr(trainer, 'orth_loss_history'):
        callbacks.append(
            OrthogonalLossMonitor(
                trainer_with_orth_loss=trainer,
                print_interval=print_interval,
                enable_console_logging=enable_console_logging,
                use_swanlab=use_swanlab,
            )
        )
    
    return callbacks
