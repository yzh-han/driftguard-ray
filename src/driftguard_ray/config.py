
import logging
import sys
from pathlib import Path

def setup_logging(
    level: str = "INFO",
    log_file: str | Path | None = None,
    format_str: str | None = None
) -> logging.Logger:
    """设置项目日志配置
    
    Args:
        level: 日志级别 (DEBUG, INFO, WARNING, ERROR)
        log_file: 可选的日志文件路径
        format_str: 自定义格式字符串
        
    Returns:
        配置好的 logger
    """
    if format_str is None:
        format_str = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    
    # 获取 root logger
    logger = logging.getLogger("driftguard")
    
    # 避免重复配置
    if logger.handlers:
        return logger
        
    logger.setLevel(getattr(logging, level.upper()))
    
    # 控制台输出
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper()))
    console_formatter = logging.Formatter(format_str)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # 文件输出 (可选)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(getattr(logging, level.upper()))
        file_formatter = logging.Formatter(format_str)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger

# 在其他模块中的用法示例:
def get_logger(name: str | None = None) -> logging.Logger:
    """获取子 logger
    
    Args:
        name: logger 名称，默认使用调用模块的 __name__
        
    Returns:
        logger 实例
    """
    if name is None:
        import inspect
        frame = inspect.currentframe()
        if frame and frame.f_back:
            name = frame.f_back.f_globals.get('__name__', 'unknown')
        else:
            name = 'unknown'
    
    return logging.getLogger(f"driftguard.{name}")


# 便捷的日志级别函数
def set_log_level(level: str):
    """动态调整日志级别"""
    logging.getLogger("driftguard").setLevel(getattr(logging, level.upper()))

# setup_logging(level="DEBUG")  # 默认日志级别为 INFO
setup_logging(level="INFO")  # 默认日志级别为 INFO
