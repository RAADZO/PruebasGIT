import logging
import logging.config
import os
from contextlib import contextmanager
import yaml

logging.getLogger().setLevel(logging.CRITICAL)

# list current variables
# locals()
current_dir = os.getcwd()
project_dir = current_dir
project_abs_dir = os.path.abspath(os.path.join(project_dir, ".."))
project_dir = os.path.dirname(os.path.abspath(""))
project_abs_dir = os.path.dirname(r"..\\")
data_dir          = os.path.join(project_abs_dir, "data")
raw_data          = os.path.join(data_dir, "0_raw")
intermediate_data = os.path.join(data_dir, "1_intermediate")
processed_data    = os.path.join(data_dir, "2_processed")

def create_log_directory(log_dir='logs'):
    """Create a log directory if it doesn't exist."""
    try:
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
    except Exception as e:
        function_name = "create_log_directory"
        message = f"[ERROR]: {function_name} {e}"
        print(message)
        logger.error(message)
        raise e

def close_logger_handlers(logger):
    """Close all handlers associated with a logger."""
    try:
        handlers = logger.handlers[:]
        for handler in handlers:
            handler.close()
            logger.removeHandler(handler)
    except Exception as e:
        function_name = "close_logger_handlers"
        message = f"[ERROR]: {function_name} {e}"
        print(message)
        logger.error(message)
        raise e

def clear_log_files(log_files, log_dir='logs'):
    """Delete log files to start fresh."""
    try:
        for log_file in log_files:
            log_file_path = os.path.join(log_dir, log_file)
            if os.path.exists(log_file_path):
                os.remove(log_file_path)
    except Exception as e:
        function_name = "clear_log_files"
        message = f"[ERROR]: {function_name} {e}"
        print(message)
        logger.error(message)
        raise e
    
def setup_logging(default_path='logs/logging_config.yaml', default_level=logging.INFO, env_key='LOG_CFG'):
    """Setup logging configuration"""
    try:
        path = default_path
        value = os.getenv(env_key, None)
        if value:
            path = value
        if os.path.exists(path):
            with open(path, 'rt') as f:
                config = yaml.safe_load(f.read())
            logging.config.dictConfig(config)
        else:
            logging.basicConfig(level=default_level)
    except Exception as e:
        function_name = "setup_logging"
        message = f"[ERROR]: {function_name} {e}"
        print(message)
        logger.error(message)
        raise e

@contextmanager
def use_logger(logger_name):
    """Context manager for using a logger."""
    try:
        logger = logging.getLogger(logger_name)
        try:
            yield logger
        finally:
            close_logger_handlers(logger)
    except Exception as e:
        function_name = "setup_logging"
        message = f"[ERROR]: {function_name} {e}"
        print(message)
        logger.error(message)
        raise e

# Create log directory
create_log_directory()

# logging configuration in YAML format
logging_config_yaml = """
version: 1
disable_existing_loggers: False
formatters:
  standard:
    format: "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
handlers:
  file_handler_incident:
    class: logging.handlers.RotatingFileHandler
    level: INFO
    formatter: standard
    filename: logs/incident_management.log
    maxBytes: 10485760  # 10 MB
    backupCount: 3
    encoding: utf8
  file_handler_transform:
    class: logging.handlers.RotatingFileHandler
    level: INFO
    formatter: standard
    filename: logs/transform_df_optimization.log
    maxBytes: 10485760  # 10 MB
    backupCount: 3
    encoding: utf8
loggers:
  incident_management:
    handlers: [file_handler_incident]
    level: INFO
    propagate: False
  transform_df_optimization:
    handlers: [file_handler_transform]
    level: INFO
    propagate: False
root:
  handlers: []
  level: WARNING
"""

# Save the logging configuration to a file
os.makedirs('logs', exist_ok=True)
with open('logs/logging_config.yaml', 'w', encoding='utf-8') as file:
    file.write(logging_config_yaml)

# Setup logging
setup_logging()
logger = logging.getLogger("incident_management")
logger_transform = logging.getLogger("transform_df_optimization")

# Close handlers before clearing log files
close_logger_handlers(logger)
close_logger_handlers(logger_transform)
clear_log_files(['incident_management.log', 'transform_df_optimization.log'])

# Setup logging
setup_logging()
logger = logging.getLogger("incident_management")
logger_transform = logging.getLogger("transform_df_optimization")

logger.info('configured')
logger_transform.info('configured')


casos_audit = {}

# ------------------------------------------------------
# ------------------------------------------------------ MOLD FUNCTION

# def function_name(df, par1, par2, par3):
#     try:
#         df = df.copy()
#         # code
#         def internal_function_name(group):
#             try:
#                 # code
#                 # return n
#             except Exception as e:
#                 function_name = "internal_function_name"
#                 message = f"[ERROR]: {function_name} {e}"
#                 print(message)
#                 logger.error(message)
#                 raise e
#         # code
#         return result
#     except Exception as e:
#         function_name = "function_name"
#         message = f"[ERROR]: {function_name} {e}"
#         print(message)
#         logger.error(message)
#         raise e

# ------------------------------------------------------
# ------------------------------------------------------ MOLD MAIN SECTION

# log_separator(title="Section: [____] Description: [_____]")
# try:
#     pass
#     logger.info(f"Group details:\n{df_base}")
# except Exception as e:
#     function_name = "___"
#     message = f"[ERROR]: {function_name} {e}"
#     print(message)
#     logger.error(message)
#     raise e