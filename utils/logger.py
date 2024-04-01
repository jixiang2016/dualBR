# Copyright (c) Facebook, Inc. and its affiliates.
import base64
import logging
import os
import sys

from utils.timer import Timer
from utils.distributed_utils import is_main_process

class Logger:

    def __init__(self, config):
        self.logger = None
        self.timer = Timer()
        self.save_dir = config.output_dir 
        self.log_folder = config.dataset_name+'_'+config.data_mode1+'-'+config.data_mode2+'_'+str(config.input_num)+'_'+str(config.output_num)  
        time_format = "%Y-%m-%dT%H:%M:%S"
        self.log_filename = self.log_folder + "_"   
        self.log_filename += self.timer.get_time_hhmmss(None, format=time_format) 
        self.log_filename += ".log" 

        self.log_folder = os.path.join(self.save_dir, self.log_folder,'log') 
		# if not exist, create
        if not os.path.exists(self.log_folder):
            os.makedirs(self.log_folder, exist_ok=True)

        self.log_filename = os.path.join(self.log_folder, self.log_filename) 

        if is_main_process():
            print("Logging to:", self.log_filename)

        logging.captureWarnings(True)

        self.logger = logging.getLogger(__name__)#__name__
        self._file_only_logger = logging.getLogger(__name__)
        warnings_logger = logging.getLogger("py.warnings")

        # Set level 
        level = "info"
        self.logger.setLevel(getattr(logging, level.upper()))
        self.logger.propagate = False
        self._file_only_logger.setLevel(getattr(logging, level.upper()))
        formatter = logging.Formatter(
            "%(asctime)s %(levelname)s: %(message)s", datefmt="%Y-%m-%dT%H:%M:%S"
        )

        # Add handler to file
        channel = logging.FileHandler(filename=self.log_filename, mode="a")
        channel.setFormatter(formatter)

        self.logger.addHandler(channel)
        self._file_only_logger.addHandler(channel)
        warnings_logger.addHandler(channel)

        # Add handler to stdout
        channel = logging.StreamHandler(sys.stdout)
        channel.setFormatter(formatter)

        self.logger.addHandler(channel)
        warnings_logger.addHandler(channel)

        self.should_log = config.should_log 

    def write(self, x, level="info", donot_print=False, log_all=False):
        if self.logger is None:
            return

        if log_all is False and not is_main_process():
            return
    
        if self.should_log:
            if hasattr(self.logger, level):
                if donot_print:
                    getattr(self._file_only_logger, level)(str(x))
                else:
                    getattr(self.logger, level)(str(x))
            else:
                self.logger.error("Unknown log level type: %s" % level)
        else:
            print(str(x) + "\n")

			
			
