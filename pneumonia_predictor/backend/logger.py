from datetime import datetime
from pathlib import Path
from sys import exit

from pneumonia_predictor.config import LOGFILE_ENABLED, LOGFILE_LOC, LOGS_ENABLED


class Logger:
    def __init__(
        self, logfile_enabled: bool = LOGFILE_ENABLED, logfile_loc: str = LOGFILE_LOC
    ) -> None:
        self.time_log_fmt = "%d/%m/%Y - %H:%M:%S"
        self.logfile_enabled = logfile_enabled
        self.logfile_loc = logfile_loc
        self.log_types = {
            "err": lambda m: self.raise_error(m),
            "op": lambda m: self.log_operation(m),
            "inf": lambda m: self.log_info(m),
            "sep": lambda m: self.sep(m),
        }

    def log(self, log_type: str, message: str) -> None:
        if not LOGS_ENABLED:
            return None
        self.log_types[log_type](message)

    def raise_error(self, message: str) -> None:
        output = f"[{self.get_curr_datetime}][ERROR]: {message}. Exiting...\n"
        self.update_logfile(output)
        exit(output)

    def log_operation(self, message: str) -> None:
        self.update_logfile(f"[{self.get_curr_datetime()}][OP]: {message}.")

    def log_info(self, message: str) -> None:
        self.update_logfile(f"[{self.get_curr_datetime()}][INFO]: {message}")

    def sep(self, chosen_char: str, n_chars: int = 50) -> None:
        self.update_logfile(chosen_char * n_chars)

    def get_curr_datetime(self) -> str:
        curr_datetime = datetime.now()
        return curr_datetime.strftime(self.time_log_fmt)

    def create_logs_file(self) -> None:
        out_file = Path(self.logfile_loc)
        out_file.parent.mkdir(exist_ok=True, parents=True)
        out_file.write_text(f"TRAINING LOGS (Created at: {self.get_curr_datetime()})\n")

    def update_logfile(self, new_log: str) -> None:
        with open(self.logfile_loc, "a") as logfile:
            logfile.write(new_log + "\n")
