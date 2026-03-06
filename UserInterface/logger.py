"""
logger.py — ANSI colour terminal logging
"""

import time


class C:
    RESET  = "\033[0m"
    BOLD   = "\033[1m"
    CYAN   = "\033[96m"
    GREEN  = "\033[92m"
    YELLOW = "\033[93m"
    RED    = "\033[91m"
    PURPLE = "\033[95m"
    DIM    = "\033[2m"
    WHITE  = "\033[97m"


def log(tag: str, msg: str, colour: str = C.CYAN) -> None:
    ts = time.strftime("%H:%M:%S")
    print(f"{C.DIM}[{ts}]{C.RESET} {colour}{C.BOLD}[{tag}]{C.RESET} {msg}")


def log_cmd(tag: str, msg: str)  -> None: log(tag, msg, C.GREEN)
def log_warn(tag: str, msg: str) -> None: log(tag, msg, C.YELLOW)
def log_err(tag: str, msg: str)  -> None: log(tag, msg, C.RED)
def log_evt(tag: str, msg: str)  -> None: log(tag, msg, C.PURPLE)
def log_info(tag: str, msg: str) -> None: log(tag, msg, C.WHITE)
