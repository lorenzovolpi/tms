import json
import os
import subprocess as sb

import cap
import quapy as qp

if os.path.exists("env.json"):
    _hostname = sb.run(["hostname"], capture_output=True).stdout.decode("UTF-8").strip()
    with open("env.json", "r") as f:
        _jdict = json.load(f)
        _env = _hostname if _hostname in _jdict else "global"
        ext_env = _jdict.get(_env, cap.env)
    cap.env |= ext_env

PROJECT = "tms"
root_dir = os.path.join(cap.env["OUT_DIR"], PROJECT)
NUM_TEST = 1000
qp.environ["_R_SEED"] = 0

_valid_problems = ["binary", "multiclass"]
PROBLEM = "binary"
