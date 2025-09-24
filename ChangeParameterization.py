import sys

sys.path.append(r"C:\IPG\carmaker\win64-11.1.2\Python\python3.9")

from pathlib import Path

import cmapi
from cmapi import Project, simcontrol, testrun

project_path = Path("C:\CM_Projects\Behavior_cloning")
Project.load(project_path)

testrun_path = Path("Examples/BasicFunctions/Driver/RaceDriver")
testrun = Project.instance().load_testrun_parametrization(testrun_path)

# testrun.set_parameter_value("VehicleLoad.0.mass", 200)
# testrun.set_parameter_value("VehicleLoad.0.pos", "3.2 0.0 0.0")

infofile_path = Path("./MyInfofile")
testrun.set_path(infofile_path)

cmapi.Project.instance().write_parametrization(testrun)
