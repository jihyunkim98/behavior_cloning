import sys
import asyncio
import pandas as pd
import numpy as np

sys.path.append(r"C:\IPG\carmaker\win64-11.1.2\Python\python3.9")

from pathlib import Path

import cmapi
from cmapi import Project, simcontrol, testrun

project_path = Path(r"C:\CM_Projects\Behavior_cloning")
testrun_path = Path("Examples/BasicFunctions/Driver/RaceDriver")
# testrun_path = Path("Examples/VehicleDynamics/Handling/LaneChange_ISO")

async def main():
    cmapi.Project.load(project_path)
    testrun = Project.instance().load_testrun_parametrization(testrun_path)

    testrun.set_parameter_value("Driver", "None")
    testrun.set_parameter_value("DrivMan.0.LatDyn", "OpenLoop 0 0")
    testrun.set_parameter_value("DrivMan.0.LongDyn", "OpenLoop 0 0")
    # testrun.set_parameter_value("DrivMan.OW.Active", "1")

    variation = cmapi.Variation.create_from_testrun(testrun.clone())
    variation.set_name("BC_Openloop")

    # Set Storage Mode 
    variation.set_storage_mode(cmapi.StorageMode.save)

    # Set Quantities
    OutputQuantities = cmapi.OutputQuantities()
    OutputQuantities.set_output_format(cmapi.OutputFormat.mdf)
    OutputQuantities.add_quantities(['VC.Steer.Ang', 'VC.Gas', 'Time'])
    variation.set_outputquantities(OutputQuantities)

    simcontrol = cmapi.SimControlInteractive()
    simcontrol.set_variation(variation)
    
    master = cmapi.CarMaker()
    await simcontrol.set_master(master)

    movie = cmapi.IPGMovie()
    await simcontrol.start_and_connect()
    movie.attach_to_cm(simcontrol.get_master())
    cmapi.logger.info(f"IPG Movie Started")

    await movie.start()
    await simcontrol.start_sim()
    await asyncio.sleep(1.0)

    df = pd.read_csv(r"C:\Users\BM\Desktop\behavior_cloning\predictions.csv")
    # Prefer predicted columns if present; fall back to GT
    gas_col = "pred_gas" if "pred_gas" in df.columns and df["pred_gas"].notna().any() else None
    steer_col = "pred_steering_angle" if "pred_steering_angle" in df.columns and df["pred_steering_angle"].notna().any() else None
    gas = df[gas_col].astype(float).to_numpy()
    steering_angle = df[steer_col].astype(float).to_numpy()

    # Optional: use CSV time column for playback dt; fallback to 10ms
    if "Time" in df.columns:
        time_s = df["Time"].astype(float).to_numpy()
        dt_series = np.diff(time_s, prepend=time_s[0])
        # Guard against non-positive/NaN dts
        dt_series = np.where((dt_series > 0) & np.isfinite(dt_series), dt_series, 0.01)
    else:
        dt_series = np.full_like(gas, 0.01, dtype=float)

    # Clamp gas to valid range
    gas = np.clip(gas, 0.0, 1.0)

    try:
        cmapi.logger.info(f"Gas min/max: {float(np.nanmin(gas)):.3f} / {float(np.nanmax(gas)):.3f}")
        cmapi.logger.info(f"Steer(rad) min/max: {float(np.nanmin(steering_angle)):.4f} / {float(np.nanmax(steering_angle)):.4f}")
    except Exception:
        pass

    for gas_val, steer_val, dt in zip(gas, steering_angle, dt_series):
        simcontrol.simio.dva_write_absolute_value("VC.Gas", float(gas_val), float(dt))
        simcontrol.simio.dva_write_absolute_value("VC.Steer.Ang", float(steer_val), float(dt))
        # cmapi.logger.info(f"Gas: {float(gas_val):.3f}")
        # cmapi.logger.info(f"Steer(rad): {float(steer_val):.4f}")
        
        await asyncio.sleep(float(dt))

    await simcontrol.create_simstate_condition(cmapi.ConditionSimState.finished).wait()
    
    # simcontrol.save_storage_buffer(ms_preceeding=2000, ms_following = 3000)
    simcontrol.stop_storage_save_all()
    await simcontrol.stop_and_disconnect()
    await movie.stop()

    cmapi.logger.info(f"Execution with SimControl finished.")


if __name__ == "__main__":
    cmapi.Task.run_main_task(main())
dlr