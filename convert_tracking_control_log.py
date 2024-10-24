#!/usr/bin/env python3
import json
import os
import re
from collections import defaultdict

import numpy as np
import pandas as pd

# relative to this script
LOG_DIRS = [
    "../local/log_1024_2/转弯第一次",
    "../local/log_1024_2/转弯第二次",
]

SCRIPT_DIR = os.path.dirname(__file__)

# NOTE: leave DATA_PATH undefined to enable log file auto detection
# which finds the .log file automatically.
DATA_PATH = None
# DATA_PATH = os.path.join(SCRIPT_DIR, "log_1234/tracking_control_node.log")

START_PATTERN = re.compile(r"\[(?= INFO| WARN|ERROR)")
POS_PATTERN = re.compile(
    r"""(?x)
        CENTERX:(?P<x_center>-?[\d\.]+),
        CENTERY:(?P<y_center>-?[\d\.]+),
        heading:(?P<heading>-?[\d\.]+),
        valid:(?P<valid>0|1)
    """
)
ANTENNA_POS_PATTERN_LEFT = re.compile(
    r"AntennaCenter_F: x = (?P<x>-?\d*(?:\.\d*)?), y = (?P<y>-?\d*(?:\.\d*)?)"
)
ANTENNA_POS_PATTERN_RIGHT = re.compile(
    r"AntennaCenter_R: x = (?P<x>-?\d*(?:\.\d*)?), y = (?P<y>-?\d*(?:\.\d*)?)"
)
ANTENNA_POS_PATTERN_FRONT = re.compile(
    r"AntennaCenter_VF: x = (?P<x>-?\d*(?:\.\d*)?), y = (?P<y>-?\d*(?:\.\d*)?)"
)
ANTENNA_POS_PATTERN_REAR = re.compile(
    r"AntennaCenter_VR: x = (?P<x>-?\d*(?:\.\d*)?), y = (?P<y>-?\d*(?:\.\d*)?)"
)
TIMESTAMP_PATTERN = re.compile(r"\[(\d+(?:\.\d+))\]")
OMEGA_Z_PATTERN = re.compile(
    r"""(?x)
        Rotate_OmgZ:\ (?P<omega_z>-?\d*(?:\.\d*)?),
        cyclictime:\ (?P<cyclic_time>-?\d*(?:\.\d*)?)
    """
)
TRANS_IN_AGV_PATTERN = re.compile(
    r"transInAGVX: (?P<x>-?\d*(?:\.\d*)?), Y:(?P<y>(?: |-)\d*(?:\.\d*)?)"
)


def convert_log(log_dir: str = "", *, data_path: str = "") -> None:

    if not data_path:
        LOG_DIR = os.path.join(SCRIPT_DIR, log_dir)
        log_files = [
            file_name
            for file_name in os.listdir(LOG_DIR)
            if file_name.startswith("tracking_control_") and file_name.endswith(".log")
        ]
        if len(log_files) <= 0:
            print(
                f"ERROR: Tracking control log file not found in {LOG_DIR!r}!"
            )
            return
        elif len(log_files) > 1:
            print(
                "ERROR: Multiple tracking control log files"
                f"found in {LOG_DIR!r}!"
            )
            return
        data_path = os.path.join(LOG_DIR, log_files[0])
    else:
        LOG_DIR = os.path.dirname(data_path)

    print(f"Converting {log_dir!r}...")

    OUTPUT_PATH = os.path.join(LOG_DIR, "tracking_control_node.csv")

    data = defaultdict[str, list[float | int]](list)
    data_length = defaultdict[str, int](int)

    with open(data_path, "r") as input_file:

        for line in input_file:

            line = line.strip()

            if match_result := START_PATTERN.search(line):
                escape_length = match_result.start(0)
                if escape_length > 0:
                    line = line[escape_length:-escape_length].strip()

            if "input->gExternalEquipData2NS.Antenna_F.rel_x = " in line:
                data["rel_x_f"].append(int(line.split(" = ")[1]))
                data_length["rel_x_f"] += 1

            if "input->gExternalEquipData2NS.Antenna_F.rel_y = " in line:
                data["rel_y_f"].append(int(line.split(" = ")[1]))
                data_length["rel_y_f"] += 1

            if "input->gExternalEquipData2NS.Antenna_R.rel_x = " in line:
                data["rel_x_r"].append(int(line.split(" = ")[1]))
                data_length["rel_x_r"] += 1

            if "input->gExternalEquipData2NS.Antenna_R.rel_y = " in line:
                data["rel_y_r"].append(int(line.split(" = ")[1]))
                data_length["rel_y_r"] += 1

            if (
                "input->gExternalEquipData2NS.Antenna_F.new_valid_data_arrived = "
                in line
            ):
                data["new_valid_data_arrived_f"].append(
                    int(line.split(" = ")[1]))
                data_length["new_valid_data_arrived_f"] += 1

            if (
                "input->gExternalEquipData2NS.Antenna_R.new_valid_data_arrived = "
                in line
            ):
                data["new_valid_data_arrived_r"].append(
                    int(line.split(" = ")[1]))
                data_length["new_valid_data_arrived_r"] += 1

            if "input->gExternalEquipData2NS.Antenna_F.trans_data_get = " in line:
                data["trans_data_get_f"].append(int(line.split(" = ")[1]))
                data_length["trans_data_get_f"] += 1

            if "input->gExternalEquipData2NS.Antenna_R.trans_data_get = " in line:
                data["trans_data_get_r"].append(int(line.split(" = ")[1]))
                data_length["trans_data_get_r"] += 1

            if match_result := OMEGA_Z_PATTERN.search(line):
                data["omega_z"].append(float(match_result.group("omega_z")))
                data_length["omega_z"] += 1

            if "Save Data AGV CENTERX:" in line:
                if match_result := POS_PATTERN.search(line):
                    for key in ("x_center", "y_center", "heading"):
                        data[key].append(float(match_result.group(key)))
                        data_length[key] += 1
                    data["valid"].append(int(match_result.group("valid")))
                    data_length["valid"] += 1

            if match_result := ANTENNA_POS_PATTERN_LEFT.search(line):
                data["x_left"].append(float(match_result.group("x")))
                data["y_left"].append(float(match_result.group("y")))
                data_length["x_left"] += 1
                data_length["y_left"] += 1

            if match_result := ANTENNA_POS_PATTERN_RIGHT.search(line):
                data["x_right"].append(float(match_result.group("x")))
                data["y_right"].append(float(match_result.group("y")))
                data_length["x_right"] += 1
                data_length["y_right"] += 1

            if match_result := ANTENNA_POS_PATTERN_FRONT.search(line):
                data["x_front"].append(float(match_result.group("x")))
                data["y_front"].append(float(match_result.group("y")))
                data_length["x_front"] += 1
                data_length["y_front"] += 1

            if match_result := ANTENNA_POS_PATTERN_REAR.search(line):
                data["x_rear"].append(float(match_result.group("x")))
                data["y_rear"].append(float(match_result.group("y")))
                data_length["x_rear"] += 1
                data_length["y_rear"] += 1

            if match_result := TRANS_IN_AGV_PATTERN.search(line):
                if data_length["trans_in_agv_fx"] <= data_length["trans_in_agv_rx"]:
                    data["trans_in_agv_fx"].append(
                        float(match_result.group("x")))
                    data["trans_in_agv_fy"].append(
                        float(match_result.group("y")))
                    data_length["trans_in_agv_fx"] += 1
                    data_length["trans_in_agv_fy"] += 1
                else:
                    data["trans_in_agv_rx"].append(
                        float(match_result.group("x")))
                    data["trans_in_agv_ry"].append(
                        float(match_result.group("y")))
                    data_length["trans_in_agv_rx"] += 1
                    data_length["trans_in_agv_ry"] += 1

            if "AGVLocalization.Position.AntennaVirtualCenter_F.X = " in line:
                data["x_front_feedforward"].append(float(line.split(" = ")[1]))
                data_length["x_front_feedforward"] += 1

            if "AGVLocalization.Position.AntennaVirtualCenter_F.Y = " in line:
                data["y_front_feedforward"].append(float(line.split(" = ")[1]))
                data_length["y_front_feedforward"] += 1

            if "AGVLocalization.Position.AntennaVirtualCenter_R.X = " in line:
                data["x_rear_feedforward"].append(float(line.split(" = ")[1]))
                data_length["x_rear_feedforward"] += 1

            if "AGVLocalization.Position.AntennaVirtualCenter_R.Y = " in line:
                data["y_rear_feedforward"].append(float(line.split(" = ")[1]))
                data_length["y_rear_feedforward"] += 1

            if "one ann heading:" in line:
                data["heading_one_antenna"].append(float(line.split(":")[-1]))
                data_length["heading_one_antenna"] += 1

            if " heading: " in line:
                data["heading_two_antennas"].append(
                    float(line.split(": ")[-1]))
                data_length["heading_two_antennas"] += 1

            if "2 AGV_MotionStateData.RunningState -->" in line:
                data["running_state"].append(int(line.split("--> ")[1]))
                data_length["running_state"] += 1

            if "AGV_MotionStateData.Speed_AGV.Vs -->" in line:
                data["v_s"].append(float(line.split("--> ")[1]))
                data_length["v_s"] += 1

            if "Wheel_FL.SteerDegree --> " in line:
                data["steer_degree_fl"].append(float(line.split("--> ")[1]))
                data_length["steer_degree_fl"] += 1

            if "Wheel_FR.SteerDegree --> " in line:
                data["steer_degree_fr"].append(float(line.split("--> ")[1]))
                data_length["steer_degree_fr"] += 1

            if "Wheel_RL.SteerDegree --> " in line:
                data["steer_degree_rl"].append(float(line.split("--> ")[1]))
                data_length["steer_degree_rl"] += 1

            if "Wheel_RR.SteerDegree --> " in line:
                data["steer_degree_rr"].append(float(line.split("--> ")[1]))
                data_length["steer_degree_rr"] += 1

            if "Wheel_FS.SteerAngle --> " in line:
                data["steer_angle_fs"].append(float(line.split("--> ")[1]))
                data_length["steer_angle_fs"] += 1

            if "Wheel_RS.SteerAngle --> " in line:
                data["steer_angle_rs"].append(float(line.split("--> ")[1]))
                data_length["steer_angle_rs"] += 1

            if "Wheel_FL.Velocity --> " in line:
                data["v_fl"].append(float(line.split("--> ")[1]))
                data_length["v_fl"] += 1

            if "Wheel_FR.Velocity --> " in line:
                data["v_fr"].append(float(line.split("--> ")[1]))
                data_length["v_fr"] += 1

            if "Wheel_RL.Velocity --> " in line:
                data["v_rl"].append(float(line.split("--> ")[1]))
                data_length["v_rl"] += 1

            if "Wheel_RR.Velocity --> " in line:
                data["v_rr"].append(float(line.split("--> ")[1]))
                data_length["v_rr"] += 1

            if "Wheel_FS.Velocity --> " in line:
                data["v_fs"].append(float(line.split("--> ")[1]))
                data_length["v_fs"] += 1

            if "Wheel_RS.Velocity --> " in line:
                data["v_rs"].append(float(line.split("--> ")[1]))
                data_length["v_rs"] += 1

            if "AGV_MotionStateData.Speed_Global.Vx --> " in line:
                data["v_gx"].append(float(line.split("--> ")[1]))
                data_length["v_gx"] += 1

            if "AGV_MotionStateData.Speed_Global.Vy --> " in line:
                data["v_gy"].append(float(line.split("--> ")[1]))
                data_length["v_gy"] += 1

            if "AGV_MotionStateData.Speed_Global.Vs --> " in line:
                data["v_gs"].append(float(line.split("--> ")[1]))
                data_length["v_gs"] += 1

            if "gFrontRoute.PathIndex_offsetAngle = " in line:
                data["front_index_angle"].append(
                    int(line.split(" = ")[1])
                )
                data_length["front_index_angle"] += 1

            if "gFrontRoute.PathIndex_offsetDistance = " in line:
                data["front_index_distance"].append(
                    int(line.split(" = ")[1])
                )
                data_length["front_index_distance"] += 1

            if "gGetRouteIndexFlag_F = " in line:
                data["front_index_flag"].append(int(line.split(" = ")[1]))
                data_length["front_index_flag"] += 1

            if "gRearRoute.PathIndex_offsetAngle = " in line:
                data["rear_index_angle"].append(
                    int(line.split(" = ")[1])
                )
                data_length["rear_index_angle"] += 1

            if "gRearRoute.PathIndex_offsetDistance = " in line:
                data["rear_index_distance"].append(
                    int(line.split(" = ")[1])
                )
                data_length["rear_index_distance"] += 1

            if "gGetRouteIndexFlag_R = " in line:
                data["rear_index_flag"].append(int(line.split(" = ")[1]))
                data_length["rear_index_flag"] += 1

            if "MotionControlData.RunDirection = " in line:
                data["run_direction"].append(int(line.split(" = ")[1]))
                data_length["run_direction"] += 1

            if "MotionControl::GetOffsetToTargetTrajectory::OffsetY_ToTargetNow == " in line:
                if data_length["offset_y_front"] <= data_length["offset_y_rear"]:
                    data["offset_y_front"].append(float(line.split(" == ")[1]))
                    data_length["offset_y_front"] += 1
                else:
                    data["offset_y_rear"].append(float(line.split(" == ")[1]))
                    data_length["offset_y_rear"] += 1

            if "MotionControl::GetOffsetToTargetTrajectory::OffsetX_ToEndPoint == " in line:
                if data_length["offset_x_front"] <= data_length["offset_x_rear"]:
                    data["offset_x_front"].append(float(line.split(" == ")[1]))
                    data_length["offset_x_front"] += 1
                else:
                    data["offset_x_rear"].append(float(line.split(" == ")[1]))
                    data_length["offset_x_rear"] += 1

            if "PathOffset_FrontAnn.AheadPointAngle = " in line:
                data["ahead_point_angle_front"].append(
                    float(line.split(" = ")[1]))
                data_length["ahead_point_angle_front"] += 1

            if "PathOffset_FrontAnn.DiffAngleToTarget = " in line:
                data["diff_angle_to_target_front"].append(
                    float(line.split(" = ")[1])
                )
                data_length["diff_angle_to_target_front"] += 1

            if "PathOffset_RearAnn.AheadPointAngle = " in line:
                data["ahead_point_angle_rear"].append(
                    float(line.split(" = ")[1]))
                data_length["ahead_point_angle_rear"] += 1

            if "PathOffset_RearAnn.DiffAngleToTarget = " in line:
                data["diff_angle_to_target_rear"].append(
                    float(line.split(" = ")[1])
                )
                data_length["diff_angle_to_target_rear"] += 1

            if "GetTargetActiveVelocity::lcr_ramp_velocity.x == " in line:
                data["ramp_x"].append(float(line.split(" == ")[1]))
                data_length["ramp_x"] += 1

            # if "MotionControl::PID_Controller::kp = " in line:
            if "MotionControl::NewSetPIDControllerParameter : *kp = " in line:
                if data_length["kp_f"] <= data_length["kp_r"]:
                    data["kp_f"].append(float(line.split(" = ")[1]))
                    data_length["kp_f"] += 1
                else:
                    data["kp_r"].append(float(line.split(" = ")[1]))
                    data_length["kp_r"] += 1

            # if "MotionControl::PID_Controller::ki = " in line:
            if "MotionControl::NewSetPIDControllerParameter : *ki = " in line:
                if data_length["ki_f"] <= data_length["ki_r"]:
                    data["ki_f"].append(float(line.split(" = ")[1]))
                    data_length["ki_f"] += 1
                else:
                    data["ki_r"].append(float(line.split(" = ")[1]))
                    data_length["ki_r"] += 1

            # if "MotionControl::PID_Controller::kd = " in line:
            if "MotionControl::NewSetPIDControllerParameter : *kd = " in line:
                if data_length["kd_f"] <= data_length["kd_r"]:
                    data["kd_f"].append(float(line.split(" = ")[1]))
                    data_length["kd_f"] += 1
                else:
                    data["kd_r"].append(float(line.split(" = ")[1]))
                    data_length["kd_r"] += 1

            # if "GetAntennaTargetAngleAndVelocity::vy_cross_f == " in line:
            #     if "(before ValueLimit)" in line:
            #         data["vy_cross_f_raw"].append(
            #             float(line[:-20].split(" == ")[1])
            #         )
            #         data_length["vy_cross_f_raw"] += 1
            #     elif "(after ValueLimit)" in line:
            #         data["vy_cross_f"].append(float(line[:-19].split(" == ")[1]))
            #         data_length["vy_cross_f"] += 1

            # if "GetAntennaTargetAngleAndVelocity::vy_cross_r == " in line:
            #     if "(before ValueLimit)" in line:
            #         data["vy_cross_r_raw"].append(
            #             float(line[:-20].split(" == ")[1])
            #         )
            #         data_length["vy_cross_r_raw"] += 1
            #     elif "(after ValueLimit)" in line:
            #         data["vy_cross_r"].append(float(line[:-19].split(" == ")[1]))
            #         data_length["vy_cross_r"] += 1

            if "MotionControl::NewGetAntennaTargetAngleAndVelocity::targetData_F->SteerAngle_FeedBack == " in line:
                data["steer_angle_feedback_f"].append(
                    float(line.split(" == ")[1]))
                data_length["steer_angle_feedback_f"] += 1

            if "MotionControl::NewGetAntennaTargetAngleAndVelocity::targetData_F->SteerAngle == " in line:
                data["pid_steer_angle_f"].append(float(line.split(" == ")[1]))
                data_length["pid_steer_angle_f"] += 1

            if "MotionControl::NewGetAntennaTargetAngleAndVelocity::targetData_R->SteerAngle_FeedBack == " in line:
                data["steer_angle_feedback_r"].append(
                    float(line.split(" == ")[1]))
                data_length["steer_angle_feedback_r"] += 1

            if "MotionControl::NewGetAntennaTargetAngleAndVelocity::targetData_R->SteerAngle == " in line:
                data["pid_steer_angle_r"].append(float(line.split(" == ")[1]))
                data_length["pid_steer_angle_r"] += 1

            if "GeTargetInactiveData::beta == " in line:
                data["beta"].append(float(line.split(" == ")[1]))
                data_length["beta"] += 1

            if "MotionControlData.TargetPosArrived = " in line:
                data["target_pos_arrived"].append(int(line.split(" = ")[1]))
                data_length["target_pos_arrived"] += 1

            if "MotionControlData.Command.SteerAngle_FS = " in line:
                data["command_steer_angle_fs"].append(
                    int(line.split(" = ")[1]))
                data_length["command_steer_angle_fs"] += 1

            if "MotionControlData.Command.SteerAngle_FL = " in line:
                data["command_steer_angle_fl"].append(
                    int(line.split(" = ")[1]))
                data_length["command_steer_angle_fl"] += 1

            if "MotionControlData.Command.SteerAngle_FR = " in line:
                data["command_steer_angle_fr"].append(
                    int(line.split(" = ")[1]))
                data_length["command_steer_angle_fr"] += 1

            if "MotionControlData.Command.SteerAngle_RS = " in line:
                data["command_steer_angle_rs"].append(
                    int(line.split(" = ")[1]))
                data_length["command_steer_angle_rs"] += 1

            if "MotionControlData.Command.SteerAngle_RL = " in line:
                data["command_steer_angle_rl"].append(
                    int(line.split(" = ")[1]))
                data_length["command_steer_angle_rl"] += 1

            if "MotionControlData.Command.SteerAngle_RR = " in line:
                data["command_steer_angle_rr"].append(
                    int(line.split(" = ")[1]))
                data_length["command_steer_angle_rr"] += 1

            if "MotionControlData.Command.Speed_Front_RPM = " in line:
                data["command_speed_fs_rpm"].append(int(line.split(" = ")[1]))
                data_length["command_speed_fs_rpm"] += 1

            if "MotionControlData.Command.Speed_FL_RPM = " in line:
                data["command_speed_fl_rpm"].append(int(line.split(" = ")[1]))
                data_length["command_speed_fl_rpm"] += 1

            if "MotionControlData.Command.Speed_FR_RPM = " in line:
                data["command_speed_fr_rpm"].append(int(line.split(" = ")[1]))
                data_length["command_speed_fr_rpm"] += 1

            if "MotionControlData.Command.Speed_Rear_RPM = " in line:
                data["command_speed_rs_rpm"].append(int(line.split(" = ")[1]))
                data_length["command_speed_rs_rpm"] += 1

            if "MotionControlData.Command.Speed_RL_RPM = " in line:
                data["command_speed_rl_rpm"].append(int(line.split(" = ")[1]))
                data_length["command_speed_rl_rpm"] += 1

            if "MotionControlData.Command.Speed_RR_RPM = " in line:
                data["command_speed_rr_rpm"].append(int(line.split(" = ")[1]))
                data_length["command_speed_rr_rpm"] += 1

            if "MotionControlData.Command.Brake --> 1" in line:
                data["command_brake"].append(1)
                data_length["command_brake"] += 1

            if "MotionControlData.Command.Brake --> 0" in line:
                data["command_brake"].append(0)
                data_length["command_brake"] += 1
                # data["command_steer_angle_fl"].append(0)
                # data["command_steer_angle_fr"].append(0)
                # data["command_steer_angle_rl"].append(0)
                # data["command_steer_angle_rr"].append(0)
                # data["command_speed_fl_rpm"].append(0)
                # data["command_speed_fr_rpm"].append(0)
                # data["command_speed_rl_rpm"].append(0)
                # data["command_speed_rr_rpm"].append(0)
                # data_length["command_steer_angle_fl"] += 1
                # data_length["command_steer_angle_fr"] += 1
                # data_length["command_steer_angle_rl"] += 1
                # data_length["command_steer_angle_rr"] += 1
                # data_length["command_speed_fl_rpm"] += 1
                # data_length["command_speed_fr_rpm"] += 1
                # data_length["command_speed_rl_rpm"] += 1
                # data_length["command_speed_rr_rpm"] += 1

            if "motion_control::Cyclic() end" in line:
                if match_result := TIMESTAMP_PATTERN.search(line):
                    timestamp = float(match_result.group(1))
                    data["timestamp"].append(timestamp)
                    data_length["timestamp"] += 1
                    timestamp_count = data_length["timestamp"]
                    for key in data_length:
                        if key == "timestamp":
                            continue
                        if data_length[key] > timestamp_count:
                            last_element = data[key][-1]
                            data[key] = data[key][: timestamp_count - 1]
                            data[key].append(last_element)
                            data_length[key] = timestamp_count
                        else:
                            while data_length[key] < timestamp_count:
                                data[key].append(np.nan)
                                data_length[key] += 1

    min_length = min(data_length.values())
    if any(length > min_length for length in data_length.values()):
        print(
            f"WARNING: ({log_dir}) Some sequence(s) will be truncated "
            "due to inconsistent data length:"
        )
        print(
            json.dumps(
                {
                    key: value
                    for key, value in data_length.items()
                    if value != min_length
                },
                indent=2,
            )
        )
        for key in data:
            if len(data[key]) > min_length:
                data[key] = data[key][:min_length]

    print(f"Generated {min_length:,} lines of data from {log_dir!r}.")

    df_output = pd.DataFrame(
        zip(*data.values()),
        columns=tuple(data.keys()),
    )
    df_output["timestamp"] = pd.to_datetime(
        df_output["timestamp"], unit="s"
    ) + pd.Timedelta(8, "h")
    df_output.to_csv(OUTPUT_PATH)


if __name__ == "__main__":

    if "DATA_PATH" in locals() and DATA_PATH:
        convert_log(data_path=DATA_PATH)
    else:

        from multiprocessing import Pool

        process_count = min(len(LOG_DIRS), 8)
        with Pool(process_count) as pool:
            pool.map(convert_log, LOG_DIRS)
        print("Done.")
