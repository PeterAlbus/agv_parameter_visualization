#!/usr/bin/env python3
import json
import os
import re
import shutil
from collections import defaultdict

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(__file__)
NOTEBOOK_FILE_NAME = "tracking_control_node.ipynb"
NOTEBOOK_PATH = os.path.join(SCRIPT_DIR, NOTEBOOK_FILE_NAME)
OUTPUT_FILE_NAME = "tracking_control_node.csv"

# relative to this script
# LOG_DIRS = [
#     "../local/log/1113_8_agv",
# ]
LOG_PARENT_DIR = os.path.join(SCRIPT_DIR, "../local/log/1209")
LOG_DIRS = [
    path
    for path in filter(
        os.path.isdir,
        [
            os.path.join(LOG_PARENT_DIR, dir_name)
            for dir_name in os.listdir(LOG_PARENT_DIR)
            if not dir_name.startswith("_")
        ]
    )
    if OUTPUT_FILE_NAME not in os.listdir(path)
]

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
ESTIMATE_PATTERN = re.compile(
    r"""(?x)
        StateEstimate\s*CENTERX:\s*(?P<x_estimate>-?[\d\.]+),
        CENTERY:\s*(?P<y_estimate>-?[\d\.]+),
        heading:\s*(?P<heading_estimate>-?[\d\.]+)
    """
)
ORIGINAL_POS_PATTERN = re.compile(
    r"""(?x)
        StateCorrect\ Final\ output\s*CENTERX:\s*(?P<x_original>-?[\d\.]+),
        CENTERY:\s*(?P<y_original>-?[\d\.]+),
        heading:\s*(?P<heading_original>-?[\d\.]+)
    """
)
GNSS_PATTERN = re.compile(
    r"""(?x)
        INS\ coordinates\ by\ GNSS:\ gnss_X=(?P<x_gnss>-?[\d\.]+),
        \ gnss_Y=(?P<y_gnss>-?[\d\.]+)
    """
)
ANTENNA_POS_PATTERN_F = re.compile(
    r"AntennaCenter_F: x = (?P<x>-?\d+(?:\.\d+)?), y = (?P<y>-?\d+(?:\.\d+)?)"
)
ANTENNA_POS_PATTERN_R = re.compile(
    r"AntennaCenter_R: x = (?P<x>-?\d+(?:\.\d+)?), y = (?P<y>-?\d+(?:\.\d+)?)"
)
ANTENNA_POS_PATTERN_VF = re.compile(
    r"AntennaCenter_VF: x = (?P<x>-?\d+(?:\.\d+)?), y = (?P<y>-?\d+(?:\.\d+)?)"
)
ANTENNA_POS_PATTERN_VR = re.compile(
    r"AntennaCenter_VR: x = (?P<x>-?\d+(?:\.\d+)?), y = (?P<y>-?\d+(?:\.\d+)?)"
)
TIMESTAMP_PATTERN = re.compile(r"\[(\d+(?:\.\d+))\]")
OMEGA_Z_PATTERN = re.compile(
    r"""(?x)
        Rotate_OmgZ:\ (?P<omega_z>-?\d+(?:\.\d+)?),
        cyclictime:\ (?P<cyclic_time>-?\d+(?:\.\d+)?)
    """
)
TRANS_IN_AGV_PATTERN = re.compile(
    r"transInAGVX: (?P<x>-?\d+(?:\.\d+)?), Y:(?P<y>(?: |-)\d+(?:\.\d+)?)"
)
FUSION_LOCALIZATION_FLAG = "[FusionLocalization]"


def convert_log(log_dir: str = "", *, data_path: str = "") -> None:

    if not data_path:
        LOG_DIR = os.path.join(SCRIPT_DIR, log_dir)
        log_files = [
            file_name
            for file_name in os.listdir(LOG_DIR)
            if file_name.endswith(".log")
        ]
        if len(log_files) > 1:
            log_files = [
                file_name
                for file_name in log_files
                if "tracking_control" in file_name
            ]
        if len(log_files) <= 0:
            print(
                f'ERROR: Tracking control log file not found in "{LOG_DIR}"!'
            )
            return
        elif len(log_files) > 1:
            print(
                "ERROR: Multiple tracking control log files"
                f'found in "{LOG_DIR}"!'
            )
            return
        data_path = os.path.join(LOG_DIR, log_files[0])
    else:
        LOG_DIR = os.path.dirname(data_path)

    print(f'Converting "{os.path.relpath(log_dir, SCRIPT_DIR)}"...')

    OUTPUT_PATH = os.path.join(LOG_DIR, OUTPUT_FILE_NAME)

    data = defaultdict[str, list[float | int]](list)
    data_length = defaultdict[str, int](int)

    with open(data_path, "r", encoding="utf-8") as input_file:

        for line in input_file:

            line = line.strip()

            if match_result := START_PATTERN.search(line):
                escape_length = match_result.start(0)
                if escape_length > 0:
                    line = line[escape_length:-escape_length].strip()

            if "input->gExternalEquipData2NS.Antenna_F.rel_x = " in line:
                data["rel_x_f"].append(int(line.split(" = ")[1]))
                data_length["rel_x_f"] += 1
                continue

            if "input->gExternalEquipData2NS.Antenna_F.rel_y = " in line:
                data["rel_y_f"].append(int(line.split(" = ")[1]))
                data_length["rel_y_f"] += 1
                continue

            if "input->gExternalEquipData2NS.Antenna_R.rel_x = " in line:
                data["rel_x_r"].append(int(line.split(" = ")[1]))
                data_length["rel_x_r"] += 1
                continue

            if "input->gExternalEquipData2NS.Antenna_R.rel_y = " in line:
                data["rel_y_r"].append(int(line.split(" = ")[1]))
                data_length["rel_y_r"] += 1
                continue

            if (
                "input->gExternalEquipData2NS.Antenna_F.new_valid_data_arrived = "
                in line
            ):
                data["new_valid_data_arrived_f"].append(
                    int(line.split(" = ")[1]))
                data_length["new_valid_data_arrived_f"] += 1
                continue

            if (
                "input->gExternalEquipData2NS.Antenna_R.new_valid_data_arrived = "
                in line
            ):
                data["new_valid_data_arrived_r"].append(
                    int(line.split(" = ")[1]))
                data_length["new_valid_data_arrived_r"] += 1
                continue

            if "input->gExternalEquipData2NS.Antenna_F.trans_data_get = " in line:
                data["trans_data_get_f"].append(int(line.split(" = ")[1]))
                data_length["trans_data_get_f"] += 1
                continue

            if "input->gExternalEquipData2NS.Antenna_R.trans_data_get = " in line:
                data["trans_data_get_r"].append(int(line.split(" = ")[1]))
                data_length["trans_data_get_r"] += 1
                continue

            if match_result := OMEGA_Z_PATTERN.search(line):
                data["omega_z"].append(float(match_result.group("omega_z")))
                data_length["omega_z"] += 1
                continue

            if "Save Data AGV CENTERX:" in line:
                if match_result := POS_PATTERN.search(line):
                    for key in ("x_center", "y_center", "heading"):
                        data[key].append(float(match_result.group(key)))
                        data_length[key] += 1
                    data["valid"].append(int(match_result.group("valid")))
                    data_length["valid"] += 1
                    continue

            if "StateEstimate CENTERX: " in line:
                if match_result := ESTIMATE_PATTERN.search(line):
                    for key in ("x_estimate", "y_estimate", "heading_estimate"):
                        data[key].append(float(match_result.group(key)))
                        data_length[key] += 1
                    continue

            if "StateCorrect Final output CENTERX: " in line:
                if match_result := ORIGINAL_POS_PATTERN.search(line):
                    for key in ("x_original", "y_original", "heading_original"):
                        data[key].append(float(match_result.group(key)))
                        data_length[key] += 1
                    continue

            if "INS coordinates by GNSS: gnss_X=" in line:
                if match_result := GNSS_PATTERN.search(line):
                    for key in ("x_gnss", "y_gnss"):
                        data[key].append(float(match_result.group(key)))
                        data_length[key] += 1
                    continue

            if match_result := ANTENNA_POS_PATTERN_F.search(line):
                data["x_f"].append(float(match_result.group("x")))
                data["y_f"].append(float(match_result.group("y")))
                data_length["x_f"] += 1
                data_length["y_f"] += 1
                continue

            if match_result := ANTENNA_POS_PATTERN_R.search(line):
                data["x_r"].append(float(match_result.group("x")))
                data["y_r"].append(float(match_result.group("y")))
                data_length["x_r"] += 1
                data_length["y_r"] += 1
                continue

            if match_result := ANTENNA_POS_PATTERN_VF.search(line):
                data["x_vf"].append(float(match_result.group("x")))
                data["y_vf"].append(float(match_result.group("y")))
                data_length["x_vf"] += 1
                data_length["y_vf"] += 1
                continue

            if match_result := ANTENNA_POS_PATTERN_VR.search(line):
                data["x_vr"].append(float(match_result.group("x")))
                data["y_vr"].append(float(match_result.group("y")))
                data_length["x_vr"] += 1
                data_length["y_vr"] += 1
                continue

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
                continue

            if "AGVLocalization.Position.AntennaVirtualCenter_F.X = " in line:
                data["x_front_feedforward"].append(float(line.split(" = ")[1]))
                data_length["x_front_feedforward"] += 1
                continue

            if "AGVLocalization.Position.AntennaVirtualCenter_F.Y = " in line:
                data["y_front_feedforward"].append(float(line.split(" = ")[1]))
                data_length["y_front_feedforward"] += 1
                continue

            if "AGVLocalization.Position.AntennaVirtualCenter_R.X = " in line:
                data["x_rear_feedforward"].append(float(line.split(" = ")[1]))
                data_length["x_rear_feedforward"] += 1
                continue

            if "AGVLocalization.Position.AntennaVirtualCenter_R.Y = " in line:
                data["y_rear_feedforward"].append(float(line.split(" = ")[1]))
                data_length["y_rear_feedforward"] += 1
                continue

            if "one ann heading:" in line:
                data["heading_one_antenna"].append(float(line.split(":")[-1]))
                data_length["heading_one_antenna"] += 1
                continue

            if " heading: " in line:
                data["heading_two_antennas"].append(
                    float(line.split(": ")[-1]))
                data_length["heading_two_antennas"] += 1
                continue

            if "FusionLocalization::result.Available = " in line:
                data["fusion_available"].append(int(line.split(" = ")[1]))
                data_length["fusion_available"] += 1
                continue

            if "FusionLocalization::result.VehicleCenter.X = " in line:
                data["x_fusion"].append(float(line.split(" = ")[1]))
                data_length["x_fusion"] += 1
                continue

            if "FusionLocalization::result.VehicleCenter.Y = " in line:
                data["y_fusion"].append(float(line.split(" = ")[1]))
                data_length["y_fusion"] += 1
                continue

            if "AGV_MotionStateData.RunningState --> " in line:
                data["running_state"].append(int(line.split(" --> ")[-1]))
                data_length["running_state"] += 1
                continue

            if "AGV_MotionStateData.RunningState: " in line:
                data["running_state"].append(int(line.split(": ")[-1]))
                data_length["running_state"] += 1
                continue

            if "AGV_MotionStateData.Speed_AGV.Vs -->" in line:
                data["v_s"].append(float(line.split("--> ")[1]))
                data_length["v_s"] += 1
                continue

            if "Wheel_FL.SteerDegree --> " in line:
                data["steer_degree_fl"].append(float(line.split("--> ")[1]))
                data_length["steer_degree_fl"] += 1
                continue

            if "Wheel_FR.SteerDegree --> " in line:
                data["steer_degree_fr"].append(float(line.split("--> ")[1]))
                data_length["steer_degree_fr"] += 1
                continue

            if "Wheel_RL.SteerDegree --> " in line:
                data["steer_degree_rl"].append(float(line.split("--> ")[1]))
                data_length["steer_degree_rl"] += 1
                continue

            if "Wheel_RR.SteerDegree --> " in line:
                data["steer_degree_rr"].append(float(line.split("--> ")[1]))
                data_length["steer_degree_rr"] += 1
                continue

            if "Wheel_FS.SteerAngle --> " in line:
                data["steer_angle_fs"].append(float(line.split("--> ")[1]))
                data_length["steer_angle_fs"] += 1
                continue

            if "Wheel_RS.SteerAngle --> " in line:
                data["steer_angle_rs"].append(float(line.split("--> ")[1]))
                data_length["steer_angle_rs"] += 1
                continue

            if "Wheel_FL.Velocity --> " in line:
                data["v_fl"].append(float(line.split("--> ")[1]))
                data_length["v_fl"] += 1
                continue

            if "Wheel_FR.Velocity --> " in line:
                data["v_fr"].append(float(line.split("--> ")[1]))
                data_length["v_fr"] += 1
                continue

            if "Wheel_RL.Velocity --> " in line:
                data["v_rl"].append(float(line.split("--> ")[1]))
                data_length["v_rl"] += 1
                continue

            if "Wheel_RR.Velocity --> " in line:
                data["v_rr"].append(float(line.split("--> ")[1]))
                data_length["v_rr"] += 1
                continue

            if "Wheel_FS.Velocity --> " in line:
                data["v_fs"].append(float(line.split("--> ")[1]))
                data_length["v_fs"] += 1
                continue

            if "Wheel_RS.Velocity --> " in line:
                data["v_rs"].append(float(line.split("--> ")[1]))
                data_length["v_rs"] += 1
                continue

            if "AGV_MotionStateData.Speed_Global.Vx --> " in line:
                data["v_gx"].append(float(line.split("--> ")[1]))
                data_length["v_gx"] += 1
                continue

            if "AGV_MotionStateData.Speed_Global.Vy --> " in line:
                data["v_gy"].append(float(line.split("--> ")[1]))
                data_length["v_gy"] += 1
                continue

            if "AGV_MotionStateData.Speed_Global.Vs --> " in line:
                data["v_gs"].append(float(line.split("--> ")[1]))
                data_length["v_gs"] += 1
                continue

            if "gFrontRoute.PathIndex_offsetAngle = " in line:
                data["front_index_angle"].append(
                    int(line.split(" = ")[1])
                )
                data_length["front_index_angle"] += 1
                continue

            if "gFrontRoute.PathIndex_offsetDistance = " in line:
                data["front_index_distance"].append(
                    int(line.split(" = ")[1])
                )
                data_length["front_index_distance"] += 1
                continue

            if "gGetRouteIndexFlag_F = " in line:
                data["front_index_flag"].append(int(line.split(" = ")[1]))
                data_length["front_index_flag"] += 1
                continue

            if "gRearRoute.PathIndex_offsetAngle = " in line:
                data["rear_index_angle"].append(
                    int(line.split(" = ")[1])
                )
                data_length["rear_index_angle"] += 1
                continue

            if "gRearRoute.PathIndex_offsetDistance = " in line:
                data["rear_index_distance"].append(
                    int(line.split(" = ")[1])
                )
                data_length["rear_index_distance"] += 1
                continue

            if "gGetRouteIndexFlag_R = " in line:
                data["rear_index_flag"].append(int(line.split(" = ")[1]))
                data_length["rear_index_flag"] += 1
                continue

            if "MotionControlData.RunDirection = " in line:
                data["run_direction"].append(int(line.split(" = ")[1]))
                data_length["run_direction"] += 1
                continue

            if "GetOffsetToTargetTrajectory::OffsetY_ToTargetNow == " in line:
                if data_length["offset_y_front"] <= data_length["offset_y_rear"]:
                    data["offset_y_front"].append(float(line.split(" == ")[1]))
                    data_length["offset_y_front"] += 1
                else:
                    data["offset_y_rear"].append(float(line.split(" == ")[1]))
                    data_length["offset_y_rear"] += 1
                continue

            if "GetOffsetToTargetTrajectory::OffsetX_ToEndPoint == " in line:
                if data_length["offset_x_front"] <= data_length["offset_x_rear"]:
                    data["offset_x_front"].append(float(line.split(" == ")[1]))
                    data_length["offset_x_front"] += 1
                else:
                    data["offset_x_rear"].append(float(line.split(" == ")[1]))
                    data_length["offset_x_rear"] += 1
                continue

            if "PathOffset_FrontAnn.AheadPointAngle = " in line:
                data["ahead_point_angle_front"].append(
                    float(line.split(" = ")[1]))
                data_length["ahead_point_angle_front"] += 1
                continue

            if "PathOffset_FrontAnn.DiffAngleToTarget = " in line:
                data["diff_angle_to_target_front"].append(
                    float(line.split(" = ")[1])
                )
                data_length["diff_angle_to_target_front"] += 1
                continue

            if "PathOffset_RearAnn.AheadPointAngle = " in line:
                data["ahead_point_angle_rear"].append(
                    float(line.split(" = ")[1]))
                data_length["ahead_point_angle_rear"] += 1
                continue

            if "PathOffset_RearAnn.DiffAngleToTarget = " in line:
                data["diff_angle_to_target_rear"].append(
                    float(line.split(" = ")[1])
                )
                data_length["diff_angle_to_target_rear"] += 1
                continue

            if "GetTargetActiveVelocity::lcr_ramp_velocity.x == " in line:
                data["ramp_x"].append(float(line.split(" == ")[1]))
                data_length["ramp_x"] += 1
                continue

            # if "PID_Controller::kp = " in line:
            if "NewSetPIDControllerParameter : *kp = " in line:
                if data_length["kp_f"] <= data_length["kp_r"]:
                    data["kp_f"].append(float(line.split(" = ")[1]))
                    data_length["kp_f"] += 1
                else:
                    data["kp_r"].append(float(line.split(" = ")[1]))
                    data_length["kp_r"] += 1
                continue

            # if "PID_Controller::ki = " in line:
            if "NewSetPIDControllerParameter : *ki = " in line:
                if data_length["ki_f"] <= data_length["ki_r"]:
                    data["ki_f"].append(float(line.split(" = ")[1]))
                    data_length["ki_f"] += 1
                else:
                    data["ki_r"].append(float(line.split(" = ")[1]))
                    data_length["ki_r"] += 1
                continue

            # if "PID_Controller::kd = " in line:
            if "NewSetPIDControllerParameter : *kd = " in line:
                if data_length["kd_f"] <= data_length["kd_r"]:
                    data["kd_f"].append(float(line.split(" = ")[1]))
                    data_length["kd_f"] += 1
                else:
                    data["kd_r"].append(float(line.split(" = ")[1]))
                    data_length["kd_r"] += 1
                continue

            # if "GetAntennaTargetAngleAndVelocity::vy_cross_f == " in line:
            #     if "(before ValueLimit)" in line:
            #         data["vy_cross_f_raw"].append(
            #             float(line[:-20].split(" == ")[1])
            #         )
            #         data_length["vy_cross_f_raw"] += 1
            #     elif "(after ValueLimit)" in line:
            #         data["vy_cross_f"].append(float(line[:-19].split(" == ")[1]))
            #         data_length["vy_cross_f"] += 1
            #     continue

            # if "GetAntennaTargetAngleAndVelocity::vy_cross_r == " in line:
            #     if "(before ValueLimit)" in line:
            #         data["vy_cross_r_raw"].append(
            #             float(line[:-20].split(" == ")[1])
            #         )
            #         data_length["vy_cross_r_raw"] += 1
            #     elif "(after ValueLimit)" in line:
            #         data["vy_cross_r"].append(float(line[:-19].split(" == ")[1]))
            #         data_length["vy_cross_r"] += 1
            #     continue

            if "NewGetAntennaTargetAngleAndVelocity::targetData_F->SteerAngle_FeedBack == " in line:
                data["steer_angle_feedback_f"].append(
                    float(line.split(" == ")[1]))
                data_length["steer_angle_feedback_f"] += 1
                continue

            if "NewGetAntennaTargetAngleAndVelocity::targetData_F->SteerAngle == " in line:
                data["pid_steer_angle_f"].append(float(line.split(" == ")[1]))
                data_length["pid_steer_angle_f"] += 1
                continue

            if "NewGetAntennaTargetAngleAndVelocity::targetData_R->SteerAngle_FeedBack == " in line:
                data["steer_angle_feedback_r"].append(
                    float(line.split(" == ")[1]))
                data_length["steer_angle_feedback_r"] += 1
                continue

            if "NewGetAntennaTargetAngleAndVelocity::targetData_R->SteerAngle == " in line:
                data["pid_steer_angle_r"].append(float(line.split(" == ")[1]))
                data_length["pid_steer_angle_r"] += 1
                continue

            if "GeTargetInactiveData::beta == " in line:
                data["beta"].append(float(line.split(" == ")[1]))
                data_length["beta"] += 1
                continue

            if "MotionControlData.TargetPosArrived = " in line:
                data["target_pos_arrived"].append(int(line.split(" = ")[1]))
                data_length["target_pos_arrived"] += 1
                continue

            if "MotionControlData.Command.SteerAngle_FS = " in line:
                data["command_steer_angle_fs"].append(
                    int(line.split(" = ")[1]))
                data_length["command_steer_angle_fs"] += 1
                continue

            if "MotionControlData.Command.SteerAngle_FL = " in line:
                data["command_steer_angle_fl"].append(
                    int(line.split(" = ")[1]))
                data_length["command_steer_angle_fl"] += 1
                continue

            if "MotionControlData.Command.SteerAngle_FR = " in line:
                data["command_steer_angle_fr"].append(
                    int(line.split(" = ")[1]))
                data_length["command_steer_angle_fr"] += 1
                continue

            if "MotionControlData.Command.SteerAngle_RS = " in line:
                data["command_steer_angle_rs"].append(
                    int(line.split(" = ")[1]))
                data_length["command_steer_angle_rs"] += 1
                continue

            if "MotionControlData.Command.SteerAngle_RL = " in line:
                data["command_steer_angle_rl"].append(
                    int(line.split(" = ")[1]))
                data_length["command_steer_angle_rl"] += 1
                continue

            if "MotionControlData.Command.SteerAngle_RR = " in line:
                data["command_steer_angle_rr"].append(
                    int(line.split(" = ")[1]))
                data_length["command_steer_angle_rr"] += 1
                continue

            if "MotionControlData.Command.Speed_Front_RPM = " in line:
                data["command_speed_fs_rpm"].append(int(line.split(" = ")[1]))
                data_length["command_speed_fs_rpm"] += 1
                continue

            if "MotionControlData.Command.Speed_FL_RPM = " in line:
                data["command_speed_fl_rpm"].append(int(line.split(" = ")[1]))
                data_length["command_speed_fl_rpm"] += 1
                continue

            if "MotionControlData.Command.Speed_FR_RPM = " in line:
                data["command_speed_fr_rpm"].append(int(line.split(" = ")[1]))
                data_length["command_speed_fr_rpm"] += 1
                continue

            if "MotionControlData.Command.Speed_Rear_RPM = " in line:
                data["command_speed_rs_rpm"].append(int(line.split(" = ")[1]))
                data_length["command_speed_rs_rpm"] += 1
                continue

            if "MotionControlData.Command.Speed_RL_RPM = " in line:
                data["command_speed_rl_rpm"].append(int(line.split(" = ")[1]))
                data_length["command_speed_rl_rpm"] += 1
                continue

            if "MotionControlData.Command.Speed_RR_RPM = " in line:
                data["command_speed_rr_rpm"].append(int(line.split(" = ")[1]))
                data_length["command_speed_rr_rpm"] += 1
                continue

            if "MotionControlData.Command.Brake = " in line:
                data["command_brake"].append(int(line.split(" = ")[1]))
                data_length["command_brake"] += 1
                continue

            if "ReflectorCorrect : reflector_distance_front = " in line:
                data["reflector_distance_front"].append(
                    float(line.split(" = ")[1])
                )
                data_length["reflector_distance_front"] += 1
                continue

            if "ReflectorCorrect : reflector_distance_rear = " in line:
                data["reflector_distance_rear"].append(
                    float(line.split(" = ")[1])
                )
                data_length["reflector_distance_rear"] += 1
                continue

            if "ReflectorCorrect : delta_angle = " in line:
                data["reflector_delta_angle"].append(
                    float(line.split(" = ")[1])
                )
                data_length["reflector_delta_angle"] += 1
                continue

            if "ReflectorCorrect : vehicle_heading_new = " in line:
                data["heading_reflector"].append(float(line.split(" = ")[1]))
                data_length["heading_reflector"] += 1
                continue

            if "ReflectorCorrect : new_center_x = " in line:
                data["reflector_new_x"].append(float(line.split(" = ")[1]))
                data_length["reflector_new_x"] += 1
                continue

            if "ReflectorCorrect : new_center_y = " in line:
                data["reflector_new_y"].append(float(line.split(" = ")[1]))
                data_length["reflector_new_y"] += 1
                continue

            try:
                flag_index = line.index(FUSION_LOCALIZATION_FLAG)
            except ValueError:
                pass
            else:
                begin_index = flag_index + len(FUSION_LOCALIZATION_FLAG)
                parts = line[begin_index:].split("=")
                key = parts[0].strip()
                value = float(parts[1].strip())
                data[key].append(value)
                data_length[key] += 1
                continue

            if "Cyclic() end" in line:
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
                    continue

    min_length = min(data_length.values())
    if any(length > min_length for length in data_length.values()):
        print(
            f"WARNING: ({log_dir}) The following sequence(s) will be truncated "
            "due to excess data length:"
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

    print(
        f"Generated {min_length:,} lines of data from "
        f'"{os.path.relpath(log_dir, SCRIPT_DIR)}".'
    )

    df_output = pd.DataFrame(
        zip(*data.values()),
        columns=tuple(data.keys()),
    )
    df_output["timestamp"] = \
        pd.to_datetime(df_output["timestamp"], unit="s") \
        + pd.Timedelta(8, "h")
    df_output.to_csv(OUTPUT_PATH)

    if (
        os.path.exists(NOTEBOOK_PATH)
        and not os.path.exists(os.path.join(LOG_DIR, NOTEBOOK_FILE_NAME))
    ):
        shutil.copy(NOTEBOOK_PATH, LOG_DIR)
    else:
        print("WARNING: Notebook file is not copied.")


if __name__ == "__main__":

    if "DATA_PATH" in locals() and DATA_PATH:
        convert_log(data_path=DATA_PATH)
    else:

        from multiprocessing import Pool

        n_logs = len(LOG_DIRS)
        if n_logs < 1:
            raise RuntimeError("No logs to convert!")

        process_count = min(n_logs, 8)
        with Pool(process_count) as pool:
            pool.map(convert_log, LOG_DIRS)

        print("Done.")
