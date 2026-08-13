cmake_minimum_required(VERSION 3.8)
project(robot_grasp_msgs)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  msg/Pose6D.msg
  msg/GraspCandidate.msg
  msg/GraspPlan.msg
  msg/PlacePlan.msg
  msg/PerceptionSummary.msg
  msg/PipelineStatus.msg
  srv/CaptureScene.srv
  srv/AnalyzeScene.srv
  srv/DetectTarget2D.srv
  srv/MatchItemLabel.srv
  srv/GetRobotState.srv
  srv/ExecuteNamedPose.srv
  srv/ExecuteGraspPlan.srv
  srv/ExecutePlacePlan.srv
  srv/MoveBaseRelative.srv
  srv/StopRobot.srv
  DEPENDENCIES builtin_interfaces geometry_msgs sensor_msgs
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()
