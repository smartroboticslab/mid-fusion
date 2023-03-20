import os

from _dataset import *
from systemsettings import *


#
# TUM RGB-D fr1/desk Settings
#
TUM_RGB_FR1_DESK = Dataset()
TUM_RGB_FR1_DESK.dataset_path = os.path.join(DATASETS_PATH, 'tum-rgbd/rgbd_dataset_freiburg1_desk/scene.raw')
TUM_RGB_FR1_DESK.ground_truth = os.path.join(DATASETS_PATH, 'tum-rgbd/rgbd_dataset_freiburg1_desk/groundtruth.txt')
TUM_RGB_FR1_DESK.camera = '525.0,525.0,319.5,239.5'#ROS default
TUM_RGB_FR1_DESK.quat = '0.6529,-0.5483,0.3248,-0.4095'
TUM_RGB_FR1_DESK.init_pose = '0.5,0.5,0.2'
TUM_RGB_FR1_DESK.pyramid_levels = '20, 20, 20' #'10, 5, 4'
TUM_RGB_FR1_DESK.pre_assoc_file_path = os.path.join(DATASETS_PATH, 'tum-rgbd/rgbd_dataset_freiburg1_desk/associations.txt')
TUM_RGB_FR1_DESK.descr = 'fr1_desk'
TUM_RGB_FR1_DESK.ate_associate_identity = False

#
# TUM RGB-D fr1/room Settings
#
TUM_RGB_FR1_ROOM = Dataset()
TUM_RGB_FR1_ROOM.dataset_path = os.path.join(DATASETS_PATH, 'tum-rgbd/rgbd_dataset_freiburg1_room/scene.raw')
TUM_RGB_FR1_ROOM.ground_truth = os.path.join(DATASETS_PATH, 'tum-rgbd/rgbd_dataset_freiburg1_room/groundtruth.txt')
TUM_RGB_FR1_ROOM.camera = '517.3,516.5,318.6,255.3'
TUM_RGB_FR1_ROOM.quat = '0.6529,-0.5483,0.3248,-0.4095'
TUM_RGB_FR1_ROOM.init_pose = '0.5,0.5,0.5'
TUM_RGB_FR1_ROOM.pre_assoc_file_path = os.path.join(DATASETS_PATH, 'tum-rgbd/rgbd_dataset_freiburg1_room/timings.assoc.txt')
TUM_RGB_FR1_ROOM.descr = 'fr1_room'
TUM_RGB_FR1_ROOM.ate_associate_identity = False

#
# TUM RGB-D fr1/plant Settings
#
TUM_RGB_FR1_PLANT = Dataset()
TUM_RGB_FR1_PLANT.dataset_path = os.path.join(DATASETS_PATH, 'tum-rgbd/rgbd_dataset_freiburg1_plant/scene.raw')
TUM_RGB_FR1_PLANT.ground_truth = os.path.join(DATASETS_PATH, 'tum-rgbd/rgbd_dataset_freiburg1_plant/groundtruth.txt')
TUM_RGB_FR1_PLANT.camera = '517.3,516.5,318.6,255.3'
TUM_RGB_FR1_PLANT.quat = '0.6529,-0.5483,0.3248,-0.4095'
TUM_RGB_FR1_PLANT.init_pose = '0.5,0.5,0.5'
TUM_RGB_FR1_PLANT.pre_assoc_file_path = os.path.join(DATASETS_PATH, 'rgbd_dataset_freiburg1_plant/timings.assoc.txt')
TUM_RGB_FR1_PLANT.descr = 'fr1_plant'
TUM_RGB_FR1_PLANT.ate_associate_identity = False

#
# TUM RGB-D fr1/floor Settings
#
TUM_RGB_FR1_FLOOR = Dataset()
TUM_RGB_FR1_FLOOR.dataset_path = os.path.join(DATASETS_PATH, 'tum-rgbd/rgbd_dataset_freiburg1_floor/scene.raw')
TUM_RGB_FR1_FLOOR.ground_truth = os.path.join(DATASETS_PATH, 'tum-rgbd/rgbd_dataset_freiburg1_floor/groundtruth.txt')
TUM_RGB_FR1_FLOOR.camera = '517.3,516.5,318.6,255.3'
TUM_RGB_FR1_FLOOR.quat = '0.6529,-0.5483,0.3248,-0.4095'
TUM_RGB_FR1_FLOOR.init_pose = '0.5,0.5,0.5'
TUM_RGB_FR1_FLOOR.pre_assoc_file_path = os.path.join(DATASETS_PATH, 'tum-rgbd/rgbd_dataset_freiburg1_floor/timings.assoc.txt')
TUM_RGB_FR1_FLOOR.descr = 'fr1_floor'
TUM_RGB_FR1_FLOOR.ate_associate_identity = False


#
# TUM RGB-D fr1/xyz Settings
#

TUM_RGB_FR1_XYZ = Dataset()
TUM_RGB_FR1_XYZ.dataset_path = os.path.join(DATASETS_PATH, 'tum-rgbd/rgbd_dataset_freiburg1_xyz/scene.raw') 
TUM_RGB_FR1_XYZ.ground_truth = os.path.join(DATASETS_PATH, 'tum-rgbd/rgbd_dataset_freiburg1_xyz/groundtruth.txt') 
# TUM_RGB_FR1_XYZ.camera = '591.1,590.1,331.0,234.0'#CALIBRATION OF THE INFRARED CAMERA (default) 
# TUM_RGB_FR1_XYZ.camera = '525.0,525.0,319.5,239.5'#ROS default
TUM_RGB_FR1_XYZ.camera = '517.3,516.5,318.6,255.3 '#CALIBRATION OF THE COLOR CAMERA: Ofuision test
TUM_RGB_FR1_XYZ.quat = '0.6132,0.5962,-0.3311,-0.3986'
TUM_RGB_FR1_XYZ.pyramid_levels = '10, 5, 4' #'10, 5, 4'
TUM_RGB_FR1_XYZ.rgb_image = os.path.join(DATASETS_PATH, 'tum-rgbd/rgbd_dataset_freiburg1_xyz/rgb/') 
TUM_RGB_FR1_XYZ.pre_assoc_file_path = os.path.join(DATASETS_PATH, 'tum-rgbd/rgbd_dataset_freiburg1_xyz/associations.txt') 
TUM_RGB_FR1_XYZ.init_pose = '0.5,0.5,0.5'
TUM_RGB_FR1_XYZ.ate_associate_identity = False
TUM_RGB_FR1_XYZ.descr = 'fr1_xyz'



# TUM RGB-D fr1/360 Settings
TUM_RGB_FR1_360 = Dataset()
TUM_RGB_FR1_360.dataset_path = os.path.join(DATASETS_PATH, 'tum-rgbd/rgbd_dataset_freiburg1_360/scene.raw')
TUM_RGB_FR1_360.ground_truth = os.path.join(DATASETS_PATH, 'tum-rgbd/rgbd_dataset_freiburg1_360/groundtruth.txt')
#TUM_RGB_FR1_360.ground_truth = os.path.join(DATASETS_PATH, 'rgbd_dataset_freiburg2_xyz/scene-ground.txt')
# TUM_RGB_FR1_360.camera_file = os.path.join(DATASETS_PATH, 'freiburg1.txt')
TUM_RGB_FR1_360.camera = '525.0,525.0,319.5,239.5'#ROS default
TUM_RGB_FR1_360.quat = '-0.8156,0.0346,-0.0049,0.5775'
TUM_RGB_FR1_360.init_pose = '0.2685,0.5,0.4'
TUM_RGB_FR1_360.pyramid_levels = '20, 20, 20' #'10, 5, 4'
TUM_RGB_FR1_360.rgb_image = os.path.join(DATASETS_PATH, 'tum-rgbd/rgbd_dataset_freiburg1_360/rgb/')
TUM_RGB_FR1_360.pre_assoc_file_path = os.path.join(DATASETS_PATH, 'tum-rgbd/rgbd_dataset_freiburg1_360/associations.txt')
TUM_RGB_FR1_360.ate_associate_identity = False
TUM_RGB_FR1_360.descr = 'fr1_360'


#
# TUM RGB-D fr2/xyz Settings
#
TUM_RGB_FR2_XYZ = Dataset()
TUM_RGB_FR2_XYZ.dataset_path = os.path.join(DATASETS_PATH, 'tum-rgbd/rgbd_dataset_freiburg2_xyz/scene.raw')
TUM_RGB_FR2_XYZ.ground_truth = os.path.join(DATASETS_PATH, 'tum-rgbd/rgbd_dataset_freiburg2_xyz/groundtruth.txt')
#TUM_RGB_FR2_XYZ.ground_truth = os.path.join(DATASETS_PATH, 'rgbd_dataset_freiburg2_xyz/scene-ground.txt')
TUM_RGB_FR2_XYZ.camera_file = os.path.join(DATASETS_PATH, 'freiburg2.txt')
TUM_RGB_FR2_XYZ.camera = '525.0,525.0,319.5,239.5'#ROS default
TUM_RGB_FR2_XYZ.quat = '-0.5721,0.6521,-0.3565,0.3469'
TUM_RGB_FR2_XYZ.init_pose = '0.5,0.5,0.5'
TUM_RGB_FR2_XYZ.rgb_image = os.path.join(DATASETS_PATH, 'tum-rgbd/rgbd_dataset_freiburg2_xyz/rgb/')
TUM_RGB_FR2_XYZ.pre_assoc_file_path = os.path.join(DATASETS_PATH, 'tum-rgbd/rgbd_dataset_freiburg2_xyz/timing.assoc.txt')
TUM_RGB_FR2_XYZ.ate_associate_identity = False
TUM_RGB_FR2_XYZ.descr = 'fr2_xyz'


#
# TUM RGB-D fr2/desk Settings
#
TUM_RGB_FR2_DESK = Dataset()
TUM_RGB_FR2_DESK.dataset_path = os.path.join(DATASETS_PATH, 'tum-rgbd/rgbd_dataset_freiburg2_desk/scene.raw') 
TUM_RGB_FR2_DESK.ground_truth = os.path.join(DATASETS_PATH, 'tum-rgbd/rgbd_dataset_freiburg2_desk/groundtruth.txt') 
TUM_RGB_FR2_DESK.camera_file = os.path.join(DATASETS_PATH, 'freiburg2.txt')
# TUM_RGB_FR2_DESK.camera = '525.0,525.0,319.5,239.5'#ROS default
TUM_RGB_FR2_DESK.camera = '520.9,521.0,325.1,249.7' #Ofusion test
TUM_RGB_FR2_DESK.quat = '0.6529,-0.5483,0.3248,-0.4095'
TUM_RGB_FR2_DESK.init_pose = '0.5,0.5,0.3'
TUM_RGB_FR2_DESK.pre_assoc_file_path = os.path.join(DATASETS_PATH, 'tum-rgbd/rgbd_dataset_freiburg2_desk/associations.txt') 
TUM_RGB_FR2_DESK.descr = 'fr2_desk'
TUM_RGB_FR2_DESK.pyramid_levels = '10, 5, 4' #'10, 5, 4'
TUM_RGB_FR2_DESK.ate_associate_identity = False


#
# TUM RGB-D fr2/coke Settings
#
TUM_RGB_FR2_COKE = Dataset()
TUM_RGB_FR2_COKE.dataset_path = os.path.join(DATASETS_PATH, 'tum-rgbd/rgbd_dataset_freiburg2_coke/scene.raw')
TUM_RGB_FR2_COKE.ground_truth = os.path.join(DATASETS_PATH, 'tum-rgbd/rgbd_dataset_freiburg2_coke/groundtruth.txt')
# TUM_RGB_FR2_COKE.camera_file = os.path.join(DATASETS_PATH, 'freiburg2.txt')
TUM_RGB_FR2_COKE.camera = '525.0,525.0,319.5,239.5'#ROS default 
TUM_RGB_FR2_COKE.quat = '0.6529,-0.5483,0.3248,-0.4095'
TUM_RGB_FR2_COKE.init_pose = '0.5,0.5,0.3'
TUM_RGB_FR2_COKE.pre_assoc_file_path = os.path.join(DATASETS_PATH, 'tum-rgbd/rgbd_dataset_freiburg2_coke/associations.txt')
TUM_RGB_FR2_COKE.descr = 'fr2_coke'
TUM_RGB_FR2_COKE.pyramid_levels = '20, 20, 20' #'10, 5, 4' 
TUM_RGB_FR2_COKE.ate_associate_identity = False



#
# TUM RGB-D fr3/desk Settings
#
TUM_RGB_FR3_DESK = Dataset()
TUM_RGB_FR3_DESK.dataset_path = os.path.join(DATASETS_PATH, 'tum-rgbd/rgbd_dataset_freiburg3_long_office_household/scene.raw')
TUM_RGB_FR3_DESK.ground_truth = os.path.join(DATASETS_PATH, 'tum-rgbd/rgbd_dataset_freiburg3_long_office_household/groundtruth.txt')
TUM_RGB_FR3_DESK.camera = '535.4,539.2,320.1,247.6'
TUM_RGB_FR3_DESK.init_pose = '0.5,0.5,0.3'
TUM_RGB_FR3_DESK.pre_assoc_file_path = os.path.join(DATASETS_PATH, 'tum-rgbd/rgbd_dataset_freiburg3_long_office_household/timings.assoc.txt')
TUM_RGB_FR3_DESK.descr = 'fr3_desk'
TUM_RGB_FR3_DESK.ate_associate_identity = False

#
# TUM RGB-D fr3/cabinet Settings
#
TUM_RGB_FR3_CABINET = Dataset()
TUM_RGB_FR3_CABINET.dataset_path = os.path.join(DATASETS_PATH, 'tum-rgbd/rgbd_dataset_freiburg3_cabinet/scene.raw')
TUM_RGB_FR3_CABINET.ground_truth = os.path.join(DATASETS_PATH, 'tum-rgbd/rgbd_dataset_freiburg3_cabinet/groundtruth.txt')
TUM_RGB_FR3_CABINET.camera = '535.4,539.2,320.1,247.6'
TUM_RGB_FR3_CABINET.init_pose = '0.5,0.5,0'
TUM_RGB_FR3_CABINET.pre_assoc_file_path = os.path.join(DATASETS_PATH, 'tum-rgbd/rgbd_dataset_freiburg3_cabinet/timings.assoc.txt')
TUM_RGB_FR3_CABINET.descr = 'fr3_cabinet'
TUM_RGB_FR3_CABINET.ate_associate_identity = False

#
# TUM RGB-D fr3/cabinet Settings
#
TUM_RGB_FR3_LARGE_CABINET = Dataset()
TUM_RGB_FR3_LARGE_CABINET.dataset_path = os.path.join(DATASETS_PATH, 'tum-rgbd/rgbd_dataset_freiburg3_large_cabinet/scene.raw')
TUM_RGB_FR3_LARGE_CABINET.ground_truth = os.path.join(DATASETS_PATH, 'tum-rgbd/rgbd_dataset_freiburg3_large_cabinet/groundtruth.txt')
TUM_RGB_FR3_LARGE_CABINET.camera = '535.4,539.2,320.1,247.6'
TUM_RGB_FR3_LARGE_CABINET.init_pose = '0.5,0.5,0'
TUM_RGB_FR3_LARGE_CABINET.pre_assoc_file_path = os.path.join(DATASETS_PATH, 'rgbd_dataset_freiburg3_large_cabinet/timings.assoc.txt')
TUM_RGB_FR3_LARGE_CABINET.descr = 'fr3_large_cabinet'
TUM_RGB_FR3_LARGE_CABINET.ate_associate_identity = False


#
# ICL-NUIM Living Room 0
#
ICL_NUIM_LIV_0 = Dataset()
ICL_NUIM_LIV_0.dataset_path = os.path.join(DATASETS_PATH, 'ICL-NUIM/living_room_traj0_frei_png/scene.raw')
ICL_NUIM_LIV_0.ground_truth = os.path.join(DATASETS_PATH, 'ICL-NUIM/living_room_traj0_frei_png/livingRoom0.gt.freiburg')
ICL_NUIM_LIV_0.camera_file = os.path.join(DATASETS_PATH, 'ICL-NUIM/living_room_traj0_frei_png/camera.txt')
ICL_NUIM_LIV_0.camera = '481.2,-480,320,240'
ICL_NUIM_LIV_0.init_pose = '0.34,0.5,0.24'
ICL_NUIM_LIV_0.ate_associate_identity = True
ICL_NUIM_LIV_0.descr = 'liv_traj_0'


#
# ICL-NUIM Living Room 1-short for debug
#
ICL_NUIM_LIV_1_SHORT = Dataset()
ICL_NUIM_LIV_1_SHORT.dataset_path = os.path.join(DATASETS_PATH, 'ICL-NUIM/living_room_traj1_loop_short/scene.raw') 
ICL_NUIM_LIV_1_SHORT.ground_truth = os.path.join(DATASETS_PATH, 'ICL-NUIM/living_room_traj1_loop_short/livingRoom1_short.gt.freiburg.txt') 
# ICL_NUIM_LIV_1_SHORT.camera_file = os.path.join(DATASETS_PATH, 'living_room_traj1_frei_png/camera.txt')
ICL_NUIM_LIV_1_SHORT.camera = '481.2,-480,320,240'
ICL_NUIM_LIV_1_SHORT.init_pose = '0.485,0.5,0.55'
ICL_NUIM_LIV_1_SHORT.pyramid_levels = '20, 20, 20' #'10, 5, 4' 
ICL_NUIM_LIV_1_SHORT.ate_associate_identity = True
ICL_NUIM_LIV_1_SHORT.descr = 'living_room_traj1_loop_short'

#
# ICL-NUIM Living Room 1
#
ICL_NUIM_LIV_1 = Dataset()
ICL_NUIM_LIV_1.dataset_path = os.path.join(DATASETS_PATH, 'ICL-NUIM/living_room_traj1_loop/scene.raw') 
ICL_NUIM_LIV_1.ground_truth = os.path.join(DATASETS_PATH, 'ICL-NUIM/living_room_traj1_loop/livingRoom1.gt.freiburg.txt') 
# ICL_NUIM_LIV_1.camera_file = os.path.join(DATASETS_PATH, 'living_room_traj1_frei_png/camera.txt')
ICL_NUIM_LIV_1.camera = '481.2,-480,320,240'
ICL_NUIM_LIV_1.init_pose = '0.485,0.5,0.55'
ICL_NUIM_LIV_1.pyramid_levels = '20, 20, 20' #'10, 5, 4' 
ICL_NUIM_LIV_1.ate_associate_identity = True
ICL_NUIM_LIV_1.descr = 'liv_traj_1'


#
# ICL-NUIM Living Room 2
#
ICL_NUIM_LIV_2 = Dataset()
ICL_NUIM_LIV_2.dataset_path = os.path.join(DATASETS_PATH, 'ICL-NUIM/living_room_traj2_frei_png/scene.raw')
ICL_NUIM_LIV_2.ground_truth = os.path.join(DATASETS_PATH, 'ICL-NUIM/living_room_traj2_frei_png/livingRoom2.gt.freiburg')
ICL_NUIM_LIV_2.camera_file = os.path.join(DATASETS_PATH, 'ICL-NUIM/living_room_traj2_frei_png/camera.txt')
ICL_NUIM_LIV_2.camera = '481.2,-480,320,240'
ICL_NUIM_LIV_2.init_pose = '0.34,0.5,0.24'
ICL_NUIM_LIV_2.ate_associate_identity = True
ICL_NUIM_LIV_2.descr = 'liv_traj_2'

#
# ICL-NUIM Living Room 3
#
ICL_NUIM_LIV_3 = Dataset()
ICL_NUIM_LIV_3.dataset_path = os.path.join(DATASETS_PATH, 'ICL-NUIM/living_room_traj3_frei_png/scene.raw')
ICL_NUIM_LIV_3.ground_truth = os.path.join(DATASETS_PATH, 'ICL-NUIM/living_room_traj3_frei_png/livingRoom3.gt.freiburg')
ICL_NUIM_LIV_3.camera_file = os.path.join(DATASETS_PATH, 'ICL-NUIM/living_room_traj3_frei_png/camera.txt')
ICL_NUIM_LIV_3.camera = '481.2,-480,320,240'
ICL_NUIM_LIV_3.init_pose = '0.2685,0.5,0.4'
ICL_NUIM_LIV_3.ate_associate_identity = True
ICL_NUIM_LIV_3.descr = 'liv_traj_3'

#
# ICL-NUIM Office 0
#
ICL_NUIM_OFF_0 = Dataset()
ICL_NUIM_OFF_0.dataset_path = os.path.join(DATASETS_PATH, 'ICL-NUIM/office_room_traj0_loop/scene.raw')
ICL_NUIM_OFF_0.ground_truth = os.path.join(DATASETS_PATH, 'ICL-NUIM/office_room_traj0_loop/traj0.gt.freiburg')
ICL_NUIM_OFF_0.camera_file = os.path.join(DATASETS_PATH, 'ICL-NUIM/living_room_traj2_loop/camera.txt')
ICL_NUIM_OFF_0.camera = '481.2,480,320,240'
ICL_NUIM_OFF_0.init_pose = '0.5,0.5,0.5'


#
# ICL-NUIM Office 2
#
ICL_NUIM_OFF_2 = Dataset()
ICL_NUIM_OFF_2.dataset_path = os.path.join(DATASETS_PATH, 'ICL-NUIM/office_room_traj2_loop/scene.raw')
ICL_NUIM_OFF_2.ground_truth = os.path.join(DATASETS_PATH, 'ICL-NUIM/office_room_traj2_loop/traj2.gt.freiburg.txt')
ICL_NUIM_OFF_2.camera_file = os.path.join(DATASETS_PATH, 'ICL-NUIM/living_room_traj2_loop/camera.txt')
ICL_NUIM_OFF_2.camera = '481.2,480,320,240'
ICL_NUIM_OFF_2.init_pose = '0.5,0.5,0.5'

#################### Low Dynamic #################
#
# TUM RGB-D fr3-sitting/static Settings
#
TUM_RGB_FR3_SIT_static = Dataset()
TUM_RGB_FR3_SIT_static.dataset_path = os.path.join(DATASETS_PATH, 'tum-rgbd/rgbd_dataset_freiburg3_sitting_static/scene.raw')
TUM_RGB_FR3_SIT_static.ground_truth = os.path.join(DATASETS_PATH, 'tum-rgbd/rgbd_dataset_freiburg3_sitting_static/groundtruth.txt')
TUM_RGB_FR3_SIT_static.camera = '525.0,525.0,319.5,239.5'	
TUM_RGB_FR3_SIT_static.init_pose = '0.5,0.5,0'
TUM_RGB_FR3_SIT_static.pre_assoc_file_path = os.path.join(DATASETS_PATH, 'tum-rgbd/rgbd_dataset_freiburg3_sitting_static/associations.txt')
TUM_RGB_FR3_SIT_static.descr = 'fr3_sit_static'
TUM_RGB_FR3_SIT_static.ate_associate_identity = False
TUM_RGB_FR3_SIT_static.maskrcnn_folder = os.path.join(DATASETS_PATH, 'tum-rgbd/rgbd_dataset_freiburg3_sitting_static/mask_RCNN')

#
# TUM RGB-D fr3-sitting/xyz Settings
#
TUM_RGB_FR3_SIT_xyz = Dataset()
TUM_RGB_FR3_SIT_xyz.dataset_path = os.path.join(DATASETS_PATH, 'tum-rgbd/rgbd_dataset_freiburg3_sitting_xyz/scene.raw')
TUM_RGB_FR3_SIT_xyz.ground_truth = os.path.join(DATASETS_PATH, 'tum-rgbd/rgbd_dataset_freiburg3_sitting_xyz/groundtruth.txt')
TUM_RGB_FR3_SIT_xyz.camera = '525.0,525.0,319.5,239.5'	
TUM_RGB_FR3_SIT_xyz.init_pose = '0.5,0.5,0'
TUM_RGB_FR3_SIT_xyz.pre_assoc_file_path = os.path.join(DATASETS_PATH, 'tum-rgbd/rgbd_dataset_freiburg3_sitting_xyz/associations.txt')
TUM_RGB_FR3_SIT_xyz.descr = 'fr3_sit_xyz'
TUM_RGB_FR3_SIT_xyz.ate_associate_identity = False
TUM_RGB_FR3_SIT_xyz.maskrcnn_folder = os.path.join(DATASETS_PATH, 'tum-rgbd/rgbd_dataset_freiburg3_sitting_xyz/mask_RCNN')

#
# TUM RGB-D fr3-sitting/half-sphere Settings
#
TUM_RGB_FR3_SIT_halfsphere = Dataset()
TUM_RGB_FR3_SIT_halfsphere.dataset_path = os.path.join(DATASETS_PATH, 'tum-rgbd/rgbd_dataset_freiburg3_sitting_halfsphere/scene.raw')
TUM_RGB_FR3_SIT_halfsphere.ground_truth = os.path.join(DATASETS_PATH, 'tum-rgbd/rgbd_dataset_freiburg3_sitting_halfsphere/groundtruth.txt')
TUM_RGB_FR3_SIT_halfsphere.camera = '525.0,525.0,319.5,239.5'	
TUM_RGB_FR3_SIT_halfsphere.init_pose = '0.5,0.5,0'
TUM_RGB_FR3_SIT_halfsphere.pre_assoc_file_path = os.path.join(DATASETS_PATH, 'tum-rgbd/rgbd_dataset_freiburg3_sitting_halfsphere/associations.txt')
TUM_RGB_FR3_SIT_halfsphere.descr = 'fr3_sit_hs'
TUM_RGB_FR3_SIT_halfsphere.ate_associate_identity = False
TUM_RGB_FR3_SIT_halfsphere.maskrcnn_folder = os.path.join(DATASETS_PATH, 'tum-rgbd/rgbd_dataset_freiburg3_sitting_halfsphere/mask_RCNN')

#################### High Dynamic #################

#
# TUM RGB-D fr3-walking/static Settings
#
TUM_RGB_FR3_WALK_static = Dataset()
TUM_RGB_FR3_WALK_static.dataset_path = os.path.join(DATASETS_PATH, 'tum-rgbd/rgbd_dataset_freiburg3_walking_static/scene.raw')
TUM_RGB_FR3_WALK_static.ground_truth = os.path.join(DATASETS_PATH, 'tum-rgbd/rgbd_dataset_freiburg3_walking_static/groundtruth.txt')
TUM_RGB_FR3_WALK_static.camera = '525.0,525.0,319.5,239.5'	
TUM_RGB_FR3_WALK_static.init_pose = '0.5,0.5,0'
TUM_RGB_FR3_WALK_static.pre_assoc_file_path = os.path.join(DATASETS_PATH, 'tum-rgbd/rgbd_dataset_freiburg3_walking_static/associations.txt')
TUM_RGB_FR3_WALK_static.descr = 'fr3_walk_static'
TUM_RGB_FR3_WALK_static.ate_associate_identity = False
TUM_RGB_FR3_WALK_static.maskrcnn_folder = os.path.join(DATASETS_PATH, 'tum-rgbd/rgbd_dataset_freiburg3_walking_static/mask_RCNN')



#
# TUM RGB-D fr3-walking/xyz Settings
#
TUM_RGB_FR3_WALK_xyz = Dataset()
TUM_RGB_FR3_WALK_xyz.dataset_path = os.path.join(DATASETS_PATH, 'tum-rgbd/rgbd_dataset_freiburg3_walking_xyz/scene.raw')
TUM_RGB_FR3_WALK_xyz.ground_truth = os.path.join(DATASETS_PATH, 'tum-rgbd/rgbd_dataset_freiburg3_walking_xyz/groundtruth.txt')
TUM_RGB_FR3_WALK_xyz.camera = '525.0,525.0,319.5,239.5'	
TUM_RGB_FR3_WALK_xyz.init_pose = '0.5,0.5,0'
TUM_RGB_FR3_WALK_xyz.pre_assoc_file_path = os.path.join(DATASETS_PATH, 'tum-rgbd/rgbd_dataset_freiburg3_walking_xyz/associations.txt')
TUM_RGB_FR3_WALK_xyz.descr = 'fr3_walk_xyz'
TUM_RGB_FR3_WALK_xyz.ate_associate_identity = False
TUM_RGB_FR3_WALK_xyz.maskrcnn_folder = os.path.join(DATASETS_PATH, 'tum-rgbd/rgbd_dataset_freiburg3_walking_xyz/mask_RCNN')

#
# TUM RGB-D fr3-walking/half-sphere Settings
#
TUM_RGB_FR3_WALK_halfsphere = Dataset()
TUM_RGB_FR3_WALK_halfsphere.dataset_path = os.path.join(DATASETS_PATH, 'tum-rgbd/rgbd_dataset_freiburg3_walking_halfsphere/scene.raw')
TUM_RGB_FR3_WALK_halfsphere.ground_truth = os.path.join(DATASETS_PATH, 'tum-rgbd/rgbd_dataset_freiburg3_walking_halfsphere/groundtruth.txt')
TUM_RGB_FR3_WALK_halfsphere.camera = '525.0,525.0,319.5,239.5'	
TUM_RGB_FR3_WALK_halfsphere.init_pose = '0.5,0.5,0'
TUM_RGB_FR3_WALK_halfsphere.pre_assoc_file_path = os.path.join(DATASETS_PATH, 'tum-rgbd/rgbd_dataset_freiburg3_walking_halfsphere/associations.txt')
TUM_RGB_FR3_WALK_halfsphere.descr = 'fr3_walk_hs'
TUM_RGB_FR3_WALK_halfsphere.ate_associate_identity = False
TUM_RGB_FR3_WALK_halfsphere.maskrcnn_folder = os.path.join(DATASETS_PATH, 'tum-rgbd/rgbd_dataset_freiburg3_walking_halfsphere/mask_RCNN/')
