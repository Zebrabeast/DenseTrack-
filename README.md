This is for crowded tracking on UAVMOT. 

Using the densetrack can be divided into four steps:
   
   1.use PET to detect the position of targets

      In pet-main 
      test_nms.py  for one video sequence
      
      pets_dets_output.py can handle multiple video sequences,
      all you need is just putting on the right directory
   
   2.use the position information from MPM to generate the fused position
      in this part,you need the origin photos and position information from pet, and perform mpm_track_modify.py in directory MPM,
    you can get the fused position placed under their respective video sequences.
      

   3.use lavis/clip v-B16  extract the appearance features corresponding to the target
     With the fused postion and origin photos, we can use clip to get the appearance features of targets in each frame.
      
      the same as step 1, we can perform make_npy_test1.py for one video sequence,make_npy_test2.py for multiple video sequences. For more effective utilization，
     we save the coordinates and corresponding features in npy format.
 

   4.data assocation 
      Depending on the apperance features、fused_postion from clip, we use matching method to get the track list. The optical flow information 
     between consecutive mpm frames is used to predict the new coordinates based on the current coordinates.
      perform mpm_track_plus_multi.py in directory MPM,  you can get the track results placed under their respective video sequences.
