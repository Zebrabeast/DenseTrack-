
# This is for crowd tracking on UAV --DenseTrack 

DenseTrack proposes a **_multiple association paradigm_**, can take full advantage of motion and appearance features.

<!-- <p align="center"><img src="figs/pipeline.png" width="600"/></p> -->

<!-- > [**TOPIC: A Parallel Association Paradigm for Multi-Object Tracking under Complex Motions and Diverse Scenes**](https://ieeexplore.ieee.org/document/10851814)
>
> Xiaoyan Cao, Yiyao Zheng, Yao Yao, Huapeng Qin, Xiaoyu Cao and Shihui Guo
>
> _[IEEE TIP](https://ieeexplore.ieee.org/document/10851814)_ -->

   
## Using the densetrack can be divided into four steps:
   
   1.use PET to detect the position of targets
       
           # In pet-main--all you need is just putting on the right directory
           python test_nms.py  # one video sequence
         
   or  
         
           python pets_dets_output.py # multiple video sequences
   
   
   2.use the position information from MPM to generate the fused position


  in this part,you need the origin photos and position information from pet, and perform mpm_track_modify.py in directory MPM,
  you can get the fused position placed under their respective video sequences.
         
           cd MPM
           python mpm_track_modify.py 

   3.use lavis/clip v-B16  extract the appearance features corresponding to the target
   
       """With the fused postion and origin photos, we can use clip to get the appearance features of targets in each frame."""
           # the same as step 1
           python  make_npy_test1.py  # one video sequence
   
   or
   
           python make_npy_test2.py   # multiple video sequences. 
   
   For more effective utilization，we save the coordinates and corresponding features in npy format.
 

   4.data assocation 

 
  Depending on the apperance features、fused_postion from clip, we use matching method to get the track list. The optical flow information 
 between consecutive mpm frames is used to predict the new coordinates based on the current coordinates.
         
           cd MPM
           python mpm_track_plus_multi.py  # multiple video sequences. 

   you can get the track results placed under their respective video sequences.




   To be honest, in those steps we can realize that the MPM be used twice. Actually we can just use once through using fused_position to 
extra features, then  directly use it for matching.  In consideration of extensive reuse 、unmature matching method and the time spent on extracting 
features,we divide it in four steps.
