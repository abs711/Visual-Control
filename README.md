# Visual Control
Using egocentric vision to improve the imu-based prediction of future knee and ankle joint angles, in complex out-of-the-lab environments.

Paper: https://ieeexplore.ieee.org/abstract/document/9729197

![Optical_Flow_boe_font_updated2](https://user-images.githubusercontent.com/42185229/177664658-80144c7c-4224-4de4-aeac-fae5744160ac.png)

## Summary
Here we fuse motion capture data with egocentric videos to improve the joint angle prediction performance in complex uncontrolled environment like public classrooms, atrium and stairwells. The optical flow features are generated from the raw images, by PWC-net trained on the synthetic MPI-Sintel dataset, and processed by a LSTM before being fused with the joint kinematics stream.

In the following video, we can see that the information about the future movements of the subject is available in their visual field, both in terms of what lies ahead of them e.g. stairs or chairs, as well as how they move their head and eyes for path-planning. Thus, vision acts as a "window into the future".
https://youtu.be/axBb37dWbko

## Egocentric vision improves the prediction of lower limb joint angles

The following videos and the corresponding figures show example maneuvers and the improvement achieved over just kinematics inputs (red line), by fusing kinematics and vision inputs (green line).

https://user-images.githubusercontent.com/42185229/235016479-a5fd240d-c245-4f59-8155-1cf356616a77.mp4

![Picture7](https://user-images.githubusercontent.com/42185229/235016499-e5d20136-b3bd-49ab-8f55-45dfff6e6196.png)



https://user-images.githubusercontent.com/42185229/235016508-85f2a725-7a28-45b1-b9ec-be2279edae0d.mp4

![Picture5](https://user-images.githubusercontent.com/42185229/235016525-484c83f7-026a-4596-a24c-eee5ce55b29a.png)



## The benefits of egocentric vision can be amplified with more data

In the figure below, we compared performance improvements due to vision with increase in the amount of data per subject (left) and increase in the number of subjects (right). We see that inclusion of vision shows better improvement than no vision condition, for both the cases. We also see that rate of improvement for vision reduces slowly compared to the no vision condition. Indicating that with more data better performance could be achieved.

![datasize_combined](https://user-images.githubusercontent.com/42185229/235016538-4fb904cd-0d2c-4a7e-81bb-ea9b76ef27d7.png)

NOTE: The dataset used in the paper can be accessed on the following repository: https://github.com/abs711/The-way-of-the-future . The detailed description of the dataset is available in the following publication: https://doi.org/10.1038/s41597-023-01932-7

Run 'torchVision/MainPain/main.py' to start training a model. The models used in the paper are defined 'UtilX/Vision4Prosthetics_modules.py'. 
