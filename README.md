[![Cloud Programming World Cup Winners](images/cpwc-18.png)](http://cpwc.forum8.co.jp/)



<p align="center">
  <img src="images/uet-logo.png" width="100" title="Tech.Divas"
   href=http://www.uettaxila.edu.pk/>
</p>

# Background

Recently, My team **Tech.Divas** has won a Grand prize Cloud Programming World Cup 2018. Where we used openCV and Deep Learing for various object detection in the UC-Win/Road Virtual Reality based Driving Simulator.

In the project, We have used [MobileNet-SSD](https://github.com/chuanqi305/MobileNet-SSD) for detecting Car, Person, Bikes, Bus using python. Here, we call this model as **standard deep detector**.

During the project work, we have faced few problems while applying the standard deep detector to a UC-WIN/Road simulator scenes:

 - Since the standard deep detector is trained using real world images, we have observed some missing detections some scenes.
 - In addition to the standard deep detector object categories, we also need to detect extra categories such as  street light, and trash bin.
 
We have overcome such problems by using Transfer Learning. I am going to share this code with the opensource python community. **Here, I will write every  step needed for fine tuning the standard deep detector and how to add additional object categories** . Such steps could be applied in any other similar situations.

Before going into details, I would like to share a result before hand as shown bellow:

![](images/vlcsnap-2018-11-25-18h06m41s966.png  "Comparison Between Standard and Our Fine Tune MobileNet-SSD models")

As we can see in the left side of the above figure, the standard deep dector failed in detecting any object in the scene. **However, my fine tuned model detects**, results are shown on right side of the above figure, **not only detect car and bus but an additional object name streetlight**.

# How to fine tune and how to add additional categories 


