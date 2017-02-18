#**Finding Lane Lines on the Road** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file. But feel free to use some other method and submit a pdf if you prefer.

---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//]: # (Image References)

[image1]: ./examples/grayscale.jpg "Grayscale"

---

### Reflection

###1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

My first version of pipeline consisted of 5 steps. First, I converted the images to gray-level image. Second, I using gaussian function to blur the image with kernel_size = 3. Third, I put the blur image to canny edge detection function, the canny edge detector will return an edge image, The fourth step is got the ROI region of this edge image by calling region of interest function.
The final step of my pipeline is set the ROI masked edges as input data to hough transform, and then display the detector results on the original images.

And I found in Optional Challenge section, hsv channel is works better than rgb channel. also some color selection methods can works here, but in night, the color selection methods is definaily not works.

In order to draw a single line on the left and right lanes, I modified the draw_lines() function by calculate the slope for each detected lines, and I do lots of test, found that the lane lines slope is normaly bigger than 0.5 or smaller than -0.5. so my function ignore the slope between 0.5 to -0.5(I think they are noise for our function). and for both left&right lines, the equation is y = m*x + b, here I set slope as m, and if m > 0.5, this line is right lines, m < -0.5 means this is left lines. so I also calculate the sum of slope and sum of bias for both left&right lines. After the loop function scan the each detected lines, I can have the average m and average bias for each left&right lines, then I just use y = m*x + b to extrapolate to the top and bottom of the lane, and draw both two lines on the same images.

If you'd like to include images to show how the pipeline works, here is how to include an image: 

![alt text][image1]
lists show as below is the results of my pipeline works, you can find them in the directory test_images:
[test_solidWhiteRight.jpg]
[test_solidYellowCurve2.jpg]
[test_whiteCarLaneSwitch.jpg]
[test_solidWhiteCurve.jpg]
[test_solidYellowLeft.jpg]
[test_solidYellowCurve.jpg]
video:
[white_1.mp4]
[yellow_1.mp4]
[white_2.mp4]
[yellow_2.mp4]
###2. Identify potential shortcomings with your current pipeline


One potential shortcoming would be what would happen when there may have many cars in the frount of us, the detection algorithm may confused and have many wrong results.

Another shortcoming could be if the weather is snow or rainy, the reults may also wrong.


###3. Suggest possible improvements to your pipeline

A possible improvement would be to using machine learning technology to fill the lines and classify the lines. for example we use image classification algorithm to detect the lane lines, and for each single detected lane lines, we then using regression technology to draw the best fitting lines for both left lines and right lines
