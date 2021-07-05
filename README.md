# Advanced Lane Finding
________________________________________

The goals / steps of this project are the following:
* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images & use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.
________________________________________
### Reflection

##### Camera Calibration: Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image 
* To get the camera matrix and distortion coefficients, I used the chessboard images provided for this project. I used cv2 functions to find the corners of the chessboards and the cv2.calibratecamera function to find the camera matrix and distortion coefficient.

###### _Calibration image before and after distortion correction:_

![image](https://user-images.githubusercontent.com/74683142/122590564-b98d0800-d02f-11eb-961d-380c6280808c.png)  ![image](https://user-images.githubusercontent.com/74683142/122590716-f2c57800-d02f-11eb-8e7f-20c58ff44ab6.png)

###### _Test image before and after distortion correction:_

![image](https://user-images.githubusercontent.com/74683142/122590801-0a9cfc00-d030-11eb-9750-c3cf97787931.png)  ![image](https://user-images.githubusercontent.com/74683142/122590865-21dbe980-d030-11eb-914a-ceb20568b625.png)


##### Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image. Provide an example of a binary image result.
* After experimenting with different gradient and color transform combinations, I ended up using the x and y Sobel gradients, and color thresholding for the S and V channels. The x-direction Sobel gradient and S channel of the HSL color threshold seemed to have the biggest impact with filtering out the lanes. I was able to tune these parameters enough to find the lanes accurately in most frames, but the lanes were shaky when there were lighting changes or changes in the pavement color during the video.
 
###### _Test image and thresholded binary image:_

![image](https://user-images.githubusercontent.com/74683142/122593384-95332a80-d033-11eb-993a-0f6d35324918.png) ![image](https://user-images.githubusercontent.com/74683142/122593399-9bc1a200-d033-11eb-87f6-257320048154.png)
 
The sobel gradients were in the abs_sobel_thresh function, and color thresholding was in the hsl_select function. I combined both thresholds to create a binary image in the function called process_image, and here is a snippet of the code showing which values I used for the thresholds:

```
# Pipeline for applying color transform and gradients to image
def process_image(image):

    preprocessimage = np.zeros_like(image[:,:,0])

    gradx = abs_sobel_thresh(image, orient='x', thresh_min = 40, thresh_max = 100)
    grady = abs_sobel_thresh(image, orient='y', thresh_min = 25, thresh_max = 100)
    hls_binary = hls_select(image, sthresh=(130, 255), vthresh=(75, 255))

    preprocessimage[(gradx == 1) & (grady == 1) | (hls_binary == 1)] = 255

    return preprocessimage
```

##### Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.
 
###### _Binary image before and after perspective transform:_

![image](https://user-images.githubusercontent.com/74683142/122594033-7da87180-d034-11eb-8837-9b1ed2269f2d.png) ![image](https://user-images.githubusercontent.com/74683142/122594105-957ff580-d034-11eb-86f6-0866586497bc.png)


* Using the width and height of the image, I visually estimated where the source and destination points appeared to be. I created a perspective_transform function to warp the image and get a “birds eye view,” and checked to make sure the lines appeared parallel with each other. Here is the function, as well as the source and destination points I used for the perspective transform in the image pipeline:
```
# Perspective transform to get a "birds eye view" of lane
def perspective_transform(image, src, dst):

    img_size = (image.shape[1], image.shape[0])
    M = cv2.getPerspectiveTransform(src, dst)
    warp = cv2.warpPerspective(image, M, img_size, flags=cv2.INTER_NEAREST)
    return warp
    
# Source and destination points in the image pipeline:
src = np.float32([[590, 450], [690, 450], [1120, 710], [160, 710]])
dst = np.float32([[320, 0], [960, 0], [960, 720], [320, 720]])
warp = perspective_transform(processed, src, dst)
```

##### Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial.
* I found starting points for the left and right lanes from a histogram, using the peaks on each side of the midpoint. From there, I used the sliding window approach to follow the curve for the left and right lanes. If enough points were found in a window, the next window shifted to the mean x-coordinate of these points to fit them to a second order polynomial. Next, I searched for points around the polynomial to keep following the curve. The functions included find_lane_pixels, find_poly, and search_around_poly.

![image](https://user-images.githubusercontent.com/74683142/122594285-d7a93700-d034-11eb-948e-4df690e2c46e.png) ![image](https://user-images.githubusercontent.com/74683142/122594342-e5f75300-d034-11eb-9592-3c487250c1b4.png) ![image](https://user-images.githubusercontent.com/74683142/122594364-ebed3400-d034-11eb-8ca9-0adf92b4e6c4.png)



##### Describe how (and identify where in your code) you calculated the radius of curvature of the lane and position of the vehicle with respect to center.
- The equation for the radius of curvature was given in the lecture videos, and I used the meters to pixel conversions to modify the measure_curvature function. For the vehicle position, I used the horizontal midpoint of the image as the center of the vehicle, based on the assumption that the camera was centered. For the center of the lane position, I took the average of the left and right fit x values. The position of the vehicle relative to the center of the lane was displayed later in the draw_lanes function.
```
    # Fit a second order polynomial to each using `np.polyfit`
    left_fit = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)
    
    # Define y-value where we want radius of curvature
    # We'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = ym_per_pix*np.max(ploty)
    
    # Calculation of R_curve (radius of curvature)
    left_curverad = ((1 + (2*left_fit[0]*y_eval + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
    right_curverad = ((1 + (2*right_fit[0]*y_eval + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])
```
Here is the function for calculating the vehicle position relative to the lane:
```
def vehicle_position(image, left_pos, right_pos):

    # Define conversions in x from pixels space to meters
    xm_per_pix = 3.7/700 # meters per pixel in x dimension

    # Calculate position based on midpoint - center of lanes distance
    vehicle_center = (image.shape[1])/2
    lane_center = (left_pos + right_pos)/2
    position = vehicle_center - lane_center

    # Get value in meters
    vehicle_position = position*xm_per_pix

    return vehicle_position
```

##### Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.
* Here is an example of the results from the search_around_poly function, and an example of the lane warped back onto the original image. The display includes the average radius of curvature from the left and right lanes, as well as the vehicle position relative to the center of the lane.

![image](https://user-images.githubusercontent.com/74683142/122594407-f7d8f600-d034-11eb-9573-469ec4d5c781.png) ![image](https://user-images.githubusercontent.com/74683142/122594429-fdced700-d034-11eb-9dee-d7a8d402226e.png)


##### Provide a link to your final video output.
Both videos are included in the Jupyter project notebook, and here are links to the Youtube videos:

* Output from project video: https://youtu.be/UPWBiz8IjyQ
* Output from my own recorded video: https://youtu.be/caSt__-QoV4

### Discussion
##### Briefly discuss any problems/issues you faced in your implementation of this project. Where will your pipeline likely fail? What could you do to make it more robust?
* The lane detection pipeline was jumpy during significant shadows or lighting changes, or additional lines near the lane markings. It could fail when there are multiple lines on the road, changes in pavement color, or faded lane markings. The current lane detection pipeline was accurate for most of the project video, but the lane detection was inconsistent and unstable in the challenge videos.

* To make my pipeline more robust, I could have averaged the polynomial coefficients from an N number of previous frames to smooth lane detection and filter out some noise. Other improvements could include a weighted average, filtering out lines based on known distance between left and right lanes, or filtering out white and yellow colors from the image as potential lane markings.
