# Analogue Gauger

This is a Python3 script that uses OpenCV to read an analogue gauge needle's angle from a video and map it to a linear scale. It was written for a specific video of a temperature gauge with a red needle over a white background, but can be adapted to other gauges as well.

## Usage

Takes one input argument, the file to read. On POSIX systems, a camera device (e.g. /dev/video0) ought to work, but you'll have to press <kbd>Esc</kbd> to stop reading the video and start mapping the values at some point.

Note that in the current version, the most common parameters are all hard-coded in the main function. You have two options:

* Rewrite the values in the main function
* Import this script as a module in your own script and write your own call to `read_video()`.

### Video capture

My first version of this script was incredibly convoluted, to deal with moving reflections and noise in the image, and occasional camera movement. In the end, it was much easier to re-record the video while making sure that this didn't happen again. To that end I took the following steps to improve the video quality:

* I removed the plastic outer dust-cover from the measurement device, so it would not cause reflections.
* I faced the measurement device towards a white wall, so it would not show any reflections of people moving in the room.
* I aimed a small light directly at the wall, so it would provide a constant and diffuse light towards the gauge.
  * To eliminate any possibility that moving shadows on the wall would be reflected in the gauge.
  * Sufficient lighting significantly reduces the noise in the video.
  * Diffuse lighting meant that the camera itself did not cast such a hard shadow on the gauge.
* I put my smartphone (with some padding) in a vice, so it would not move at all during the recording.
* I moved the smartphone as close as possible.
  * Minimises the amount of crop needed.
  * Maximises the resolution of the gauge in the video, which improves the signal-to-noise ratio.
* I stabilised the measurement device itself, so it would not move during the recording.

## Processing steps

1. Open the video device
1. As long as there are frames available (or until <kbd>Esc</kbd> is pressed)
    1. Load frame
    1. Crop frame (hard coded crop at the moment)
    1. Convert cropped image to HSV
    1. Create saturation mask for coloured needle
    1. Apply Probabilistic Hough Lines Transform to mask
    1. Select the first line found
    1. Calculate the angle θ and store it with the frame index
    1. Show the cropped frame with the found line drawn on it just for visual feedback
1. Map all found (index,θ) combinations from the (frame index, angle) domain to the (seconds, supplied scale) domain.

If you run the file as a script, the main function then plots the results in a graph.

## Why?

I needed to plot heating and cooling curves for an oven that I was going to outfit with a control system. So I used my smartphone with a timelapse app to capture a video of the temperature gauge of the oven as it heated and cooled, at 0.1 FPS (one frame per 10 seconds). In the end I had thousands of frames, and little desire to read those all manually. I also suspected that I might have to do this again later, so I decided to dive into OpenCV a bit and see where that got me. It took me about eight hours to figure this script out. Most of that was spent on compiling OpenCV, finding out what processing options to use, in what order, and what parameters to use.

In the end I learned a lot, but the most important conclusion was that I needed a better video recording. My second video recording was of much better quality (see the pointers above) and I subsequently reduced all the preprocessing steps for the video to a simple saturation mask.

## Tentative Roadmap

I'd like to add a two features:

* Find the point where the oven is switched off and starts cooling
* Automatically find approximate polynomials for the heating and cooling curves