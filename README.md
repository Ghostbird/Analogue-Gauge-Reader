# Analogue Gauger

This is a Python3 script that uses OpenCV to read an analogue gauge needle's angle from a video and map it to a linear scale. It was written for a specific video of a temperature gauge with a red needle over a white background, but can be adapted to other gauges as well.

## Usage

Takes one input argument, the file to read. On POSIX systems, a camera device (e.g. /dev/video0) ought to work, but you'll have to press <kbd>Esc</kbd> to stop reading the video and start mapping the values at some point.

# Processing steps

1. Load frame
1. Crop frame (hard coded crop at the moment)
1. Calculate foeground mask using MOG2
1. Convert cropped image to HSV
1. Create colour mask for reddish needle
1. Combine colour mask with foreground mask
1. Apply small (2×2) kerneled morphological closing to combined mask
1. Apply Probabilistic Hough Lines Transform to final mask
1. Select the first line found that is not obviously wrong
1. Calculate the angle θ and store it with the frame index
1. Show the cropped frame with the found line drawn on it just for visual feedback

Finally all found lines for all frames are mapped from the θ domain to the supplied domain. The default target domain is 0-1200°C. The result is plotted in a graph.

## Why?

I needed to plot heating and cooling curves for an oven that I was going to outfit with a control system. So I used my smartphone with a timelapse app to capture a video of the temperature gauge of the oven as it heated and cooled, at 0.1 FPS (one frame per 10 seconds). In the end I had thousands of frames, and little desire to read those all manually. I also suspected that I might have to do this again later, so I decided to dive into OpenCV a bit and see where that got me. It took me about eight hours to figure this script out. Most of that was spent on compiling OpenCV, finding out what processing options to use, in what order, and what parameters to use.

In the end I'm fairly happy with the result, but I plan on doing another video of the gauge because the current one had issues with reflections in the gauge window, lighting changes and camera movement due to vibrations.

## Tentative Roadmap

I'd like to add a two features:

* Find the point where the oven is switched off and starts cooling
* Automatically find approximate polynomials for the heating and cooling curves