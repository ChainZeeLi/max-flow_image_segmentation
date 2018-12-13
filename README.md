# max-flow_image_segmentation
##EC504 Fall 2019 final project
##This projects uses Boykov algorithm to find the best graph cut for image segmentation
# Usage:
# PART1:
To train gaussian for a centerin type of image:
in ternminal, cd to the project folder, do "python parts_selector.py - i <image_path>"

After window with your image pops up, drag your mouse to select a region as foreground, press key F to finish

Then select a region as background, press key B to finish

After done, press key Q to quit
# PART2:
To use the image segmenation tool 
in ternminal, do "python segmentor.py - i <image_path>"
segmented foreground will be saved to project folder the
