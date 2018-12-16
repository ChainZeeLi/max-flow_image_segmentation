# max-flow_image_segmentation
EC504 Fall 2019 final project
This projects uses max-flow and min-cut to find the best graph cut for image segmentation. 
### Usage:
* PART1:
To train gaussian for a centerin type of image:
in ternminal, cd to the project folder, do "<addr>python parts_selector.py - i <image_path><addr>"

After window with your image pops up, drag your mouse to select a region as foreground, press key F to finish

Then select a region as background, press key B to finish
When done, press key Q to quit
* PART2:
To use the image segmenation tool 
in ternminal, do "<addr>python segmentor.py - i <image_path><addr>"
segmented foreground will be saved to this project folder
