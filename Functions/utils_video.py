#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 10:30:36 2023

@author: keving
"""

import imageio
import moviepy.editor as ed


def gen_gif(files, outfile, frame_duration=1, format_input='png', npause_last=5):
    """
    generates a gif from image files 

    Parameters
    ----------
    files : list of strs
        list if full paths of image files.
    outfile : str
        full path of output file.
    frame_duration : float, optional
        duration of each image frame for the gif in seconds. The default is 1.
    format_input : str, optional
        file format of input image files. The default is 'png'.
    npause_last : int, optional
        number of copies of the last image to create a break in the gif . The default is 5.

    Returns
    -------
    None.

    """
    frame_duration_ms = frame_duration*1000
    
    file_formats=['.png','.jpg','.eps']
    if files[0][-4::] not in file_formats:
        files = [f + '.' + format_input for f in files]
    
    if npause_last>1: 
        files_pro = files.copy()
        for i in range(npause_last): 
            files_pro.append(files[-1])
    else: 
        files_pro = files.copy()
        
    if outfile[-4::] != '.gif': 
         outfile =  outfile + '.gif'

    with imageio.get_writer(outfile, mode='I', duration=frame_duration_ms) as writer:
        for filename in files_pro:
            #for i in range(3):
            #print(f'file: {filename}') #debug
            image = imageio.imread(filename)
            writer.append_data(image)
    print(f'Gif created: {outfile}')
    return 
    
    
def gif2mp4(gif, video_file=None): 
    "create mp4 video file from gif"
    if gif[-4::] != '.gif': 
        gif = gif + '.gif'
    if video_file is None: 
        video_file = gif[0:-4] + '.mp4'
    elif video_file[-4::] != '.mp4': 
        video_file = video_file + '.mp4'
    clip = ed.VideoFileClip(gif)
    clip.write_videofile(video_file)
    print(f'Gif converted to mp4: {video_file}')
    return 