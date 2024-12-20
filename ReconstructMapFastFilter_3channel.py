#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 09:26:55 2020

@author: tianxianhao
"""

import numpy as np
import os
from PIL import Image 
import sys
import math
import time
from numba import njit
from scipy.ndimage import median_filter, grey_closing, binary_opening

def view_bar(message,num,total):
    rate=num/total
    rate_num=int(rate*40)
    rate_nums=math.ceil(rate*100)
    r='\r%s:[%d/%d][%s%s] %d%%' % (message,num,total,">"*rate_num," "*(40-rate_num),rate_nums,) 
    sys.stdout.write(r)
    sys.stdout.flush()
    
@njit    
def Seq2Block(seqs,MB_x_num,MB_y_num):
    MB_number=MB_x_num*MB_y_num
    MB44xnum=MB_x_num*4
    MB44ynum=MB_y_num*4
    img=np.zeros((MB44ynum,MB44xnum),dtype=np.uint8)    
    for i in range(MB_number):
        curr_seq=seqs[i,:]
        if np.sum(curr_seq)>0:
            img44=curr_seq.copy().reshape((4,4))
            MB_x=i%MB_x_num
            MB_y=(i-MB_x)/MB_x_num
            MB_x,MB_y= int(MB_x*4),int(MB_y*4)
            img[MB_y:MB_y+4,MB_x:MB_x+4]=img44
    return img


def preprocessing(data, kernel):
    frame_interval = 250
    framenumber, mbnumber = data[-1,0], data[-1,1]+1
    feat = data.reshape((framenumber,mbnumber,40))
    
    #interpolate I frame
    feat[:,:,3:40] = interpolate(feat[:,:,3:40], framenumber, frame_interval)
    
    #med_filter    
    feat[:,:,3] = median_filter(feat[:,:,3],size=(kernel,1),mode='nearest')
    feat[:,:,8:24] = median_filter(feat[:,:,8:24],size=(kernel,1,1),mode='nearest')
    # residual filter
    tmp = grey_closing(feat[:,:,24:40],(kernel-1,1,1))
#    tmp1 = binary_opening(tmp,np.ones((kernel,1,1)))
    feat[:,:,24:40] = tmp# * tmp1
    feat = feat.reshape((framenumber*mbnumber,40))
    return feat
    
@njit
def interpolate(feat, framenumber, frame_interval):
    ind = np.arange(0,framenumber,frame_interval)
    feat[0] = feat[1] * 0.67 + feat[2] * 0.33
    endbound = framenumber - ind[-1] - 1
    if endbound == 0:
        feat[-1] = feat[-2] * 0.67 + feat[-3] * 0.33
    elif endbound == 1:
        feat[-2] = feat[-3] * 0.333 + feat[-4] * 0.167 + feat[-1] * 0.5
    else:
        feat[ind[-1]] = feat[ind[-1]-2] * 0.167 + feat[ind[-1]-1] * 0.333 + feat[ind[-1]+1] * 0.333 + feat[ind[-1]+2] * 0.167
    for i in range(1,len(ind)-2):
        feat[ind[i]] = feat[ind[i]-2] * 0.167 + feat[ind[i]-1] * 0.333 + feat[ind[i]+1] * 0.333 + feat[ind[i]+2] * 0.167
    feat = feat.astype(np.int32)
    return feat

def main():
    FilePath=os.getcwd()
    VideoResolutionPath=os.path.join(FilePath,'Encryption','EncryptTimeLog.txt')
    MBfeat_file='MBfeature.txt'
    ''' MB file context
    column 0: frame index
    column 1: mb addr
    column 2: mb type
    column 3: mb size, 16x16
    column 4-7 : mb depth, 8x8
    column 8-23: mvd, 4x4
    column 24-39: number of nonzero dct coefficient, 4x4
    '''
    videolist=os.listdir(os.path.join(FilePath,'FeatureInfo'))
    video_info=np.loadtxt(VideoResolutionPath,delimiter=',',dtype=str)
    video_name=video_info[:,0]
    video_res=video_info[:,1]
    pid_ind=0
    pid_num=1
    start_time=time.perf_counter()
    mean_nc=[]
    mean_mvd=[]
    mean_mbs=[]
    mean_dep=[]
    std_nc=[]
    std_mvd=[]
    std_mbs=[]
    std_dep=[]
    for vn in videolist[pid_ind::pid_num]:
    # for vn in ['hockey']:
        feat_filepath=os.path.join(FilePath,'FeatureInfo',vn,MBfeat_file)
        save_dir=os.path.join(FilePath,'feature_map_dir','features',vn)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        print(f'Processing video {vn} ......')    
        video_ind=np.where(video_name==vn)
        res=[int(np.ceil(int(i)/16)) for i in video_res[video_ind[0][0]].split('x')]
        MB_x_num, MB_y_num= int(res[0]), int(res[1])
        load_begin_time=time.perf_counter()
        feat_data=np.loadtxt(feat_filepath,delimiter=',',dtype=np.int32)
        print('loading txt time: ',time.perf_counter()-load_begin_time)
        pre_pro_time = time.perf_counter()
        feat_data = preprocessing(feat_data, 5)
        print('pre-processing data time: ',time.perf_counter()-pre_pro_time)
        frame_ind=feat_data[:,0]
        mb_addr=feat_data[:,1]
        mb_size=feat_data[:,3]
        mb_size=np.expand_dims(mb_size,axis=-1)
        mb_size[mb_size>256]=256
        mb_size=np.repeat(mb_size,16,axis=-1)
        mb_dep=feat_data[:,4:8]
        mb_dep=mb_dep/4*256
        mb_dep=mb_dep.astype(np.int32)
        mb_dep=mb_dep[:,[0,0,1,1,0,0,1,1,2,2,3,3,2,2,3,3]]
        mb_mvd=feat_data[:,8:24]
        mb_mvd[mb_mvd>20]=20
        mb_mvd=mb_mvd/20*256
        mb_mvd=mb_mvd.astype(np.int32)
        mb_nc=feat_data[:,24:40]
        mb_nc[mb_nc>16]=16
        mb_nc=mb_nc/16*256
        mb_nc=mb_nc.astype(np.int32)
        frame_number=frame_ind[-1]
        mb_number=mb_addr[-1]+1

        assert mb_number==(MB_x_num*MB_y_num),f'MB number {mb_number} not equal to MB_x * MB_y'
        for fn in range(1,frame_number+1):
            
            begin_ind=(fn-1)*mb_number
            end_ind=begin_ind+mb_number

            nc_img=Seq2Block(mb_nc[begin_ind:end_ind,:],MB_x_num,MB_y_num)
            mean_nc.append(np.mean(nc_img))
            std_nc.append(np.std(nc_img))

            mvd_img=Seq2Block(mb_mvd[begin_ind:end_ind,:],MB_x_num,MB_y_num)
            mean_mvd.append(np.mean(mvd_img))
            std_mvd.append(np.std(mvd_img))

            mbs_img=Seq2Block(mb_size[begin_ind:end_ind,:],MB_x_num,MB_y_num)
            mean_mbs.append(np.mean(mbs_img))
            std_mbs.append(np.std(mbs_img))

            dep_img=Seq2Block(mb_dep[begin_ind:end_ind,:],MB_x_num,MB_y_num)
            mean_dep.append(np.mean(dep_img))
            std_dep.append(np.std(dep_img))

            # 创建4通道特征
            one_frame_feat = np.stack((nc_img, mvd_img, mbs_img, dep_img), axis=-1)
            
            # 保存为npy文件
            savepath = os.path.join(save_dir, '%06d.npy' % fn)
            np.save(savepath, one_frame_feat)
            
            view_bar('Processed frames', fn, frame_number)
        print('\n')
        print("Mean_NC=", mean_nc[-1], ", Mean_MVD=", mean_mvd[-1], ", Mean_MBS=", mean_mbs[-1], ", Mean_DEP=", mean_dep[-1])
        print("Std_NC=", std_nc[-1], ", Std_MVD=", std_mvd[-1], ", Std_MBS=", std_mbs[-1], ", Std_DEP=", std_dep[-1])

    print('Finish the generation of feature tensors!')
    end_time=time.perf_counter()
    print("TIME USE: ",end_time-start_time,' sec.')
    print("Mean_NC=",np.mean(mean_nc),", Mean_MVD=",np.mean(mean_mvd),", Mean_MBS=",np.mean(mean_mbs), ", Mean_DEP=",np.mean(mean_dep))
    print("Std_NC=",np.mean(std_nc),", Std_MVD=",np.mean(std_mvd),", Std_MBS=",np.mean(std_mbs),", Std_DEP=",np.mean(std_dep))
if __name__=='__main__':
    main()
