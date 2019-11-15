from ksdyn.core import KeypressEventReceiver, Named, DictTree

import numpy as np
from abc import ABCMeta, abstractmethod
from collections import defaultdict


class Feature(Named):
    '''A feature (in the machine learning sense) from a given typist'''
    __metaclass__=ABCMeta

class CompositeFeature(DictTree, Feature): #组合特征
    '''A feature composed of multiple sub-features'''
    pass

class FloatSeq(Feature): #实数序列
    '''A sequence of real numbers'''
    def __init__(self, name, data):
        Feature.__init__(self, name)
        self.data= data

class KeyDwellTimes( FloatSeq ): #按键按下时的时间序列
    '''A sequence of times time while a certain keyboard key is pressed.
    The "name" attribute of this feature is the key name'''
    pass
 
class FeatureExtractor(KeypressEventReceiver): #特征获取器
    '''Extracts features from keypress data'''
    def __init__(self, timing_threshold=500): 
        self.pt=0           #last press  time 上一次按键时间
        self.pk=0           #last pressed key 上一次按键键值
        self.press_time={}  #dictionary that associates currently depressed keys and 将当前按键与其按下时间相关联的字典
                            #their press times. Necessary because a key may be pressed
                            #before the preceding key is released 释放前一个键前可能已按下下一个键
        #时间间隔大于阈值则无视该击键事件，中间停顿时间过长不符合连续击键特征
        self.timing_threshold= timing_threshold   # if timing betweeen events is bigger than this, ignore those events
        self.dwell_times=           defaultdict(list) #按键持续时间
        self.flight_times_before=   defaultdict(list) #与上次按键间隔
        self.flight_times_after=    defaultdict(list) #与下次按键间隔

    def on_key(self, key, type, time): #基于每次按键事件的时间数据计算
        if type==self.KEY_DOWN: #若为按下按键
            flight_time= time - self.pt #本次按键与上次按键的间隔时间
            if flight_time<self.timing_threshold: #判断是否在时间阈值内
                self.flight_times_before[key].append(flight_time) #添加至该击键的上次按键间隔列表字典
                self.flight_times_after[self.pk].append(flight_time) #添加至上次击键的下次按键间隔列表字典
            self.press_time[key]=time #记录本次按键时间及键值至字典，用于按键持续时间计算
            self.pt=time #更改击键时间，用于下一按键时间间隔计算
            self.pk=key #更改击键键值，作用同上
        
        if type==self.KEY_UP: #若为释放按键
            try:
                dwell_time= time - self.press_time.pop(key) #计算按键持续时间并删除该按键（该按键动作已结束）
            except KeyError:
                #can happen because we initiated capture with a key pressed down
                return
            if dwell_time<self.timing_threshold: #若按键持续时间小于阈值，则加入列表字典
                self.dwell_times[key].append(dwell_time)
    
    def extract_features( self ): 
        '''Extracts the features from the processed data.
        Returns a CompositeFeature, composed of multiple other CompositeFeatures.'''
        dwell_times= [KeyDwellTimes(k, v) for k,v in self.dwell_times.items()]

        return CompositeFeature( "dwell_times", dwell_times ) 
