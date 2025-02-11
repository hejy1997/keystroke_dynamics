#!/usr/bin/python
#import sys
#import os
；from Xlib import X, XK, display  #X为应用程序使用的X协议声明类型和常量；XK为键盘扩展功能；display对象用于显示信息，包含很多方法
from Xlib.ext import record #record用于将X window的事件和请求都记录下来
from Xlib.protocol import rq #protocol为基本的X协议声明类型和符号，用于实现扩展
#Xlib是一个用c语言编写的X Window System协议的客户端库，它包含有与x服务器进行通信的函数，编程者可以在不了解x底层协议的情况下直接使用它进行编程。

local_dpy = display.Display()
record_dpy = display.Display()
myrecordcallback=''

if not record_dpy.has_extension("RECORD"):
    print "RECORD extension not found"
    sys.exit(1)
ctx = record_dpy.record_create_context(
            0,
            [record.AllClients],
            [{
                    'core_requests': (0, 0),
                    'core_replies': (0, 0),
                    'ext_requests': (0, 0, 0, 0),
                    'ext_replies': (0, 0, 0, 0),
                    'delivered_events': (0, 0),
                    'device_events': (X.KeyPress, X.KeyRelease),
                    'errors': (0, 0),
                    'client_started': False,
                    'client_died': False,
            }])


def lookup_keycode(keycode):
    keysym = local_dpy.keycode_to_keysym(keycode, 0)
    if keysym:
        for name in dir(XK):
            if name[:3] == "XK_" and getattr(XK, name) == keysym:
                return name[3:]
        return "[%d]" % keysym
    else:
        return None
    


def record_callback(reply):
    if reply.category != record.FromServer:
        return
    if reply.client_swapped:
        print "* received swapped protocol data, cowardly ignored"
        return
    if not len(reply.data) or ord(reply.data[0]) < 2:
        return

    data = reply.data
    while len(data):
        event, data = rq.EventField(None).parse_binary_value(data, record_dpy.display, None, None)
        if event.type in [X.KeyPress, X.KeyRelease]:
            key=event.detail
            
            if key == 9: #escape
                local_dpy.record_disable_context(ctx)
                local_dpy.flush()
                return
                
            if event.type==X.KeyPress:
                myrecordcallback(key,0, event.time)
            if event.type==X.KeyRelease:
                myrecordcallback(key,1, event.time)
                
def start(mycallback):
    # Enable the context; this only returns after a call to record_disable_context,
    # while calling the callback function in the meantime
    global myrecordcallback
    myrecordcallback=mycallback
    record_dpy.record_enable_context(ctx, record_callback)
    record_dpy.record_free_context(ctx)

