from tkinter import *
from sdd import *

"""
by Nadav Loebl
nadavloebl@gmail.com
"""

### HEADER ###
root = Tk()
root.title("Social Distancing Detector 1.0")
CENTER = 3
LEFT = 3
RIGHT = 4
BUTTONS_WIDTH = 16
title = Label(root, text="Social Distancing Detector", font=("Helvetica", 28, 'bold')).grid(row=1, column=CENTER)

Label(root, text="If you want to use IP Camera:", font=("Helvetica", 20)).grid(row=4, column=CENTER)
Label(root, text="or leave it empty and the webcam will be used", font=("Helvetica", 16)).grid(row=5, column=CENTER)

user = StringVar()
Entry(root, width = 15, textvariable = user).grid(row=4, column=RIGHT+1)
Label(root, text="username:", font=("Helvetica", 16)).grid(row=4, column=RIGHT)

password = StringVar()
Entry(root, width = 15, textvariable = password).grid(row=5, column=RIGHT+1)
Label(root, text="password:", font=("Helvetica", 16)).grid(row=5, column=RIGHT)

address = StringVar()
Entry(root, width = 15, textvariable = address).grid(row=6, column=RIGHT+1)
Label(root, text="address:", font=("Helvetica", 16)).grid(row=6, column=RIGHT)

Label(root, text="Select alarm type:", font=("Helvetica", 20)).grid(row=8, column=CENTER)
Label(root, text="or leave it empty for no alarm", font=("Helvetica", 16)).grid(row=9, column=CENTER)

sound = BooleanVar()
phone = StringVar()

Checkbutton(root, text = "sound", variable = sound, onvalue = True, offvalue = False, font=("Helvetica", 16)).grid(row=8,column=RIGHT)
Entry(root, width = 15, textvariable = phone).grid(row=9, column=RIGHT+1)
Label(root, text="sms to phone:", font=("Helvetica", 16)).grid(row=9, column=RIGHT)

def pre_run():
    camera = 'webcam'
    username = user.get()
    passw = password.get()
    addr = address.get()

    cam_prefix  = ''
    if len(username) > 0:
        cam_prefix = username + ':' + passw + '@'

    if len(addr) > 0:
        camera = 'http://' + cam_prefix + addr

    run(camera=camera, sound=sound.get(), sms=phone.get())


Button(root, text='START', font=('helvetica', 28, 'bold'), command=pre_run).grid(row=15,column=CENTER)

root.mainloop()
