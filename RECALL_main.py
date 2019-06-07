#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 23 15:27:15 2019

@author: matthewconlin
"""

# Need to run %gui tk in console, and/or go to preferences and change graphics backend to tk #

import tkinter

# Create the window #
root = tkinter.Tk()
root.title('RECALL')

# Create the main frame #
placeVar = tkinter.StringVar(root)
choices = {'Miami','Twin Piers/Bradenton','St. Augustine','Folly Beach South','Folly Beach North','Cherry Grove South','Buxton','Other'}

# Create the frame %
tkinter.Label(root,text = 'Select Camera:').grid(row=0)
menu = tkinter.OptionMenu(root,placeVar,*choices).grid(row=0,column=1)

root.mainloop()


import tkinter as tk 
r = tk.Tk() 
r.title('Counting Seconds') 
button = tk.Button(r, text='Stop', width=25, command=r.destroy) 
button.pack() 
r.mainloop() 



import tkinter as tk

class Application(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.pack()
        self.create_widgets()

    def create_widgets(self):
        self.hi_there = tk.Button(self)
        self.hi_there["text"] = "Hello World\n(click me)"
        self.hi_there["command"] = self.say_hi
        self.hi_there.pack(side="top")

        self.quit = tk.Button(self, text="QUIT", fg="red",
                              command=self.master.destroy)
        self.quit.pack(side="bottom")

    def say_hi(self):
        print("hi there, everyone!")

root = tk.Tk()
app = Application(master=root)
app.mainloop()



