#!/usr/bin/env python
#############################################################################

import argparse
import glob
import json
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

import os
import time

from PIL import Image, ImageTk
from tkinter import Label, Button, Text, Scrollbar, VERTICAL, RIGHT, Y, Frame, filedialog, messagebox
import threading
import tkinter as tk
import tkinter.ttk as ttk


##
 #
 # Welcome to Trainspotting
 #
 ##
class Trainspotting:
    LOG = logging.getLogger(__name__)

    def __init__(self):
        self.parser = argparse.ArgumentParser(description="This is pythong")
        self.parser.add_argument("--sleep", type=int, default=10)
        self.parser.add_argument("--scale", type=int, default=1)
        self.parser.add_argument("--dir",   type=str, default="last/images")
        self.images = {}
        self.pngs = []

    def main(self):
        args = self.args = self.parser.parse_args()
        Trainspotting.LOG.info(f'settings: {json.dumps(vars(args))}')

        tk_root = tk.Tk()
        tk_root.title(f"TrainSpotting 🚂 {self.args.dir} 👀")
        top_frame = Frame(tk_root)
        top_frame.pack(side="top")

        #image = Image.open('images/lenna.png')
        #self.photo = ImageTk.PhotoImage()#image)
        #self.label = Label(top_frame, image=self.photo)
        self.label = Label(top_frame, text="1 sec")
        self.label.pack(side="top", padx=10, pady=10)

        info = Label(top_frame, text="<waiting on images>")
        info.pack(side="bottom", padx=10, pady=10)

        running = True
        self.current = -1

        def show_image():
            count = len(self.pngs)
            if self.current < 0:
                self.current = count - 1
            elif self.current >= count:
                self.current = count - 1
            name = self.pngs[self.current]
            if name in self.images:
                self.photo = self.images[name]
                self.label.image = self.photo
                self.label.config(image=self.photo)
                text = f"{self.current + 1} of {count}: {name}"
                print("SHOWING:", text, "and", name)
                info.config(text = text)
            else:
                print("derp")

        def key_press(event):
            print(f"Key pressed: {event.keysym}")
            if "q" == event.keysym or "Escape" == event.keysym:
                tk_root.destroy()
                tk_root.quit()
                running = False
                ###exit(0)
            if "Prior" == event.keysym:
                self.current = self.current - 10
                show_image()
            if "Next" == event.keysym:
                self.current = self.current + 10
                show_image()
        tk_root.bind("<Key>", key_press)

        def mouse_wheel(event, idk = 0):
            delta = idk + event.delta
            self.current = self.current + idk + event.delta
            show_image()
            print(f">Mouse wheel scrolled: {event.delta} and {idk} so {delta} -> {self.current}")
        tk_root.bind("<MouseWheel>", mouse_wheel)
        tk_root.bind("<B2-Motion>", mouse_wheel)
        tk_root.bind("<Button-4>", lambda event: mouse_wheel(event, 1))
        tk_root.bind("<Button-5>", lambda event: mouse_wheel(event, -1))


        def old_watch_dir():
            latest = None
            while running:
                last = self.get_latest_png()
                if last and last != latest:
                    print("loaded ", last)
                    image = Image.open(last)
                    photo = ImageTk.PhotoImage(image)
                    label.config(image=self.photo)
                    label.image = self.photo
                    latest = last
                time.sleep(self.args.sleep)
            print("cu")

        def watch_dir():
            while running:
                loaded = self.imager()
                if loaded:
                    count = len(self.images)
                    if self.current < 0 or self.current >= count - 2:
                        print(f"YEEP: {self.current} of {count}")
                        self.current = -1
                        show_image()
                        self.current = -1
                    else:
                        print(f"NOPE: {self.current} of {count}")
                    info.config(text = f"{self.current+1} of {count}: {self.pngs[self.current]}")
                time.sleep(self.args.sleep)
            print("cu")
        thread = threading.Thread(target=watch_dir)
        thread.start()


        tk_root.mainloop()
        running = False

        #tk_root.destroy()
        tk_root.quit()
    # end of main

    def get_latest_png(self):
        directory = self.args.dir
        latest_png = None
        latest_mtime = 0
        for filename in os.listdir(directory):
            if filename.endswith(".png"):
                filepath = os.path.join(directory, filename)
                mtime = os.path.getmtime(filepath)
                if mtime > latest_mtime:
                    latest_png = filepath
                    latest_mtime = mtime
        if not latest_png:
            latest_png = 'images/lenna.png'
        return latest_png
    # end of get_latest_png

    def imager(self):
        self.pngs = glob.glob(os.path.join(self.args.dir, '*.png'))
        self.pngs.sort(key=lambda x: os.path.getmtime(x))
        loaded = False
        for png in self.pngs:
            if png in self.images:
                continue
            try:
                image = Image.open(png)
                image = image.resize((image.width * self.args.scale, image.height * self.args.scale))
                self.images[png] = ImageTk.PhotoImage(image)#Image.open(png))
                loaded = True
            except Exception as error:
                pass
        return loaded


# end of class Trainspotting

if __name__ == "__main__":
    Trainspotting().main()

# EOF
#############################################################################
