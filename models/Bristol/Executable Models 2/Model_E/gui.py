import tkinter as tk
from tkinter import filedialog

import ctypes
from tkinter import *
from tkinter.ttk import *

#创建窗口，root可替换成自己定义的窗口
app = tk.Tk()
#调用api设置成由应用程序缩放
ctypes.windll.shcore.SetProcessDpiAwareness(1)
#调用api获得当前的缩放因子
ScaleFactor=ctypes.windll.shcore.GetScaleFactorForDevice(0)
#设置缩放因子
app.tk.call('tk', 'scaling', ScaleFactor / 75)


def browse_file():
    file_path = filedialog.askopenfilename()
    if file_path:
        file_path_label.config(text="File path: " + file_path)
    else:
        file_path_label.config(text="No file selected.")


# Create the main application window
app.title("File Selector")

# Set the window size and position
app.geometry("400x150")
#app.resizable(False, False)

# Set the font size for the label
label_font = ("Arial", 12)

# Create a label to display the selected file path
file_path_label = tk.Label(app,
                           text="No file selected.",
                           font=label_font,
                           wraplength=350)
file_path_label.pack(pady=20)

# Create a button to browse for a file
button_font = ("Arial", 12, "bold")
browse_button = tk.Button(app,
                          text="Browse",
                          font=button_font,
                          command=browse_file)
browse_button.pack(pady=10)

# Start the main event loop
app.mainloop()
