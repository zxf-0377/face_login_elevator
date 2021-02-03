import tkinter
from tkinter import filedialog

import win32api
import os
from face_operator import face_reconition



class My_GUI(tkinter.Tk):
    def __init__(self):
        super().__init__()
        self.resizable(False,False)
        self.title("基于卷积神经网络的网页人脸登录系统管理端")
        # self.geometry('640x540+500+200')
        self.center_window(640,440)
        # 背景组件
        self.frame_top = tkinter.Frame(width=640,height=540,bg="#021C31")
        self.frame_top.grid(row=0,column=0)
        # 图片
        # 显示图片
        self.photo = tkinter.PhotoImage(file='static\\img\\2.png')
        self.imageLabel = tkinter.Label(self.frame_top, image=self.photo,bg="#222322")
        self.imageLabel.place(x=0,y=-50)
        # 启动组件
        self.btn1 = tkinter.Button(self.frame_top,text='启动服务器',command=self.exec_face_server)
        self.btn1.place(x=410,y=140,width=200,height=50)
        # 批量注册组件
        self.frame_below = tkinter.Frame(width=300,height=100,pady=10,bg="#222322")
        self.frame_below.place(x=410,y=200)
        self.frame_below.propagate(0)
        # self.label1 = tkinter.Label(self.frame_below,text='批量注册')
        # self.label1.place(x=50,y=-30)
        self.entry1 = tkinter.Entry(self.frame_below)

        self.entry1.grid(row=1,column=0,padx=2)
        self.btn2 = tkinter.Button(self.frame_below,text='浏览',command=self.return_directory)
        self.btn2.grid(row=1,column=1,padx=5)
        self.btn3 = tkinter.Button(self.frame_below,text='批量注册',command=self.exec_face_operator)
        self.btn3.grid(row=2,column=0,padx=2,columnspan=2)

    def return_directory(self):
        self.directory_name = tkinter.filedialog.askdirectory()
        self.entry1.delete(0, tkinter.END)
        self.entry1.insert(0,self.directory_name)
        return self.directory_name

    # 执行face_server.exe执行模块，因为是死循环，face_server.py需要发布之后，通过GUI显式调用
    def exec_face_server(self):
        win32api.ShellExecute(0,'open',os.path.join(os.path.abspath('.'),'dist\\face_server\\face_server.exe'),'','',1)

    def exec_face_operator(self):
        # 获取Entry框中的路径信息，并将其内部所有文件拷贝到指定路径
        src_path = self.entry1.get()
        modelpath = '.\\models\\facenet\\20170512-110547'
        out_path = '.\\img\\pic.json'
        face_reconition_class = face_reconition()
        face_reconition_class.images_to_vectors(src_path,out_path,modelpath)

    # def kill_face_server(self):
    #     os.system('taskkill /f /t /im python.exe')

    # def exec_face_operator(self):


    # 窗体居中
    def center_window(self, width, height):
        screenwidth = self.winfo_screenwidth()
        screenheight = self.winfo_screenheight()
        # 宽高及宽高的初始点坐标
        size = '%dx%d+%d+%d' % (width, height, (screenwidth - width) / 2, (screenheight - height) / 2)
        self.geometry(size)





if __name__=='__main__':
    top = My_GUI()
    top.mainloop()