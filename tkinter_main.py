from tkinter import *
from PIL import ImageTk, Image
from TransferLearning import activateVGG,activateResNet
root = Tk()
root.title("Object classification")
root.iconbitmap('icon.ico')

temp = Image.open('back.jpg')
temp = temp.resize((350,300),Image.ANTIALIAS)
back = ImageTk.PhotoImage(temp)

def OnClick():
    print("Hello")

def ResActiv():
   activateResNet()

def VGGActiv():
   activateVGG()

mylabel=Label(image=back)
mylabel.grid(row=0,column=1,columnspan=2)

upload = Button(root,text='Classify with Resnet',padx=40,pady=20,command=ResActiv)
choose_model = Button(root,padx=40,pady=20,text='Classify with VGG',command=VGGActiv)
classify = Button(root,padx=40,pady=20,text='Classify the picture',command=OnClick)


upload.grid(row=1,column=1)
choose_model.grid(row=1,column=2)
classify.grid(row=2,column=1,columnspan=2)
root.mainloop()
