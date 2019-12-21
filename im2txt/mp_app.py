import time
import sys
import os
import json
import math
import pickle
import queue
import run_inference as rf
import find_images
import collections
import facenet.src.align.align_mp as fca
import facenet.src.validate_on_lfw as fsv
import facenet.src.align.clustering as fsac
import platform
from operator import itemgetter 
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from multiprocessing import Pool, Process, Queue



class Photo(QWidget):
    def __init__(self,parent=None):
        super(Photo,self).__init__()
        
    
    def initUI(self,img_path):
        pixmap = QPixmap( img_path)
        if( pixmap.height()>1080 or pixmap.width()>680 ):
            pixmap = pixmap.scaled(1080,680,Qt.KeepAspectRatio)
        lbl_img = QLabel()
        lbl_img.setPixmap(pixmap)
        vbox = QVBoxLayout()
        vbox.addWidget(lbl_img)        
        self.setLayout(vbox)
        self.move(0,0)
        self.show()



class Example(QWidget):
    
    def __init__(self):
        super(Example,self).__init__()
        #self.setFocusPolicy(Qt.StrongFocus)
        self.initUI()
    
    def mousePressEvent(self,event):
        wid = self.vwid.childAt(event.pos())
        try:
            
            if hasattr(wid,'fileName'): 
                if wid.fileName[:4] == "face":
                    name = wid.fileName[5:]
                    #print(name)
                    
                    self.final = []
                    for i in self.face_dic[name]:
                        tmp = {"path":i,"caption_0":self.data[i]["caption_0"]}
                        self.final.append(tmp)
                        #print(orig_name)
                        
                    self.display(0)

                if wid.fileName == "image":
                    pa = wid.parent()
                    ch = pa.children()
                    #print(ch[2].text())
                    self.img_dir = ch[2].text()
                    if self.img_dir[-4:] in self.formats or self.img_dir[-5:] in self.formats:
                        self.new_window = Photo()
                        self.new_window.initUI(self.img_dir)
                        self.vwid.setFocus()
                    #self.setFocusPolicy(Qt.StrongFocus)
        except Exception as e:
            print("error"  + str(e))
   

    def keyPressEvent(self,event):
        #print(event.key())
        if event.key() == Qt.Key_Left:
            if hasattr(self.grid.itemAtPosition(self.ind+1,0),'widget'):
                self.start -= self.display_limit
                self.display(self.start)

        if event.key() == Qt.Key_Right:
            print(self.grid.itemAtPosition(self.ind+1,self.col-1))
            if hasattr(self.grid.itemAtPosition(self.ind+1,self.col-1),'widget'):
                self.start +=self.display_limit
                self.display(self.start)
   

    def initUI(self):      

        self.dr = os.path.dirname(os.path.abspath(__file__))
        self.fname = "-1"
        self.stop_words = ["a","about","above","after","again","against","all","am","an","and","any","are","aren't","as","at","be","because","been","before","being","below","between","both","but","by","can't","cannot","could","couldn't","did","didn't","do","does","doesn't","doing","don't","down","during","each","few","for","from","further","had","hadn't","has","hasn't","have","haven't","having","he","he'd","he'll","he's","her","here","here's","hers","herself","him","himself","his","how","how's","i","i'd","i'll","i'm","i've","if","in","into","is","isn't","it","it's","its","itself","let's","me","more","most","mustn't","my","myself","no","nor","not","of","off","on","once","only","or","other","ought","our","ours   ourselves","out","over","own","same","shan't","she","she'd","she'll","she's","should","shouldn't","so","some","such","than","that","that's","the","their","theirs","them","themselves","then","there","there's","these","they","they'd","they'll","they're","they've","this","those","through","to","too","under","until","up","very","was","wasn't","we","we'd","we'll","we're","we've","were","weren't","what","what's","when","when's","where","where's","which","while","who","who's","whom","why","why's","with","won't","would","wouldn't","you","you'd","you'll","you're","you've","your","yours","yourself","yourselves"]
        self.formats = ['.jpg','.JPG','.jpeg','.JPEG']
        
        self.init_data()

        select_file = QPushButton('choose folder (optional)')
        self.file_lbl = QLabel()
        select_file.clicked.connect(self.showDialog)


        crd = QPushButton('create database')
        crd.clicked.connect(lambda: self.crdb(self.file_lbl.text()))
        self.crd_label = QLabel(self)

        up = QPushButton("update database.")
        up.clicked.connect(lambda: self.crdb(self.file_lbl.text(),True))
        self.up_lbl = QLabel()

        sq = QPushButton("search query")
        self.qle = QLineEdit()
        self.qle.returnPressed.connect(sq.click)
        sq.clicked.connect(self.search)

        self.faces = QHBoxLayout()
        self.load_faces()

        crd_box = QHBoxLayout()
        crd_box.addWidget(crd)
        crd_box.addWidget(self.crd_label)

        hbox = QHBoxLayout()
        hbox.addWidget(select_file)
        hbox.addWidget(self.file_lbl)

        hbox1 = QHBoxLayout()
        hbox1.addWidget(sq)
        hbox1.addWidget(self.qle)

        hbox2 = QHBoxLayout()
        hbox2.addWidget(up)
        hbox2.addWidget(self.up_lbl)

        vbox = QVBoxLayout()
        #vbox.addLayout(ip_box)
        vbox.addLayout(hbox)
        vbox.addLayout(crd_box)
        vbox.addLayout(hbox2)
        vbox.addLayout(hbox1)
        vbox.addLayout(self.faces)

        self.scrollArea = QScrollArea(self)
        self.scrollArea.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.container = QWidget()
        self.grid = QGridLayout()
        self.container.setLayout(self.grid)
        self.scrollArea.setWidgetResizable(True)
        self.scrollArea.setWidget(self.container)
        vbox.addWidget(self.scrollArea)
        
        self.vwid = QWidget()
        self.vwid.setLayout(vbox)
        #self.setLayout(vbox)
        #self.setWidget(vwid)
        self.vwid.setWindowTitle('MP')
        self.vwid.setWindowState(Qt.WindowMaximized)
        self.vwid.setFocusPolicy(Qt.StrongFocus)
        
        self.vwid.keyPressEvent = self.keyPressEvent
        self.vwid.mousePressEvent = self.mousePressEvent
        self.vwid.show()  


    def showDialog(self):   #to choose foder of images
        self.fname = QFileDialog.getExistingDirectory(self, 'Open file', 'c:\\')
        self.file_lbl.setText(self.fname)

    def load_faces(self):
        if os.path.isfile( "agcls.pk"):
            file = open("agcls.pk","rb")
            self.face_dic = pickle.load(file)
            self.face_dir = self.dr + "/../facenet/src/align/keys/"
            file.close()
            self.face_list = []
            for key, value in self.face_dic.items():
                self.face_list.append(key)
            self.face_menu(0)

    def face_menu(self,start_index):

        for i in reversed(range(self.faces.count())): 
            self.faces.itemAt(i).widget().setParent(None)

        fl_len = len(self.face_list)
        fm_len = 12

        for i in range(fm_len):
            if fl_len>start_index+i:
                face_lbl = QLabel()
                name = self.face_dir + self.face_list[start_index+i]
                pixmap = QPixmap( name )
                pixmap1 = pixmap.scaled(50,50,Qt.KeepAspectRatio)
                face_lbl.setPixmap(pixmap1)
                face_lbl.fileName = "face_" + self.face_list[start_index+i]
                self.faces.addWidget(face_lbl)
            else:
                break

        if start_index + fm_len< fl_len:
            next_but = QPushButton("next")
            next_but.clicked.connect(lambda : self.face_menu(start_index+fm_len))
            self.faces.addWidget(next_but)

        if start_index>0:
            prev_but = QPushButton("prev")
            prev_but.clicked.connect(lambda : self.face_menu(start_index-fm_len))
            self.faces.addWidget(prev_but)


    def crdb(self,folder_path=None,upd=False): 
        if upd:
            name = self.up_lbl
        else:
            name = self.crd_label
            
        self.msg(name,"searching for images...")
        
        que = Queue()
        start_time = time.time()
        p = Process(target=find_images,args=(que,folder_path,))
        p.start()
        fold = que.get()
        self.ploc = {}
        photos = []
        if upd:
            new_photos = []

        while not fold == "end":
            if not que.empty():
                photos.append(fold)
                ind = fold.rfind("/")
                orig_name = fold[ind+1:]
                self.ploc[orig_name] = fold
                if upd and fold not in self.photos:
                    new_photos.append(fold)
                fold = que.get()

        p.join()
        print(time.time()-start_time)


        if upd:
            self.msg(name,"runnning image captioning and face detection algorithm on "  + str(len(new_photos)) + "new photos" )
        else:
            self.msg(name,"runnning image captioning and face detection algorithm on " + str(len(photos)) + "photos."  )

        with open("photos.pk","wb") as f:
            pickle.dump(photos,f)

        with open("ploc.pk","wb") as f:
            pickle.dump(self.ploc,f)
        

        if upd:
            if len(new_photos) > 0:
                print("here")
                """
                p1 = Process( target=rf.main, args=(new_photos,True) )
                p2 = Process( target=fca.main_align, args=(new_photos,) )

                p1.start()
                p2.start()
                p1.join()
                p2.join()
                """
                rf.main(new_photos,True)
                fca.main_align(new_photos)
                print(time.time()-start_time)
                fsv.main_embed(True)
                fsac.cluster_faces(True)
        

        else:
            
            p1 = Process( target=rf.main, args=(photos,) )
            p2 = Process( target=fca.main_align, args=(photos,) )

            p1.start()
            p2.start()
            p1.join()
            p2.join()
            fsv.main_embed()
            fsac.cluster_faces()
        
        
        print(time.time()-start_time)
        
        self.init_data()
        self.load_faces()        
            
        self.msg(name,"done.")

    
    def init_data(self):
        if os.path.isfile("photos.pk"):
            file = open("photos.pk","rb")
            self.photos = pickle.load(file)
            file.close()

        if os.path.isfile("ploc.pk"):
            file = open("ploc.pk","rb")
            self.ploc = pickle.load(file)
            file.close()

        im_cap =  "open.pk"
        #print(im_cap)
        if os.path.isfile(im_cap):
            file  = open(im_cap,"rb")
            self.data = pickle.load(file)
            file.close()


    def msg(self,lab,str):
        lab.setText(str)
        app.processEvents()    



    def search(self):
        self.qle.clearFocus()
        sent = self.qle.text()
        terms = sent.split()
        fin = []
        for term in terms:
            if term not in self.stop_words:
                fin.append(term)
                #print(term)

        
        
        index = len(self.data)

        self.final = []
        dic = {}
        for key,value in self.data.items():
            for term in fin:
                if "caption_0" in self.data[key] and term in self.data[key]["caption_0"].split():
                    if key in dic:
                        dic[key]["cnt"] += 1
                    else:
                        dic[key] = {"caption_0":self.data[key]["caption_0"],"cnt":1}
                        
        
        for key,value in dic.items():
            tmp = {"path":key,"caption_0":value["caption_0"],"cnt":value["cnt"]}
            self.final.append(tmp)
    

        self.final = sorted(self.final, key=itemgetter('cnt'), reverse=True)
        self.start = 0
        self.display(self.start)


    def display(self,start):
        self.start = start
        self.lim = len(self.final)
        self.display_limit = 6
        self.col = 3
        
        for i in reversed(range(self.grid.count())): 
            self.grid.itemAt(i).widget().setParent(None)
       
        var = False
        if start + self.display_limit < self.lim:
            cnt = self.display_limit
            var = True
            #print(start)
        else:
            cnt = self.lim-start

        self.ind = math.ceil(  cnt/self.col )
            
        lbl_count = QLabel()
        
        lbl_count.setText(str(start if cnt<1 else start+1) + " - " + str(start+cnt) + " of "  + str(self.lim) + " results." )

        self.grid.addWidget(lbl_count,0,0)


        cont_matrix = [[],[]]
        for i in range(1,self.ind+1):
            for j in range(self.col):
                if( start+ (i-1)*self.col+j< self.lim ):
                    
                    desc = QVBoxLayout()
                    dec = False
                    lbl_img = QLabel()

                    lbl_img.fileName = "image" 
                    
                    if os.path.isfile(self.final[start + (i-1)*self.col+j]["path"]):
                        pixmap = QPixmap( self.final[start + (i-1)*self.col+j]["path"])
                        pixmap1 = pixmap.scaled(200,200,Qt.KeepAspectRatio)
                        lbl_img.setPixmap(pixmap1)
                        
                    else:
                        lbl_img.setText("Image not found.")

                    
                    lbl_path = QLabel()
                    lbl_path.setText(self.final[start+(i-1)*self.col+j]["path"])
                    lbl_path.fileName = "image"

                    lbl_caption = QLabel()
                    lbl_caption.setText(self.final[start+(i-1)*self.col+j]["caption_0"])
                    lbl_caption.fileName = "image"

                    desc.addWidget(lbl_img)
                    desc.addWidget(lbl_path)
                    desc.addWidget(lbl_caption)
                    cont = QWidget()
                    cont.setLayout(desc)
                    #print(self.final[start + (i-1)*self.col+j]["cnt"])
                    self.grid.addWidget(cont, i,j)
                    

        if start >= self.display_limit:
            prev = QPushButton("< prev")
            prev.clicked.connect(lambda: self.display(self.start-self.display_limit) )
            self.grid.addWidget(prev,self.ind+1,0)

        if var:
            nxt = QPushButton("next >")
            nxt.clicked.connect(lambda: self.display(self.start+self.display_limit))
            self.grid.addWidget(nxt,self.ind+1,self.col-1)


def get_drives():
#opt = ["A:\\\\","B:\\\\","C:\\\\","D:\\\\","E:\\\\","F:\\\\","G:\\\\","H:\\\\","I:\\\\","J:\\\\","K:\\\\","L:\\\\","M:\\\\","N:\\\\","O:\\\\","P:\\\\","Q:\\\\","R:\\\\","S:\\\\","T:\\\\","U:\\\\","V:\\\\","W:\\\\","X:\\\\","Y:\\\\","Z:\\\\"]
    from ctypes import windll
    opt = ["A:/","B:/","C:/","D:/","E:/","F:/","G:/","H:/","I:/","J:/","K:/","L:/","M:/","N:","O:/","P:/","Q:/","R:/","S:/","T:/","U:/","V:/","W:/","X:/","Y:/","Z:/"]
    drives = []
    bitmask = windll.kernel32.GetLogicalDrives()
    for letter in opt:
        if bitmask & 1:
            drives.append(letter)
        bitmask >>= 1

    return drives


def find_images(que,folder_path=None): 
    q = queue.Queue()
    formats = ['.jpg','.JPG','.jpeg','.JPEG']
    opt = ["/home/","A:/","B:/","C:/","D:/","E:/","F:/","G:/","H:/","I:/","J:/","K:/","L:/","M:/","N:","O:/","P:/","Q:/","R:/","S:/","T:/","U:/","V:/","W:/","X:/","Y:/","Z:/"]
    
    if not folder_path:
        if platform.system() == 'Linux':
            q.put("/home/")
        elif platform.system() == 'Windows':
            for folder in get_drives():  
                q.put(folder)
    else:
        q.put(folder_path) 

    while not q.empty():
        fold = q.get()
        if os.path.isdir(fold) == True :
            try:
                    
                for folder in os.listdir(fold):
                    if fold in opt :
                        tmp = fold + folder
                    else:
                        tmp = fold + "/" +folder  
                    #print(tmp)
                    q.put(tmp)
            except Exception as e:
                print("error") 
        else:
            ext = ""        
            if fold[-4:-3]  == "." :
                ext = fold[-4:]
            elif fold[-5:-4] == "." :
                ext = folder[-5:]
        
            if ext in formats:
                
                que.put(fold)
    
    que.put("end")
    

if __name__ == '__main__':
    
    app = QApplication(sys.argv)
    ex = Example()
    sys.exit(app.exec_())