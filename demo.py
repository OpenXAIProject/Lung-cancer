#Copyright 2018 (Institution) under XAI Project supported by Ministry of Science and ICT, Korea

#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at

#   https://www.apache.org/licenses/LICENSE-2.0

#Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PIL.ImageQt import ImageQt
import os
import time
import tempfile
import numpy as np
try:
    _fromUtf8 = QStringListModel.fromUtf8
except AttributeError:
    def _fromUtf8(s):
        return s

try:
    _encoding = QApplication.UnicodeUTF8
    def _translate(context, text, disambig):
        return QApplication.translate(context, text, disambig, _encoding)
except AttributeError:
    def _translate(context, text, disambig):
        return QApplication.translate(context, text, disambig)

class Ui_Form(QWidget):
    def setupUi(self, Form):
        Form.setObjectName(_fromUtf8("Form"))
        Form.resize(813, 667)
        self.num =0;
        Form.setLayoutDirection(Qt.RightToLeft)
        self.label = QLabel(Form)
        self.label.setGeometry(QRect(70, 30, 681, 41))
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setObjectName(_fromUtf8("label"))
        self.pushButton = QPushButton(Form)
        self.pushButton.setGeometry(QRect(70, 100, 271, 41))
        self.pushButton.setObjectName(_fromUtf8("pushButton"))
        self.pushButton_2 = QPushButton(Form)
        self.pushButton_2.setGeometry(QRect(470, 100, 271, 41))
        self.pushButton_2.setObjectName(_fromUtf8("pushButton_2"))
        self.label_2 = QLabel(Form)
        self.label_2.setGeometry(QRect(70, 170, 281, 281))
        self.label_2.setAlignment(Qt.AlignCenter)
        self.label_2.setObjectName(_fromUtf8("label_2"))
        self.label_2.setSizePolicy(QSizePolicy.Expanding,QSizePolicy.Expanding)
        self.label_3 = QLabel(Form)
        self.label_3.setGeometry(QRect(460, 170, 281, 281))
        self.label_3.setAlignment(Qt.AlignCenter)
        self.label_3.setObjectName(_fromUtf8("label_3"))
        self.TextEdit = QTextEdit(Form)
        self.TextEdit.setGeometry(QRect(70, 490, 671, 121))
        self.TextEdit.setObjectName(_fromUtf8("TextEdit"))

        self.retranslateUi(Form)
        QMetaObject.connectSlotsByName(Form)

        self.pushButton.clicked.connect(self.getfile)

        self.pushButton_2.clicked.connect(self.lrp)
    def retranslateUi(self, Form):
        Form.setWindowTitle(_translate("Form", "데모 프로그램", None))
        self.label.setText(_translate("Form", "의료 폐 CT 영상 분류 및 해석", None))
        self.pushButton.setText(_translate("Form", "CT 영상 불러오기", None))
        self.pushButton_2.setText(_translate("Form", "분류 및 해석", None))
        self.label_2.setText(_translate("Form", "입력 영상", None))
        self.label_3.setText(_translate("Form", "결과 영상", None))

    def getfile(self):
        self.label_3.clear()
        self.TextEdit.clear()
        os.popen(
            "python ./load_image.py 0")
        self.fname = './Demo_img/img'+str(self.num)+'.gif'

        self.mov = QMovie(self.fname)
        self.mov.setSpeed(50)
        self.label_2.setMovie(self.mov)
        self.mov.start()
        print self.fname
        a = str.split(str(self.fname), '/')
        self.f_name = a[len(a) - 1]

        print self.f_name
        self.TextEdit.append("CT 영상으로부터 입력 영상 추출")
    def lrp(self):
        t = time.time()
        a = os.popen(
            "python ./load_image.py 1",'w')
        self.rname = './Demo_img/heatmap' + str(self.num) + '.gif'
        self.mov2 = QMovie(self.rname)
        self.label_3.setMovie(self.mov2)
        self.mov.stop()
        self.mov.setSpeed(50)
        self.mov.start()
        self.mov2.setSpeed(50)
        self.mov2.start()
        self.prob = np.load('./Demo_img/soft.npy')
        self.TextEdit.append(str(self.prob[self.num][1])+"% 확률로 종양으로 분류")
        self.num = self.num + 1
        if self.num == 5:
            self.num = 0
if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    Form = QWidget()
    ui = Ui_Form()
    ui.setupUi(Form)
    Form.show()
    sys.exit(app.exec_())

