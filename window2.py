import tensorflow as tf
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
import sys
import cv2
from PIL import Image
import numpy as np


class MainWindow(QTabWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('基于深度学习的花卉识别系统')
        self.model = tf.keras.models.load_model("models/mobilenet_flower.h5")
        self.to_predict_name = "images/target.png"
        self.class_names = ['雏菊', '蒲公英', '玫瑰', '向日葵', '郁金香']
        self.resize(900, 700)
        self.initUI()

    def initUI(self):
        main_widget = QWidget()
        main_layout = QHBoxLayout()
        font = QFont('宋体', 10)

        left_widget = QWidget()
        left_layout = QVBoxLayout()
        img_title = QLabel("检测图片")
        img_title.setFont(font)
        img_title.setAlignment(Qt.AlignCenter)
        self.img_label = QLabel()
        img_init = cv2.imread(self.to_predict_name)
        img_init = cv2.resize(img_init, (224, 224))
        cv2.imwrite('images/target.png', img_init)
        self.img_label.setPixmap(QPixmap('images/target.png'))
        left_layout.addWidget(img_title)
        left_layout.addWidget(self.img_label, 1, Qt.AlignCenter)
        left_widget.setLayout(left_layout)

        right_widget = QWidget()
        right_layout = QVBoxLayout()
        btn_change = QPushButton(" 选择图片 ")
        btn_change.clicked.connect(self.change_img)
        btn_change.setFont(font)
        btn_predict = QPushButton(" 开始识别")
        btn_predict.setFont(font)
        btn_predict.clicked.connect(self.predict_img)

        self.result = QLabel("待识别....")
        self.result.setFont(QFont('宋体', 10))
        right_layout.addStretch()
        right_layout.addStretch()
        right_layout.addWidget(self.result, 0, Qt.AlignCenter)
        right_layout.addStretch()
        right_layout.addWidget(btn_change)
        right_layout.addWidget(btn_predict)
        right_layout.addStretch()
        # right_layout.addSpacing(5)
        right_widget.setLayout(right_layout)

        # 关于页面，设置组件并把组件放在布局上
        about_widget = QWidget()
        about_layout = QVBoxLayout()
        about_title = QLabel('浪漫花屋')
        about_title.setFont(QFont('楷体', 18))
        about_title.setAlignment(Qt.AlignCenter)
        about_img = QLabel()
        about_img.setPixmap(QPixmap('images/3.jpg'))
        about_img.setAlignment(Qt.AlignCenter)
        label_super = QLabel("zjx")
        label_super.setFont(QFont('楷体', 12))
        # label_super.setOpenExternalLinks(True)
        label_super.setAlignment(Qt.AlignRight)
        about_layout.addWidget(about_title)
        about_layout.addStretch()
        about_layout.addWidget(about_img)
        about_layout.addStretch()
        about_layout.addWidget(label_super)
        about_widget.setLayout(about_layout)

        # 添加注释
        self.addTab(main_widget, '主页')
        self.addTab(about_widget, '关于')
        self.setTabIcon(0, QIcon('images/主页面.png'))
        self.setTabIcon(1, QIcon('images/关于.png'))

        label_super = QLabel()
        label_super.setFont(QFont('楷体', 12))
        label_super.setOpenExternalLinks(True)
        label_super.setAlignment(Qt.AlignRight)
        main_layout.addWidget(left_widget)
        main_layout.addWidget(right_widget)
        main_widget.setObjectName("color")
        main_widget.setStyleSheet("#color{background-color:pink}")
        main_widget.setLayout(main_layout)

        self.addTab(main_widget, '主页面')

    def change_img(self):
        openfile_name = QFileDialog.getOpenFileName(self, '选择文件', '', 'Image files(*.jpg , *.png)')
        img_name = openfile_name[0]
        if img_name == '':
            pass
        else:
            self.to_predict_name = img_name
            img_init = cv2.imread(self.to_predict_name)
            img_init = cv2.resize(img_init, (224, 224))
            cv2.imwrite('images/target.png', img_init)
            self.img_label.setPixmap(QPixmap('images/target.png'))

    def predict_img(self):
        img = Image.open('images/target.png')
        img = np.asarray(img)
        # gray_img = img.convert('L')
        # img_torch = self.transform(gray_img)
        outputs = self.model.predict(img.reshape(1, 224, 224, 3))
        # print(outputs)
        result_index = np.argmax(outputs)
        # print(result_index)
        result = self.class_names[result_index]
        self.result.setText(result)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    x = MainWindow()
    x.show()
    sys.exit(app.exec_())



