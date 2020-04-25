import cv2
import os
import dlib
import csv
import numpy as np
from skimage import io

path_photos = 'photo_from_camera/'

# Dlib的人脸检测器
detector = dlib.get_frontal_face_detector()

# Dlib的68点特征预测器
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# Dlib训练好的模型
face_recognition = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')

# 返回每一张图片的128个数值列表
def return_128d_features(path_img):
	img_read = io.imread(path_img)
	img_gray = cv2.cvtColor(img_read, cv2.COLOR_BGR2RGB)
	faces = detector(img_gray, 1)

	print("检测到人脸的图像: " + path_img + '\n')

	if len(faces) != 0:
		shape = predictor(img_gray, faces[0])
		face_descriptor = face_recognition.compute_face_descriptor(img_gray, shape)
	else:
		face_descriptor = 0
		print("no face")

	return face_descriptor

# 将所有图片的128个数值组成一个二维数组，并求出均值
def return_features_person_x(path_faces_person_x):
	features_person_x_list = []
	photos_list = os.listdir(path_faces_person_x)

	if photos_list:
		for i in range(len(photos_list)):
			print("正在读的人脸图像: " + path_faces_person_x + "/" + photos_list[i])
			features_128d = return_128d_features(path_faces_person_x + '/' +photos_list[i])

			if features_128d == 0:
				i += 1
			else:
				features_person_x_list.append(features_128d)
				print("一维数组")
				print(list(features_128d))
				print("长度：" + str(len(features_128d)))
	else:
		print("文件夹内图像文件为空" + path_faces_person_x + '/', '\n')

	if features_person_x_list:
		print("二维数组")
		print(features_person_x_list)
		features_person_x_mean = np.array(features_person_x_list).mean(axis = 0)
	else:
		features_person_x_mean = '0'

	return features_person_x_mean

# person_list存储每个人的文件夹
person_list = os.listdir('photo_from_camera/')

# 获得每个人的编号
person_num_list = []

for person in person_list:
	person_num_list.append(int(person[-1]))
person_count = max(person_num_list)

# 将所有数据录入csv文件
with open('features_all_person.csv', 'w', newline="") as cvsfile:
	writer = csv.writer(cvsfile)
	for person in range(person_count):
		print("正在读取：" + path_photos + "person_" + str(person + 1))
		features_person_x_mean = return_features_person_x(path_photos + "person_" + str(person + 1))
		writer.writerow(features_person_x_mean)
		print("特征值：", list(features_person_x_mean))
		print('\n')
	print("所有人脸数据存入： features_all_person.csv")