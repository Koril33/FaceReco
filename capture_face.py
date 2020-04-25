import dlib          # 人脸处理的库
import numpy as np   # 数据处理的库
import cv2           # 图像处理的库
import os            # 读写文件

# Dlib的人脸检测器
detector = dlib.get_frontal_face_detector()

# Dlib的68点特征预测器
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# 调用并设置摄像头
capture = cv2.VideoCapture(0)
capture.set(3, 480)

# 存储拍照的数量
counter_for_save_photos = 0

# 存放所有人照片的路径
path_photos = 'photo_from_camera/'

# 判断这个路径（文件夹）是否存在的函数
def check_folder_exists(path):
	global counter_for_save_photos
	if os.path.isdir(path):
		pass
	else:
		counter_for_save_photos = 0
		os.mkdir(path)

# 判断'photo_from_camera/'是否存在，不存在就创建一个
check_folder_exists(path_photos)

# 统计已经存在几个用户了，将以一个用户号码存储在person_count中
if os.listdir(path_photos):
	person_list = os.listdir(path_photos)
	person_num_list = []
	for person in person_list:
		person_num_list.append(int(person[-1]))
	person_count = max(person_num_list)
else:
	person_count = 0

# sava_flag用来判断当前能否存储照片
save_flag = 1

# press_n_flag判断用户有没有按下n来创建新的用户照片文件夹
press_n_flag = 0

# 开始检测
while capture.isOpened():
	# 读取摄像头数据
	flag, img = capture.read()
	# 按键判断
	k = cv2.waitKey(1)

	img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

	# 返回人脸的对象
	faces = detector(img, 0)

	# 选择字体
	font = cv2.FONT_HERSHEY_COMPLEX

	# 按下n键创建新的用户照片文件夹
	if k == ord('n'):
		person_count += 1
		current_face_dir = path_photos + "person_" + str(person_count)

		os.makedirs(current_face_dir)
		print('\n')
		print("新建的人脸照片文件夹:", current_face_dir)

		counter_for_save_photos = 0
		press_n_flag = 1

	# 如果人脸只有一个，就用矩形框将面部框起来
	if len(faces) == 1:
		for i, element in enumerate(faces):
			x1, y1, x2, y2, w, h = element.left(), element.top(), element.right() + 1, element.bottom() + 1, element.width(), element.height()
			# 矩形框为白色
			color_rectangle = (255, 255, 255)

			# 判断脸部矩形框有没有超过边界
			if ((x2 + w / 2) > 640 or (y2 + h / 2 > 480) or (x1 - w / 2 < 0) or (y1 - h / 2 < 0)):
				# 超过边界则显示out of range，并将矩形框颜色改为红色
				cv2.putText(img, "Out of range", (20, 300), font, 0.8, (0, 0, 255), 1, cv2.LINE_AA)
				color_rectangle = (0, 0, 255)
				save_flag = 0

				# 在超过的条件下，按save，提醒调整位置
				if k == ord('s'):
					print("请调整位置")
			else:
				# 没有超过边界的话，并且当前条件是只有一张人脸，判定为可以拍照，矩形框为白色
				color_rectangle = (255, 255, 255)
				save_flag = 1

			# 绘制矩形框
			cv2.rectangle(img, (x1 - int(w / 2), y1 - int(h / 2)), (x2 + int(w / 2), y2 + int(h / 2)), color_rectangle, 2)

			# 新建一个空白图像，存储人脸区域
			img_blank = np.zeros((int(h * 2), w * 2, 3), np.uint8)

			# 如果满足可以拍照的条件，即save_flag为1并且用户按下s，进入下一个判定条件
			if save_flag and k == ord('s'):
				# 如果用户已经创建了新的文件夹，就可以存储照片了
				if press_n_flag:
					# 每存一张照片成功，照片计数器加一
					counter_for_save_photos += 1
					for i in range(h * 2):
						for j in range(w * 2):
							img_blank[i][j] = img[y1 - int(h / 2) + i][x1 - int(w / 2) + j]

					# 这里的check主要是避免用户创建了文件夹，存了几张照片后，又将此文件夹销毁的情况
					check_folder_exists(current_face_dir)
					# 如果照片累加器大于0，就存放照片
					if(counter_for_save_photos > 0):
						path_img_face = current_face_dir + "/img_face_" + str(counter_for_save_photos) + ".jpg"
						cv2.imwrite('{}'.format(path_img_face), img_blank)
						print("img_face_" + str(counter_for_save_photos) + ".jpg""写入文件夹：" + current_face_dir)
				else:
					print("请按s前先按n新建人脸文件夹")
	# 如果摄像头没有检测到人脸，用cv2.putText输出警告信息"No face detected"
	elif len(faces) == 0:
		if k == ord('s'):
			print("未检测到人脸")
		cv2.putText(img, "No face detected", (20, 300), font, 0.8, (0, 0, 255), 1, cv2.LINE_AA)

	# 如果摄像头检测到不止一张人脸，用cv2.putText输出警告信息"Only one can be detected"
	elif len(faces) > 1:
		if k == ord('s'):
			print("人脸数量大于1")
		cv2.putText(img, "Only one can be detected", (20, 300), font, 0.8, (0, 0, 255), 1, cv2.LINE_AA)

	# 显示检测到人脸的数量
	cv2.putText(img, "faces: " + str(len(faces)), (20, 100), font, 0.8, (255, 0, 0), 1, cv2.LINE_AA)
	
	# 显示已经拍了几张照片了
	cv2.putText(img, str(counter_for_save_photos) + " pictures have been taken", (250, 40), font, 0.8, (255, 0, 0), 1, cv2.LINE_AA)
	
	# 标题以及说明文字
	cv2.putText(img, "Face Register", (10, 40), font, 1, (0, 0, 0), 1, cv2.LINE_AA)
	cv2.putText(img, "N: New face folder", (20, 350), font, 0.8, (0, 0, 0), 1, cv2.LINE_AA)
	cv2.putText(img, "S: Save current face", (20, 400), font, 0.8, (0, 0, 0), 1, cv2.LINE_AA)
	cv2.putText(img, "Q: Quit", (20, 450), font, 0.8, (0, 0, 0), 1, cv2.LINE_AA)

	# 如果按下q，就退出
	if k == ord('q'):
		break

	# 显示图像
	cv2.imshow('camera', img)

# 释放摄像头
capture.release()

# 销毁窗口
cv2.destroyAllWindows()