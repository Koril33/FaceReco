import dlib          # 人脸处理的库 Dlib
import numpy as np   # 数据处理的库 numpy
import cv2           # 图像处理的库 OpenCv
import pandas as pd  # 数据处理的库 Pandas


face_recognition = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')

def return_euclidean_distance(feature_1, feature_2):
	feature_1 = np.array(feature_1)
	feature_2 = np.array(feature_2)
	dist = np.sqrt(np.sum(np.square(feature_1 - feature_2)))
	return dist

path_csv = "features_all_person.csv"
csv_read = pd.read_csv(path_csv, header = None, error_bad_lines = False)
features_known_arr = []
names_dict = {1 : 'Ding Jinghui', 2 : 'Liang Zhiqin', 3 : 'Lee Chong Wei', 4 : 'Lin Dan'}

for i in range(csv_read.shape[0]):
	feature_someone_arr = []
	for j in range(0, len(csv_read.iloc[i, :])):
		feature_someone_arr.append(csv_read.iloc[i, :][j])
	features_known_arr.append(feature_someone_arr)
print("faces in database:", len(features_known_arr))

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

capture = cv2.VideoCapture(0)

capture.set(3, 480)

while capture.isOpened():
	flag, img = capture.read()
	k = cv2.waitKey(1)

	img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

	faces = detector(img_gray, 0)

	font = cv2.FONT_HERSHEY_COMPLEX

	pos_namelist = []
	name_namelist = []

	if k == ord('q'):
		break
	else:
		if len(faces) != 0:
			features_cap_arr = []
			for i in range(len(faces)):
				shape = predictor(img, faces[i])
				features_cap_arr.append(face_recognition.compute_face_descriptor(img, shape))

			for j in range(len(faces)):
				print("#### camera person", j + 1, "####")
				name_namelist.append("unknown")

				pos_namelist.append(tuple([faces[j].left(), int(faces[j].bottom() + (faces[j].bottom() - faces[j].top())/4)]))

				e_distance_list = []
				for i in range(len(features_known_arr)):
					if str(features_known_arr[i][0]) != '0.0':
						print("with person", str(i + 1), "the e distance: ", end = '')
						e_distance_tmp = return_euclidean_distance(features_cap_arr[j], features_known_arr[i])
						print(e_distance_tmp)
						e_distance_list.append(e_distance_tmp)
					else:
						e_distance_list.append(9999999)

				similar_person_num = e_distance_list.index(min(e_distance_list))
				print("Minimum e distance with person", int(similar_person_num) + 1)

				if min(e_distance_list) < 0.4:
					name_namelist[j] = "Person " + names_dict[int(similar_person_num) + 1]
					print("May be person " + str(int(similar_person_num) + 1))

				else:
					print("Unknown person")

				for i, element in enumerate(faces):
					x1, y1, x2, y2, w, h = element.left(), element.top(), element.right() + 1, element.bottom() + 1, element.width(), element.height()
					# 矩形框为白色
					color_rectangle = (255, 255, 255)
					cv2.rectangle(img, (x1, y1), (x2, y2), color_rectangle, 2)
				print('\n')

			for i in range(len(faces)):
				cv2.putText(img, name_namelist[i], pos_namelist[i], font, 0.8, (0, 255, 255), 1, cv2.LINE_AA)

	print("Faces in camera now:", name_namelist, "\n")

	cv2.putText(img, "Press 'q': Quit", (20, 450), font, 0.8, (84, 255, 159), 1, cv2.LINE_AA)
	cv2.putText(img, "Face Recognition", (20, 40), font, 1, (0, 0, 0), 1, cv2.LINE_AA)
	cv2.putText(img, "Faces: " + str(len(faces)), (20, 100), font, 1, (0, 0, 255), 1, cv2.LINE_AA)

	# 窗口显示 show with opencv
	cv2.imshow("camera", img)


capture.release()

cv2.destroyAllWindows()