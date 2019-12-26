import csv
import numpy as np 
import keras 
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D,Flatten,Dense,Activation,Dropout,MaxPooling2D
from keras.activations import relu
from keras.optimizers import Adam

#26x34x1 이미지 사용(1 흑백 이미지용)
height = 26
width = 34
dims = 1

def readCsv(path):

	with open(path,'r') as f:
		#csv파일 읽어오기 
		reader = csv.DictReader(f)
		rows = list(reader)

	# 이미지와 모든 사진의 태그를 저장할 두개의 빈 numpy 배열 생성
	#img는 모든 영상을 포함하는 numpy배열
	#tgs는 영상의 태그가 있는 numpy배열
	imgs = np.empty((len(list(rows)),height,width, dims),dtype=np.uint8)
	tgs = np.empty((len(list(rows)),1))

	for row,i in zip(rows,range(len(rows))):

		#리스트를 이미지 형식으로 변환 후 배열에 넣음
		img = row['image']
		img = img.strip('[').strip(']').split(', ')
		im = np.array(img,dtype=np.uint8)
		im = im.reshape((26,34))
		im = np.expand_dims(im, axis=2)
		imgs[i] = im

		#태그가 open이면 1, close이면 0로 배열에 넣음
		tag = row['state']
		if tag == 'open':
			tgs[i] = 1
		else:
			tgs[i] = 0

	#dataset배열을 섞어서 반환
	index = np.random.permutation(imgs.shape[0])
	imgs = imgs[index]
	tgs = tgs[index]

	return imgs,tgs

#CNN 만들기
def makeModel():
	model = Sequential()

	#relu activation이 있는 3개의 필터가 있으며, 각 필터는 maxpoolinglayer를 따른다.
	#영상의 작은 변화나 움직임이 특징을 추출할 때 크게 영향을 미치지 않도록 해준다.
	model.add(Conv2D(32, (3,3), padding = 'same', input_shape=(height,width,dims)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2,2)))
	model.add(Conv2D(64, (2,2), padding= 'same'))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Conv2D(128, (2,2), padding='same'))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))

	#dropout : nn에서 overfitting을 방지하는 방법. 시간은 오래걸리지만 정확하게 훈련 가능
	#relu activation이 있는 2개의 dropout layer를 추가
	#마지막으로 이항분류기에 sigmoid activation이 있는 뉴런을 추가
	model.add(Flatten())
	model.add(Dense(512))
	model.add(Activation('relu'))
	model.add(Dense(512))
	model.add(Activation('relu'))
	model.add(Dense(1))
	model.add(Activation('sigmoid'))

	#optimizer로 아담을 사용
	#loss로 binary_crossentropy를 사용
	model.compile(optimizer=Adam(lr=0.001), loss='binary_crossentropy',metrics=['accuracy'])

	return model

def main():

	xTrain ,yTrain = readCsv('dataset.csv')
	print (xTrain.shape[0])
	#0과 1 사이로 이미지 값 조정 - 학습과정을 더 빠르게 만듬
	xTrain = xTrain.astype('float32')
	xTrain /= 255

	model = makeModel()

	#데이터 증가를 통해 훈련 예시들의 수를 인위적으로 늘림
	#dataset이 작고 overfitting을 줄여야 하기 때문
	datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.2,
        height_shift_range=0.2,
        )
	datagen.fit(xTrain)

	#모델 훈련
	#batch_size : 몇 개의 샘플로 가중치를 갱신할 것인지 지정
	#epochs : 학습 반복 횟수
	model.fit_generator(datagen.flow(xTrain,yTrain,batch_size=32),
						steps_per_epoch=len(xTrain) / 32, epochs=50)

	#모델 저장
	model.save('blinkModel.hdf5')

if __name__ == '__main__':
	main()
