from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import mnist


(x_train, y_train), (x_test, y_test) = mnist.load_data(path="mnist.npz")


x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train, test_size=0.3, random_state=777)

num_x_train = x_train.shape[0]
num_x_val = x_val.shape[0]
num_x_test = x_test.shape[0]

# 모델의 입력 전에 전처리 과정
x_train = (x_train.reshape(-1, 28, 28, 1))/255
x_val = (x_val.reshape(-1, 28, 28, 1))/255
x_test = (x_test.reshape(-1, 28, 28, 1))/255


# 각 데이터의 레이블을 범주형 형태로 변경

y_train = to_categorical(y_train)
y_val = to_categorical(y_val)
y_test = to_categorical(y_test)


# 함수형 API 형태로 모델 구성 및 학습
# 함수형 API는 Input() 을 통해 입력값의 형태를 정의해야 합니다.
inputs = Input(shape=(28, 28, 1))
x = Conv2D(32, (3, 3), activation='relu')(inputs)
x = Conv2D(32, (3, 3), activation='relu')(x)
x = MaxPooling2D(strides=2)(x)
x = GlobalAveragePooling2D()(x)
x = Dense(10, activation='softmax')(x)

# 위에서 정의한 층을 포함하고 있는 모델을 생성
model = Model(inputs=inputs, outputs=x)
model.compile(optimizer='adam',
              loss='categorical_crossentropy', metrics=['acc'])

model.fit(x_train, y_train, batch_size=32,
          validation_data=(x_val, y_val), epochs=10)
