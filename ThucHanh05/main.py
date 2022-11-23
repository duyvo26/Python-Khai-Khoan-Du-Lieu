# Khai báo thư viện
import os
import sys

from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.layers import Dense, BatchNormalization
from keras.optimizers import Adamax
from keras.preprocessing.image import ImageDataGenerator

# Khai báo các tham số
input_dir = ''
train_dir = input_dir + "train/"
test_dir = input_dir + "test/"
img_dims = 224
batch_size = 64

# Tăng cường dữ liệu
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
test_datagen = ImageDataGenerator(rescale=1. / 255)
# Đọc dữ liệu ảnh
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_dims, img_dims),
    batch_size=20,
    class_mode='categorical')
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_dims, img_dims),
    batch_size=20,
    class_mode='categorical')


# Định nghĩa kiến trúc của mô hình
def build_model(base_model, lr=1e-5, nb_classes=1):
    from keras import layers
    from keras.models import Sequential
    for layer in base_model.layers[:1]:
        layer.trainable = False
    model = Sequential()
    model.add(base_model)
    model.add(BatchNormalization())
    model.add(layers.GlobalAveragePooling2D())
    model.add(Dense(nb_classes, activation="sigmoid"))
    model.compile(
        loss='categorical_crossentropy', optimizer=Adamax(lr=lr),
        metrics=['acc']
    )
    print(model.summary())
    return model


# Khai báo mô hình
# # 16
# from keras.applications.vgg16 import VGG16
# model_name = "pretrain_VGG16.h5"
# base_model = VGG16(weights='imagenet', include_top=False,
#                    input_shape=(img_dims, img_dims, 3))


# 19
from keras.applications.vgg19 import VGG19

model_name = "D:\\GIT\\Python\\Python-Khai-Khoan-Du-Lieu\\ThucHanh05\\pretrain_VGG19.h5"
base_model = VGG19(weights='imagenet', include_top=False, input_shape=(img_dims, img_dims, 3))


# # 121
# from keras.applications.densenet import DenseNet121
# model_name = "pretrain_DenseNet121.h5"
# base_model = DenseNet121(weights='imagenet', include_top=False,
# input_shape=(img_dims, img_dims, 3))


# Các thông số huấn luyện
checkpoint = ModelCheckpoint(filepath=model_name, save_best_only=True,
                             save_weights_only=False)
lr_reduce = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=2, verbose=2,
                              mode='max')
early_stop = EarlyStopping(monitor='val_loss', min_delta=0.1, patience=1, mode='min')

model = build_model(base_model=base_model, lr=0.0001, nb_classes=train_generator.num_classes)

# Huấn luyện mô hình
model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=10,
    validation_data=test_generator,
    validation_steps=10,
    verbose=1,
    callbacks=[checkpoint, lr_reduce]
)

# Load mô hình tốt nhất và hiển thị kết quả
from keras.models import load_model

model = load_model(model_name)
scores = model.evaluate(train_generator, batch_size=batch_size,
                        steps=test_generator.samples // batch_size)
print("%s%s: %.2f%%" % ("evaluate_generator ", model.metrics_names[1],
                        scores[1] * 100))
new_name_model = model_name + "." + str(round((scores[1] * 100), 2)) + ".tf"
new_name_log = model_name + "." + str(round((scores[1] * 100), 2)) + ".log"
os.rename(model_name, new_name_model)
