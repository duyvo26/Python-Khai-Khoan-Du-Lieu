import os
from keras import Model, Input
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.layers import Dense, BatchNormalization, Dropout, Conv2D, MaxPool2D, SeparableConv2D, Flatten
from keras.optimizers import Adamax
from keras.preprocessing.image import ImageDataGenerator

def DuongDan():
    import pathlib
    input_dir = "./"
    return input_dir


# Khai báo các tham số
train_dir = DuongDan() + "train"
test_dir = DuongDan() + "test"
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

# Định nghĩa kiến trúc mạng
def build_model(lr=1e-5, nb_class=1):
    # Input layer
    inputs = Input(shape=(img_dims, img_dims, 3))
    # First conv block
    x = Conv2D(filters=16, kernel_size=(3, 3), activation='relu', padding='same')(inputs)
    x = Conv2D(filters=16, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = MaxPool2D(pool_size=(2, 2))(x)
    # Second conv block
    x = SeparableConv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = SeparableConv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPool2D(pool_size=(2, 2))(x)
    # Third conv block
    x = SeparableConv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = SeparableConv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPool2D(pool_size=(2, 2))(x)
    # Fourth conv block
    x = SeparableConv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = SeparableConv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPool2D(pool_size=(2, 2))(x)
    x = Dropout(rate=0.2)(x)
    # Fifth conv block
    x = SeparableConv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = SeparableConv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPool2D(pool_size=(2, 2))(x)
    x = Dropout(rate=0.2)(x)
    # FC layer
    x = Flatten()(x)
    # Output layer
    output = Dense(units=nb_class, activation='softmax')(x)
    # Creating model and compiling
    model = Model(inputs=inputs, outputs=output)
    model.compile(optimizer=Adamax(lr=lr), loss="categorical_crossentropy", metrics=['accuracy'])
    return model


# Khai báo mô hình
model_name = DuongDan() + "/dcnn.hdf5"
nb_classes = train_generator.num_classes
model = build_model(lr=0.0001, nb_class=nb_classes)
# Các thông số huấn luyện
checkpoint = ModelCheckpoint(filepath=model_name, save_best_only=True, save_weights_only=False)
lr_reduce = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=2, verbose=2, mode='max')
early_stop = EarlyStopping(monitor='val_loss', min_delta=0.1, patience=1, mode='min')
# Huấn luyện mô hình
model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=300,
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
new_name_model = model_name + "__" + str(round((scores[1] * 100), 2)) + ".hdf5"
os.rename(model_name, new_name_model)
