import os
import cv2
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Hàm tải ảnh (giữ nguyên code của bạn)
def load_images(base_path, img_size=(64, 64)):
    images = []
    labels = []
    
    categories = {'dog': 0, 'cat': 1}
    
    for category in categories:
        folder_path = os.path.join(base_path, category)
        print(f"Đang tải thư mục {category}...")
        
        for img_name in tqdm(os.listdir(folder_path)):
            img_path = os.path.join(folder_path, img_name)
            try:
                img = cv2.imread(img_path)
                if img is None:
                    print(f"Đang đọc file {img_name}... không thành công")
                    continue
                else:
                    print(f"Đang đọc file {img_name}...")
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, img_size)
                img = img / 255.0
                
                images.append(img)
                labels.append(categories[category])
            except Exception as e:
                print(f"Lỗi khi đọc file {img_name}: {e}")
    
    return np.array(images), np.array(labels)

# Đường dẫn dữ liệu
base_path = r"C:\Users\Admin\Documents\.venv\PetImages"

# Tải dữ liệu
X, y = load_images(base_path)
print(f"Đã tải {len(X)} ảnh")
print(f"Kích thước dữ liệu: {X.shape}")

# Chia dữ liệu thành tập train và test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Kích thước tập train: {X_train.shape}")
print(f"Kích thước tập test: {X_test.shape}")

# Xây dựng mô hình CNN
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # Nhị phân: 0 (chó), 1 (mèo)
])

# Compile mô hình
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Huấn luyện mô hình
history = model.fit(X_train, y_train, epochs=20, batch_size=32, 
                   validation_data=(X_test, y_test))

# Đánh giá mô hình
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Độ chính xác trên tập test: {test_accuracy:.4f}")

# Lưu mô hình
model.save('dog_cat_classifier.h5')

# Hàm dự đoán và hiển thị ảnh
def predict_image(img_path, model):
    # Đọc và tiền xử lý ảnh
    img = cv2.imread(img_path)
    if img is None:
        print("Không thể đọc ảnh")
        return
    
    # Chuyển sang RGB và giữ nguyên bản gốc để hiển thị
    img_display = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Tiền xử lý cho dự đoán
    img = cv2.resize(img_display, (64, 64))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    
    # Dự đoán
    prediction = model.predict(img)
    label = "Cat" if prediction[0] > 0.5 else "Dog"
    probability = prediction[0][0] if prediction[0] > 0.5 else 1 - prediction[0][0]
    
    # Hiển thị ảnh với nhãn và xác suất
    plt.figure(figsize=(6, 6))
    plt.imshow(img_display)
    plt.title(f"Dự đoán: {label}\nXác suất: {probability:.4f}")
    plt.axis('off')  # Tắt trục tọa độ
    plt.show()

# Thử dự đoán một ảnh và hiển thị
sample_img_path = r"C:\Users\Admin\Documents\.venv\PetImages\dog\1000.jpg"
predict_image(sample_img_path, model)

# Vẽ biểu đồ kết quả huấn luyện
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()
