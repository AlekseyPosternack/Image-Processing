import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.metrics import mean_squared_error, structural_similarity as ssim

# 1. загрузка изображения
image_path = 'sar_1_gray.jpg'
img_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if img_gray is None:
    raise FileNotFoundError(f"Изображение {image_path} не найдено!")

print("1. Image is loaded.")

# 2. гистограмма
fig = plt.figure(figsize=(18, 14))
fig.set_constrained_layout_pads(w_pad=0.5, h_pad=0.5, hspace=0.3, wspace=0.3)

plt.subplot(3, 3, 1)
plt.imshow(img_gray, cmap='gray')
plt.title('Исходное изображение')
plt.axis('off')

histSize = 256
histRange = (0, 256)
b_hist = cv2.calcHist([img_gray], [0], None, [histSize], histRange)
plt.subplot(3, 3, 2)
plt.plot(b_hist, color='black')
plt.title('Гистограмма')

print("2. Gustogram is built.")

# 3. гамма-коррекция
def gamma_correction(image, gamma):
    image_normalized = image.astype(np.float32) / 255.0
    corrected = np.power(image_normalized, gamma)
    return np.uint8(corrected * 255)

gamma_low = 0.5
gamma_static = 1.0
gamma_high = 2.0

img_gamma_low = gamma_correction(img_gray, gamma_low)
img_gamma_static = gamma_correction(img_gray, gamma_static)
img_gamma_high = gamma_correction(img_gray, gamma_high)

plt.subplot(3, 3, 4)
plt.imshow(img_gamma_low, cmap='gray')
plt.title(f'Гамма-коррекция (γ = {gamma_low})')
plt.axis('off')

plt.subplot(3, 3, 5)
plt.imshow(img_gamma_static, cmap='gray')
plt.title(f'Гамма-коррекция (γ = {gamma_static})')
plt.axis('off')

plt.subplot(3, 3, 6)
plt.imshow(img_gamma_high, cmap='gray')
plt.title(f'Гамма-коррекция (γ = {gamma_high})')
plt.axis('off')

print(f"3. gamma-correction: <{gamma_low}, {gamma_static}, {gamma_high}>.")

# 4. сравнение по MSE и SSIM
def compare_images(img1, img2, title):
    mse_val = mean_squared_error(img1, img2)
    ssim_val = ssim(img1, img2, data_range=255)
    print(f"{title} — MSE: {mse_val:.2f}, SSIM: {ssim_val:.4f}")
    return mse_val, ssim_val

compare_images(img_gray, img_gamma_low, "Исходное и γ = 0.5")
compare_images(img_gray, img_gamma_high, "Исходное и γ = 2.0")

print("4. Compare is done.")

# 5. статическая светокоррекция
eq_gray = cv2.equalizeHist(img_gray)

plt.subplot(3, 3, 3)
plt.imshow(eq_gray, cmap='gray')
plt.title('Статическая цветокоррекция')
plt.axis('off')

compare_images(img_gray, eq_gray, "Исходное и поcле цветокоррекции")

print("5. Static color-correction is done.")

# 6. пороговая фильтрация
thresholds = [50, 127, 200]

print ("6.")

for i, thresh in enumerate(thresholds):
    _, img_thresh = cv2.threshold(img_gray, thresh, 255, cv2.THRESH_BINARY)
    plt.subplot(3, 3, 7 + i)
    plt.imshow(img_thresh, cmap='gray')
    plt.title(f'Порог = {thresh}')
    plt.axis('off')
    print(f"Filtration with thresh {thresh} is done.")

print("All done!")

output_path = "result.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.8) # сохраняем результаты в .png (для более смотрибельного результата)
plt.tight_layout()
plt.show() # текст накладывается (не придумал, как фиксить, лучше смотреть в result.png)