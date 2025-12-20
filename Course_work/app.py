from flask import Flask, request, jsonify
import os
from PIL import Image
import numpy as np
import io
import base64
from traditional_model import TraditionalClassifier
from simple_cnn import SimpleCNNClassifier
from resnet_model import ResNetClassifier
from config import Config

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

traditional = TraditionalClassifier()
simple_cnn = SimpleCNNClassifier()
resnet = ResNetClassifier()

def init_models():
    print("Загрузка моделей...")
    try:
        traditional.load_model()
        print("Traditional model loaded")
    except:
        print("Traditional model not available")
    
    try:
        simple_cnn.load_model()
        print("Simple CNN model loaded")
    except:
        print("Simple CNN model not available")
    
    try:
        resnet.load_model()
        print("ResNet model loaded")
    except:
        print("ResNet model not available")

HTML = '''
<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>EuroSAT | Классификатор спутниковых снимков</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: Arial, sans-serif;
            background-color: #f5f5f5;
            color: #333;
            line-height: 1.6;
        }

        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }

        .header {
            text-align: center;
            margin-bottom: 40px;
            padding: 20px;
            background-color: white;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }

        .header h1 {
            color: #2c5282;
            margin-bottom: 10px;
        }

        .header p {
            color: #666;
        }

        .main-content {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-bottom: 20px;
        }

        @media (max-width: 768px) {
            .main-content {
                grid-template-columns: 1fr;
            }
        }

        .upload-panel, .preview-panel {
            background: white;
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }

        .upload-title, .preview-title {
            font-size: 18px;
            font-weight: bold;
            margin-bottom: 15px;
            color: #2c5282;
        }

        .upload-description {
            color: #666;
            margin-bottom: 15px;
        }

        .upload-zone {
            border: 2px dashed #ccc;
            border-radius: 8px;
            padding: 40px 20px;
            text-align: center;
            cursor: pointer;
            background-color: #fafafa;
            margin-bottom: 15px;
        }

        .upload-zone:hover {
            border-color: #2c5282;
            background-color: #f0f7ff;
        }

        .upload-zone.dragover {
            border-color: #2c5282;
            background-color: #e6f7ff;
        }

        .btn {
            background: #2c5282;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 14px;
            margin-top: 10px;
        }

        .btn:hover {
            background: #1a365d;
        }

        .file-input {
            display: none;
        }

        .preview-image-container {
            border: 1px solid #ddd;
            border-radius: 8px;
            padding: 20px;
            text-align: center;
            background-color: #fafafa;
            min-height: 200px;
            display: flex;
            align-items: center;
            justify-content: center;
        }

        .preview-image {
            max-width: 100%;
            max-height: 300px;
            display: none;
        }

        .preview-placeholder {
            color: #999;
        }

        .results-panel {
            background: white;
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }

        .results-title {
            font-size: 18px;
            font-weight: bold;
            margin-bottom: 20px;
            color: #2c5282;
        }

        .models-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 15px;
        }

        .model-card {
            background: #f8f9fa;
            border: 1px solid #ddd;
            border-radius: 6px;
            padding: 15px;
        }

        .model-card.winner {
            border-color: #38a169;
            background-color: #f0fff4;
        }

        .model-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 10px;
        }

        .model-name {
            font-weight: bold;
            color: #333;
        }

        .winner-badge {
            background: #38a169;
            color: white;
            padding: 2px 8px;
            border-radius: 3px;
            font-size: 12px;
        }

        .prediction-class {
            font-size: 16px;
            font-weight: bold;
            color: #333;
            margin: 10px 0;
        }

        .confidence-bar {
            height: 6px;
            background: #e2e8f0;
            border-radius: 3px;
            margin: 10px 0;
            overflow: hidden;
        }

        .confidence-fill {
            height: 100%;
            background: #2c5282;
            width: 0%;
        }

        .confidence-value {
            font-size: 14px;
            color: #666;
        }

        .confidence-percent {
            font-weight: bold;
            color: #2c5282;
        }

        .loading {
            display: none;
            text-align: center;
            padding: 20px;
        }

        .loading-spinner {
            width: 30px;
            height: 30px;
            border: 3px solid #f3f3f3;
            border-top: 3px solid #2c5282;
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin: 0 auto 10px;
        }

        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }

        .footer {
            text-align: center;
            color: #666;
            font-size: 14px;
            padding: 20px;
            border-top: 1px solid #eee;
        }

        .empty-results {
            text-align: center;
            color: #666;
            padding: 40px 20px;
        }

        .error-message {
            background: #fff5f5;
            border: 1px solid #fed7d7;
            color: #c53030;
            padding: 15px;
            border-radius: 6px;
            margin: 10px 0;
            display: none;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Классификатор спутниковых снимков EuroSAT</h1>
            <p>Анализ изображений земной поверхности с использованием трех моделей машинного обучения</p>
        </div>

        <div class="main-content">
            <!-- Upload Panel -->
            <div class="upload-panel">
                <h2 class="upload-title">Загрузите спутниковый снимок</h2>
                <p class="upload-description">Загрузите изображение в формате JPG, PNG или JPEG для анализа</p>
                
                <div class="upload-zone" id="uploadZone">
                    <div style="margin-bottom: 10px;">Перетащите изображение сюда</div>
                    <div style="color: #999; margin-bottom: 10px;">или</div>
                    <button class="btn" id="selectFileBtn">Выбрать файл</button>
                    <input type="file" id="imageInput" class="file-input" accept="image/*">
                </div>
            </div>

            <div class="preview-panel">
                <h2 class="preview-title">Предпросмотр изображения</h2>
                <div class="preview-image-container">
                    <div class="preview-placeholder" id="previewPlaceholder">
                        Изображение появится здесь
                    </div>
                    <img class="preview-image" id="previewImage" alt="Preview">
                </div>
                <div class="loading" id="loading">
                    <div class="loading-spinner"></div>
                    <div>Анализируем изображение...</div>
                </div>
            </div>
        </div>

        <div class="error-message" id="errorMessage"></div>

        <div class="results-panel" id="resultsPanel">
            <h2 class="results-title">Результаты классификации</h2>
            <div id="resultsContainer">
                <div class="empty-results">
                    Загрузите изображение, чтобы увидеть результаты классификации
                </div>
            </div>
        </div>
    </div>

    <script>
        const uploadZone = document.getElementById('uploadZone');
        const imageInput = document.getElementById('imageInput');
        const selectFileBtn = document.getElementById('selectFileBtn');
        const previewImage = document.getElementById('previewImage');
        const previewPlaceholder = document.getElementById('previewPlaceholder');
        const loading = document.getElementById('loading');
        const resultsPanel = document.getElementById('resultsPanel');
        const resultsContainer = document.getElementById('resultsContainer');
        const errorMessage = document.getElementById('errorMessage');

        const models = {
            'traditional': {
                name: 'Традиционная модель',
                description: 'SIFT + SVM'
            },
            'simple_cnn': {
                name: 'Простая CNN',
                description: 'Сверточная нейронная сеть'
            },
            'resnet': {
                name: 'ResNet',
                description: 'Глубокая остаточная сеть'
            }
        };

        selectFileBtn.addEventListener('click', () => imageInput.click());
        
        imageInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                handleImageUpload(e.target.files[0]);
            }
        });

        uploadZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadZone.classList.add('dragover');
        });

        uploadZone.addEventListener('dragleave', () => {
            uploadZone.classList.remove('dragover');
        });

        uploadZone.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadZone.classList.remove('dragover');
            if (e.dataTransfer.files.length > 0) {
                handleImageUpload(e.dataTransfer.files[0]);
            }
        });

        function handleImageUpload(file) {
            if (!file.type.match('image.*')) {
                showError('Пожалуйста, выберите файл изображения');
                return;
            }

            hideError();

            const reader = new FileReader();
            reader.onload = function(e) {
                previewImage.src = e.target.result;
                previewImage.style.display = 'block';
                previewPlaceholder.style.display = 'none';
                classifyImage(file);
            };
            reader.readAsDataURL(file);
        }

        function classifyImage(file) {
            loading.style.display = 'block';
            resultsContainer.innerHTML = '<div class="empty-results">Анализируем изображение...</div>';

            const formData = new FormData();
            formData.append('image', file);

            fetch('/predict', {
                method: 'POST',
                body: formData
            })
            .then(response => {
                if (!response.ok) {
                    throw new Error('Ошибка сети');
                }
                return response.json();
            })
            .then(data => {
                loading.style.display = 'none';
                
                if (data.error) {
                    showError(data.error);
                    return;
                }

                if (data.image) {
                    previewImage.src = data.image;
                }

                displayResults(data);
            })
            .catch(error => {
                loading.style.display = 'none';
                showError('Ошибка при классификации: ' + error.message);
            });
        }

        function displayResults(data) {
            resultsPanel.style.display = 'block';
            resultsContainer.innerHTML = '';
            
            let bestModel = null;
            let bestConfidence = 0;
            let hasResults = false;

            // Find best model
            Object.keys(models).forEach(modelKey => {
                if (data.results[modelKey]) {
                    hasResults = true;
                    const confidence = data.results[modelKey].confidence;
                    if (confidence > bestConfidence) {
                        bestConfidence = confidence;
                        bestModel = modelKey;
                    }
                }
            });

            Object.keys(models).forEach(modelKey => {
                if (data.results[modelKey]) {
                    const result = data.results[modelKey];
                    const modelInfo = models[modelKey];
                    const isWinner = modelKey === bestModel;
                    
                    createModelCard(modelInfo, result, isWinner);
                } else {
                    createDisabledModelCard(models[modelKey]);
                }
            });

            if (!hasResults) {
                resultsContainer.innerHTML = '<div class="empty-results">Не удалось выполнить классификацию</div>';
            }
        }

        function createModelCard(modelInfo, result, isWinner) {
            const card = document.createElement('div');
            card.className = `model-card ${isWinner ? 'winner' : ''}`;
            
            const confidencePercent = Math.round(result.confidence * 100);
            
            card.innerHTML = `
                <div class="model-header">
                    <div>
                        <div class="model-name">${modelInfo.name}</div>
                        <div style="font-size: 12px; color: #666; margin-top: 2px;">${modelInfo.description}</div>
                    </div>
                    ${isWinner ? '<div class="winner-badge">Лучший результат</div>' : ''}
                </div>
                <div class="prediction-class">${result.class}</div>
                <div class="confidence-bar">
                    <div class="confidence-fill" style="width: ${confidencePercent}%"></div>
                </div>
                <div class="confidence-value">
                    Уверенность модели: <span class="confidence-percent">${confidencePercent}%</span>
                </div>
            `;
            
            resultsContainer.appendChild(card);
        }

        function createDisabledModelCard(modelInfo) {
            const card = document.createElement('div');
            card.className = 'model-card';
            card.style.opacity = '0.6';
            
            card.innerHTML = `
                <div class="model-header">
                    <div>
                        <div class="model-name">${modelInfo.name}</div>
                        <div style="font-size: 12px; color: #666; margin-top: 2px;">${modelInfo.description}</div>
                    </div>
                </div>
                <div class="prediction-class" style="color: #666;">Модель недоступна</div>
                <div class="confidence-bar">
                    <div class="confidence-fill" style="width: 0%"></div>
                </div>
                <div class="confidence-value">
                    Уверенность модели: <span class="confidence-percent">-</span>
                </div>
            `;
            
            resultsContainer.appendChild(card);
        }

        function showError(message) {
            errorMessage.textContent = message;
            errorMessage.style.display = 'block';
            resultsContainer.innerHTML = '<div class="empty-results">Произошла ошибка при обработке</div>';
        }

        function hideError() {
            errorMessage.style.display = 'none';
        }

        resultsPanel.style.display = 'block';
    </script>
</body>
</html>
'''

@app.route('/')
def index():
    return HTML

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'})
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No image selected'})
    
    try:
        image = Image.open(file.stream).convert('RGB')
        image = image.resize(Config.IMG_SIZE)
        image_array = np.array(image)
        
        results = {}
        
        if hasattr(traditional, 'is_trained') and traditional.is_trained:
            try:
                trad_pred, trad_probs = traditional.predict(image_array)
                trad_confidence = float(np.max(trad_probs))
                results['traditional'] = {
                    'class': Config.CLASSES[trad_pred],
                    'confidence': trad_confidence
                }
            except Exception as e:
                print(f"Traditional prediction error: {e}")
        
        try:
            cnn_pred, cnn_probs = simple_cnn.predict(image_array)
            cnn_confidence = float(np.max(cnn_probs))
            results['simple_cnn'] = {
                'class': Config.CLASSES[cnn_pred],
                'confidence': cnn_confidence
            }
        except Exception as e:
            print(f"Simple CNN prediction error: {e}")
        
        try:
            resnet_pred, resnet_probs = resnet.predict(image_array)
            resnet_confidence = float(np.max(resnet_probs))
            results['resnet'] = {
                'class': Config.CLASSES[resnet_pred],
                'confidence': resnet_confidence
            }
        except Exception as e:
            print(f"ResNet prediction error: {e}")
        
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        return jsonify({
            'success': True,
            'image': f"data:image/jpeg;base64,{img_str}",
            'results': results
        })
        
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    init_models()
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True, host='0.0.0.0', port=5000)