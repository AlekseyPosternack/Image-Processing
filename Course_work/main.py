import os
import sys
from data_analysis import DataAnalyzer
from traditional_model import TraditionalClassifier
from simple_cnn import SimpleCNNClassifier
from resnet_model import ResNetClassifier
from model_compare import ModelComparator
from config import Config

def main():
    """Основная программа"""
    print("🌍 EuroSAT Classifier")
    print("=" * 50)
    
    os.makedirs(Config.MODELS_PATH, exist_ok=True)
    
    while True:
        print("\nВыберите действие:")
        print("1. Анализ данных")
        print("2. Обучить все модели")
        print("3. Сравнить модели")
        print("4. Запуск веб-интерфейса")
        print("5. Выход")
        
        choice = input("\nВведите номер (1-5): ").strip()
        
        if choice == "1":
            analyzer = DataAnalyzer()
            analyzer.analyze()
            
        elif choice == "2":
            print("\nОбучение моделей\n")
            
            print("\nTraditional Classifier\n")
            traditional = TraditionalClassifier()
            traditional.train()
            traditional.save_model()
            
            print("\nSimple CNN\n")
            simple_cnn = SimpleCNNClassifier()
            simple_cnn.train()
            simple_cnn.save_model()
            
            print("\nResNet\n")
            resnet = ResNetClassifier()
            resnet.train()
            resnet.save_model()
            
            print("\n Все модели обучены и сохранены!")
            
        elif choice == "3":
            comparator = ModelComparator()
            results, best_model = comparator.compare_models()
            
        elif choice == "4":
            print("\nЗапуск веб-интерфейса...")
            print("http://localhost:5000")
            os.system("python Course_work/app.py")
            
        elif choice == "5":
            print("Выход из программы")
            break
            
        else:
            print("Некорректный выбор")

if __name__ == "__main__":
    main()