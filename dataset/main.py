import cv2
import numpy as np
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

class BeachPlantClassifier:
    def __init__(self, k=3):
        self.k = k
        self.model = KNeighborsClassifier(n_neighbors=k)
        self.class_names = []
        
    def extract_features(self, image_path):
        """Ekstrak fitur sederhana dari gambar"""
        # Baca gambar
        img = cv2.imread(image_path)
        if img is None:
            return None
            
        # Resize gambar ke ukuran standar
        img = cv2.resize(img, (64, 64))
        
        # Konversi ke HSV untuk fitur warna
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Ekstrak fitur:
        # 1. Histogram warna (HSV)
        hist_h = cv2.calcHist([hsv], [0], None, [16], [0, 180])
        hist_s = cv2.calcHist([hsv], [1], None, [16], [0, 256])
        hist_v = cv2.calcHist([hsv], [2], None, [16], [0, 256])
        
        # 2. Rata-rata warna
        mean_color = np.mean(img.reshape(-1, 3), axis=0)
        
        # 3. Tekstur sederhana (standar deviasi)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        texture = np.std(gray)
        
        # Gabungkan semua fitur
        features = np.concatenate([
            hist_h.flatten(),
            hist_s.flatten(), 
            hist_v.flatten(),
            mean_color,
            [texture]
        ])
        
        return features
    
    def load_dataset(self, dataset_path):
        """Muat dataset dari folder"""
        features = []
        labels = []
        
        # Ambil nama kelas dari folder dengan debugging detail
        all_items = os.listdir(dataset_path)
        print(f"Semua item di {dataset_path}: {all_items}")
        
        self.class_names = []
        skipped_folders = []
        
        for item in all_items:
            item_path = os.path.join(dataset_path, item)
            if os.path.isdir(item_path):
                # Cek apakah ada gambar di folder ini
                images = [f for f in os.listdir(item_path) 
                         if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
                
                if len(images) > 0:
                    self.class_names.append(item)
                    print(f"✅ Folder '{item}': {len(images)} gambar")
                else:
                    skipped_folders.append(item)
                    print(f"⚠️  Folder '{item}': KOSONG (tidak ada gambar)")
            else:
                print(f"📄 File (bukan folder): {item}")
        
        if skipped_folders:
            print(f"\n❌ Folder yang dilewati (kosong): {skipped_folders}")
        
        print(f"\n📊 Total kelas yang akan digunakan: {len(self.class_names)}")
        print(f"Kelas: {self.class_names}")
        
        if len(self.class_names) == 0:
            raise ValueError("Tidak ada folder dengan gambar yang ditemukan!")
        
        # Load gambar dan ekstrak fitur
        for class_idx, class_name in enumerate(self.class_names):
            class_path = os.path.join(dataset_path, class_name)
            
            print(f"\n🔄 Memproses kelas '{class_name}'...")
            
            images = [f for f in os.listdir(class_path) 
                     if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
            
            processed_count = 0
            failed_count = 0
            
            for img_file in images:
                img_path = os.path.join(class_path, img_file)
                
                # Ekstrak fitur
                feature = self.extract_features(img_path)
                if feature is not None:
                    features.append(feature)
                    labels.append(class_idx)
                    processed_count += 1
                else:
                    failed_count += 1
                    print(f"   ❌ Gagal memproses: {img_file}")
            
            print(f"   ✅ Berhasil: {processed_count}, Gagal: {failed_count}")
                        
        print(f"\n📈 Total gambar yang berhasil diproses: {len(features)}")
        return np.array(features), np.array(labels)
    
    def train(self, dataset_path):
        """Latih model KNN"""
        print("Memuat dataset...")
        X, y = self.load_dataset(dataset_path)
        
        print(f"Dataset dimuat: {len(X)} gambar, {len(self.class_names)} kelas")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Latih model
        print("Melatih model KNN...")
        self.model.fit(X_train, y_train)
        
        # Evaluasi
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Akurasi: {accuracy:.2f}")
        print("\nLaporan klasifikasi:")
        print(classification_report(y_test, y_pred, target_names=self.class_names))
        
        return accuracy
    
    def predict(self, image_path):
        """Prediksi kelas tanaman dari gambar"""
        features = self.extract_features(image_path)
        if features is None:
            return None, 0
            
        # Prediksi dengan probabilitas
        prediction = self.model.predict([features])[0]
        probabilities = self.model.predict_proba([features])[0]
        confidence = max(probabilities)
        
        return self.class_names[prediction], confidence
    
    def predict_and_show(self, image_path):
        """Prediksi dan tampilkan hasil"""
        prediction, confidence = self.predict(image_path)
        
        if prediction is None:
            print("Gagal memproses gambar")
            return
            
        # Tampilkan gambar dan hasil
        img = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        plt.figure(figsize=(8, 6))
        plt.imshow(img_rgb)
        plt.title(f'Prediksi: {prediction}\nKonfiden: {confidence:.2f}')
        plt.axis('off')
        plt.show()
        
        print(f"Hasil prediksi: {prediction}")
        print(f"Tingkat kepercayaan: {confidence:.2f}")

def check_dataset_structure(dataset_path):
    """Cek struktur dataset yang ada dengan detail lengkap"""
    if not os.path.exists(dataset_path):
        print(f"❌ Folder dataset '{dataset_path}' tidak ditemukan!")
        return False, []
    
    print(f"🔍 Mengecek struktur dataset di: {dataset_path}")
    print("=" * 60)
    
    # Cek semua item di folder dataset
    all_items = os.listdir(dataset_path)
    print(f"Semua item di folder: {len(all_items)}")
    
    class_folders = []
    empty_folders = []
    files_found = []
    
    for item in all_items:
        item_path = os.path.join(dataset_path, item)
        
        if os.path.isdir(item_path):
            # Cek gambar di folder
            images = []
            for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG', '.BMP']:
                images.extend([f for f in os.listdir(item_path) if f.endswith(ext)])
            
            if len(images) > 0:
                class_folders.append({
                    'name': item,
                    'path': item_path,
                    'count': len(images),
                    'images': images[:3]  # Sample 3 nama file
                })
                print(f"✅ 📁 {item:<25} | {len(images):3d} gambar | Sample: {', '.join(images[:2])}")
            else:
                empty_folders.append(item)
                print(f"⚠️  📁 {item:<25} | KOSONG")
        else:
            files_found.append(item)
            print(f"📄 {item}")
    
    print("=" * 60)
    
    if empty_folders:
        print(f"❌ Folder kosong: {empty_folders}")
    
    if files_found:
        print(f"📄 File di root: {files_found}")
    
    if not class_folders:
        print("❌ Tidak ada folder kelas dengan gambar yang ditemukan!")
        print("\n💡 Pastikan struktur seperti ini:")
        print("   dataset/")
        print("   ├── foto cemara laut/")
        print("   │   ├── gambar1.jpg")
        print("   │   └── gambar2.jpg")
        print("   ├── foto kelapa/")
        print("   └── dst...")
        return False, []
    
    total_images = sum(folder['count'] for folder in class_folders)
    print(f"\n📊 RINGKASAN:")
    print(f"   • Total kelas: {len(class_folders)}")
    print(f"   • Total gambar: {total_images}")
    print(f"   • Rata-rata per kelas: {total_images/len(class_folders):.1f}")
    
    return True, class_folders

def test_with_sample_images(classifier):
    """Test dengan gambar sample dari dataset"""
    print("\n=== TESTING DENGAN GAMBAR SAMPLE ===")
    
    if not classifier.class_names:
        print("Model belum dilatih!")
        return
    
    # Ambil beberapa gambar random untuk test dari setiap kelas
    test_samples = []
    dataset_path = "dataset_tanaman_pantai"  # Sesuaikan dengan nama folder Anda
    
    for plant_class in classifier.class_names:
        class_path = os.path.join(dataset_path, plant_class)
        if os.path.isdir(class_path):
            images = [f for f in os.listdir(class_path) 
                     if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
            if images:
                # Ambil maksimal 3 gambar random per kelas untuk testing
                num_samples = min(3, len(images))
                sample_imgs = random.sample(images, num_samples)
                for img in sample_imgs:
                    test_samples.append({
                        'path': os.path.join(class_path, img),
                        'true_class': plant_class,
                        'filename': img
                    })
    
    if not test_samples:
        print("Tidak ada gambar untuk testing!")
        return
    
    # Test prediksi
    correct = 0
    total = len(test_samples)
    results_by_class = {}
    
    print(f"\nTesting dengan {total} gambar sample dari {len(classifier.class_names)} kelas:")
    print("=" * 70)
    
    for i, sample in enumerate(test_samples):
        prediction, confidence = classifier.predict(sample['path'])
        is_correct = prediction == sample['true_class']
        
        if is_correct:
            correct += 1
        
        # Catat hasil per kelas
        true_class = sample['true_class']
        if true_class not in results_by_class:
            results_by_class[true_class] = {'correct': 0, 'total': 0}
        results_by_class[true_class]['total'] += 1
        if is_correct:
            results_by_class[true_class]['correct'] += 1
        
        # Tampilkan hasil
        status = "✓ BENAR" if is_correct else "✗ SALAH"
        print(f"{i+1:2d}. {sample['filename'][:25]:<25} | "
              f"Asli: {true_class:<15} | "
              f"Prediksi: {prediction:<15} | "
              f"Conf: {confidence:.3f} | {status}")
    
    print("=" * 70)
    
    # Ringkasan hasil
    accuracy = correct / total if total > 0 else 0
    print(f"AKURASI KESELURUHAN: {accuracy:.3f} ({correct}/{total})")
    
    print("\nAKURASI PER KELAS:")
    for class_name in sorted(results_by_class.keys()):
        stats = results_by_class[class_name]
        class_acc = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
        print(f"  {class_name:<20}: {class_acc:.3f} ({stats['correct']}/{stats['total']})")

def predict_single_image(classifier, image_path):
    """Prediksi dan tampilkan hasil untuk satu gambar"""
    if not os.path.exists(image_path):
        print(f"File tidak ditemukan: {image_path}")
        return
    
    print(f"\n=== PREDIKSI GAMBAR: {os.path.basename(image_path)} ===")
    
    prediction, confidence = classifier.predict(image_path)
    
    if prediction is None:
        print("❌ Gagal memproses gambar")
        return
    
    print(f"🔍 Hasil Prediksi: {prediction}")
    print(f"📊 Tingkat Keyakinan: {confidence:.3f}")
    
    # Tampilkan probabilitas untuk semua kelas
    features = classifier.extract_features(image_path)
    if features is not None:
        probabilities = classifier.model.predict_proba([features])[0]
        print(f"\n📈 Probabilitas semua kelas:")
        for i, prob in enumerate(probabilities):
            class_name = classifier.class_names[i]
            print(f"  {class_name:<20}: {prob:.3f}")

# Contoh penggunaan untuk data Anda
if __name__ == "__main__":
    import random
    
    print("=== KLASIFIKASI TANAMAN PANTAI DENGAN KNN ===")
    print("Dataset: cemara laut, kelapa, ketapang, pandan pantai, tanaman mangrove")
    
    # Sesuaikan dengan nama folder dataset Anda
    dataset_path = "dataset_tanaman_pantai"  # GANTI SESUAI NAMA FOLDER ANDA
    
    # Alternatif nama folder yang mungkin
    possible_paths = [
        "dataset_tanaman_pantai",
        "dataset", 
        "data",
        "tanaman_pantai",
        "images"
    ]
    
    # Cari folder dataset yang ada
    found_dataset = None
    for path in possible_paths:
        if os.path.exists(path):
            is_valid, folders = check_dataset_structure(path)
            if is_valid:
                found_dataset = path
                break
    
    if not found_dataset:
        print("❌ Dataset tidak ditemukan!")
        print("💡 Pastikan struktur folder seperti ini:")
        print("   dataset_tanaman_pantai/")
        print("   ├── foto cemara laut/")
        print("   ├── foto kelapa/") 
        print("   ├── foto ketapang/")
        print("   ├── foto pandan pantai/")
        print("   └── foto tanaman mangrove/")
        print("\n🔧 Atau ubah nama folder di variabel 'dataset_path' pada kode")
        exit()
    
    print(f"✅ Menggunakan dataset: {found_dataset}")
    
    # Inisialisasi classifier dengan K yang sesuai
    classifier = BeachPlantClassifier(k=3)  # K=3 karena kemungkinan data tidak terlalu banyak
    
    # Latih model
    print("\n=== 🚀 TRAINING MODEL ===")
    try:
        accuracy = classifier.train(found_dataset)
        print(f"✅ Training selesai dengan akurasi: {accuracy:.3f}")
    except Exception as e:
        print(f"❌ Error saat training: {e}")
        exit()
    
    # Test dengan sample images
    test_with_sample_images(classifier)
    
    # Demo prediksi gambar tunggal
    print("\n=== 🔍 DEMO PREDIKSI GAMBAR TUNGGAL ===")
    
    # Ambil satu gambar random untuk demo
    all_images = []
    for root, dirs, files in os.walk(found_dataset):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                all_images.append(os.path.join(root, file))
    
    if all_images:
        # Pilih gambar random
        demo_image = random.choice(all_images)
        predict_single_image(classifier, demo_image)
        
        print(f"\n💡 Untuk test gambar lain, gunakan:")
        print(f"   predict_single_image(classifier, 'path/ke/gambar.jpg')")
    
    print("\n🎉 Program selesai!")
    print("📝 Tips: Untuk akurasi lebih baik, pastikan setiap kelas punya minimal 10-20 gambar")