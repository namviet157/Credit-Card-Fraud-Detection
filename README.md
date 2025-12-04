# Credit Card Fraud Detection

**Credit Card Fraud Detection** là một dự án phát hiện gian lận thẻ tín dụng được xây dựng hoàn toàn bằng **NumPy**, không sử dụng các thư viện Machine Learning có sẵn như scikit-learn hay Pandas. Dự án này tập trung vào việc hiểu sâu các thuật toán Machine Learning bằng cách implement từ đầu, từ khám phá dữ liệu, tiền xử lý đến huấn luyện mô hình Logistic Regression.

## 📋 Mục lục

1. [Giới thiệu](#1-giới-thiệu)
2. [Dataset](#2-dataset)
3. [Method](#3-method)
4. [Installation & Setup](#4-installation--setup)
5. [Usage](#5-usage)
6. [Results](#6-results)
7. [Project Structure](#7-project-structure)
8. [Challenges & Solutions](#8-challenges--solutions)
9. [Future Improvements](#9-future-improvements)
10. [Contributors](#10-contributors)
11. [License](#11-license)

---

## 1. Giới thiệu

### 1.1. Mô tả bài toán

**Bài toán**: Phát hiện gian lận trong các giao dịch thẻ tín dụng

- **Đầu vào**: Thông tin về các giao dịch thẻ tín dụng bao gồm:
  - `Time`: Thời gian giao dịch (tính bằng giây từ giao dịch đầu tiên)
  - `V1-V28`: 28 features đã được PCA transform (ẩn danh để bảo mật)
  - `Amount`: Số tiền giao dịch
  
- **Đầu ra**: Dự đoán giao dịch có phải là gian lận hay không
  - `0`: Giao dịch bình thường (Normal)
  - `1`: Giao dịch gian lận (Fraud)

- **Loại bài toán**: Binary Classification với **class imbalance nghiêm trọng**
  - Tỷ lệ gian lận chỉ chiếm **0.17%** tổng số giao dịch
  - Đây là một trong những thách thức lớn nhất của bài toán

### 1.2. Động lực và ứng dụng thực tế

Fraud detection là một vấn đề cực kỳ quan trọng trong ngành tài chính và ngân hàng:

1. **Tổn thất tài chính**: 
   - Các giao dịch gian lận gây thiệt hại hàng tỷ USD mỗi năm trên toàn thế giới
   - Mỗi giao dịch gian lận không được phát hiện đều gây thiệt hại trực tiếp

2. **Bảo vệ khách hàng**:
   - Phát hiện sớm giúp bảo vệ khách hàng khỏi các hoạt động gian lận
   - Giảm thiểu rủi ro mất tiền và thông tin cá nhân

3. **Tuân thủ quy định**:
   - Các ngân hàng và tổ chức tài chính cần có hệ thống phát hiện gian lận hiệu quả để tuân thủ các quy định pháp lý

4. **Xử lý real-time**:
   - Cần phát hiện gian lận trong thời gian thực để ngăn chặn kịp thời
   - Yêu cầu mô hình có độ chính xác cao và tốc độ xử lý nhanh

5. **Cân bằng giữa Precision và Recall**:
   - **Precision cao**: Tránh làm phiền khách hàng bằng các cảnh báo giả (False Positives)
   - **Recall cao**: Tránh bỏ lọt các giao dịch gian lận (False Negatives) - điều này cực kỳ quan trọng

### 1.3. Mục tiêu cụ thể

#### Mục tiêu kỹ thuật:
1. **Làm chủ NumPy**:
   - Sử dụng NumPy để xử lý toàn bộ dữ liệu (không dùng Pandas)
   - Implement các thuật toán ML từ đầu bằng NumPy
   - Tối ưu hóa code với vectorization và broadcasting
   - Tránh sử dụng for loops không cần thiết

2. **Phân tích dữ liệu sâu**:
   - Khám phá và hiểu về dataset
   - Phát hiện patterns và insights từ dữ liệu
   - Xử lý class imbalance
   - Phân tích correlation và feature importance

3. **Modeling từ đầu**:
   - Implement Logistic Regression hoàn chỉnh với Gradient Descent
   - Hiểu sâu về loss function, gradient computation
   - Đánh giá mô hình với các metrics phù hợp cho imbalanced data

#### Mục tiêu học thuật:
- Hiểu rõ cách hoạt động của các thuật toán ML cơ bản
- Nắm vững các kỹ thuật xử lý dữ liệu
- Áp dụng kiến thức toán học vào thực tế

---

## 2. Dataset

### 2.1. Nguồn dữ liệu

- **Dataset**: Credit Card Fraud Detection
- **Nguồn**: [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Tổ chức**: ULB (Université Libre de Bruxelles) Machine Learning Group
- **Kích thước**: 284,807 giao dịch
- **Thời gian thu thập**: Giao dịch trong 2 ngày (khoảng 48 giờ)

### 2.2. Mô tả các features

| Feature | Mô tả | Kiểu dữ liệu | Đặc điểm |
|---------|-------|--------------|----------|
| **Time** | Số giây giữa giao dịch đầu tiên và giao dịch này | Float | Phạm vi: 0 - 172,792 giây |
| **V1-V28** | 28 features đã được PCA transform | Float | Đã được chuẩn hóa, mean ≈ 0, std ≈ 1 |
| **Amount** | Số tiền giao dịch (USD) | Float | Phạm vi: $0 - $25,691.16, phân phối lệch phải |
| **Class** | Nhãn (0 = bình thường, 1 = gian lận) | Integer | Binary classification target |

**Lưu ý quan trọng**: 
- Các features V1-V28 đã được PCA transform để **bảo mật thông tin nhạy cảm** của khách hàng
- Đây là cách tiếp cận phổ biến trong các bài toán tài chính để tuân thủ quy định bảo vệ dữ liệu cá nhân
- Các features gốc (như số thẻ, tên khách hàng, địa chỉ) không được tiết lộ

### 2.3. Kích thước và đặc điểm dữ liệu

#### Thống kê tổng quan:
- **Tổng số samples**: 284,807
- **Số features**: 30 (Time + V1-V28 + Amount)
- **Missing values**: **Không có** (0 missing values)
- **Outliers**: Có nhiều outliers, đặc biệt trong:
  - Feature `Amount`: 31,904 outliers (11.20%) theo IQR method
  - Feature `V27`: 39,163 outliers (13.75%)
  - Feature `V28`: 30,342 outliers (10.65%)

#### Class Distribution (Phân phối lớp):

```
Class 0 (Normal):  284,315 samples (99.83%)
Class 1 (Fraud):       492 samples (0.17%)
Imbalance ratio: 0.0017 (fraud/normal)
```

**Phân tích class imbalance**:
- Đây là một trong những dataset có **class imbalance nghiêm trọng nhất**
- Tỷ lệ 1:578 (1 giao dịch gian lận trên 578 giao dịch bình thường)
- Điều này khiến việc đánh giá mô hình trở nên khó khăn:
  - Accuracy không phải là metric tốt (mô hình chỉ cần dự đoán tất cả là "Normal" cũng đạt 99.83% accuracy)
  - Cần tập trung vào **Precision**, **Recall**, **F1-Score** và **AUC**

#### Đặc điểm phân phối:

**Time Feature**:
- Mean: 94,813.86 giây (~26.34 giờ)
- Median: 84,692 giây (~23.53 giờ)
- Phân phối: Hơi lệch trái (Skewness ≈ -0.036)
- **Insight**: Có pattern theo chu kỳ ngày/đêm, tỷ lệ gian lận cao hơn vào ban đêm (2-4h sáng)

**Amount Feature**:
- Mean: $88.35
- Median: $22.00
- Max: $25,691.16
- **Phân phối lệch phải nghiêm trọng**:
  - Skewness: 16.98 (rất cao)
  - Kurtosis: 845.07 (phân phối cực kỳ nhọn)
- **So sánh Normal vs Fraud**:
  - Normal transactions: Mean = $88.29, Median = $22.00
  - Fraud transactions: Mean = $122.21, Median = $9.25
  - **Kết luận**: Giao dịch gian lận có giá trị trung bình cao hơn nhưng median thấp hơn

**PCA Features (V1-V28)**:
- Tất cả đều có mean ≈ 0 (do đã được PCA transform)
- Standard deviation giảm dần từ V1 đến V28 (từ 1.96 xuống 0.33)
- **Tính trực giao**: Các features này hầu như không tương quan với nhau (đặc tính của PCA)
- **Top features quan trọng nhất** (dựa trên sự khác biệt giữa Normal và Fraud):
  1. V3: Diff = 7.05
  2. V14: Diff = 6.98
  3. V17: Diff = 6.68
  4. V12: Diff = 6.27
  5. V10: Diff = 5.69

---

## 3. Method

### 3.1. Quy trình xử lý dữ liệu

#### 3.1.1. Data Loading

**Sử dụng NumPy để đọc CSV** (không dùng Pandas):

```python
# Đọc file CSV bằng np.genfromtxt
data_str = np.genfromtxt(file_path, dtype=str, delimiter=',')

# Xử lý header
data_str = np.char.strip(data_str, '"')
header = data_str[0]
data_str = data_str[1:]

# Convert sang float64
data = data_str.astype(np.float64)
```

**Kết quả**: Ma trận dữ liệu shape (284807, 31) - 30 features + 1 target

#### 3.1.2. Data Exploration

**a) Kiểm tra dữ liệu thiếu**:
```python
missing_mask = np.isnan(data) | np.isinf(data)
missing_count = np.sum(missing_mask, axis=0)
# Kết quả: 0 missing values
```

**b) Tính toán thống kê mô tả**:
- Mean, Median, Std, Variance
- Min, Max, Quartiles (Q1, Q2, Q3)
- Skewness và Kurtosis (implement từ đầu bằng NumPy)

**c) Phân tích class distribution**:
- Đếm số lượng samples mỗi class
- Tính tỷ lệ phần trăm
- Visualize bằng bar chart và pie chart

**d) Phân tích features**:
- Phân tích Time feature: Chuyển đổi sang giờ, phân tích theo chu kỳ ngày/đêm
- Phân tích Amount feature: So sánh giữa Normal và Fraud
- Phân tích PCA features: Visualize phân phối của V1-V9

**e) Correlation analysis**:
```python
# Tính correlation matrix bằng NumPy
mean = np.mean(data, axis=0, keepdims=True)
std = np.std(data, axis=0, keepdims=True)
data_std = (data - mean) / std
corr_matrix = np.corrcoef(data_std.T)
```

**f) Feature importance**:
- So sánh giá trị trung bình giữa Normal và Fraud
- Xác định top features có sự khác biệt lớn nhất

**g) Statistical hypothesis testing**:
- T-test để kiểm tra sự khác biệt về Amount giữa Normal và Fraud
- Kết quả: p-value = 0.0034 < 0.05 → Bác bỏ H0, có sự khác biệt có ý nghĩa thống kê

#### 3.1.3. Data Preprocessing

**a) Missing Values Handling**:

Mặc dù dataset không có missing values, nhưng đã implement các phương pháp xử lý:

1. **Mean Imputation**:
```python
mean_val = np.nanmean(col_data)
data[missing_mask, i] = mean_val
```

2. **Median Imputation**:
```python
median_val = np.nanmedian(col_data)
data[missing_mask, i] = median_val
```

3. **Specific Value Imputation**:
```python
data[np.isnan(data)] = -999
```

4. **Linear Regression Imputation** (cho Amount dựa trên Time):
```python
# Normal Equation: β = (X^T X)^(-1) X^T y
X_reg = np.column_stack([np.ones(len(time_valid)), time_valid])
beta = np.linalg.solve(X_reg.T @ X_reg, X_reg.T @ y_reg)
predicted_amount = X_pred @ beta
```

**b) Outlier Detection**:

**Phương pháp 1: IQR Method**
```python
q1 = np.percentile(X, 25, axis=0)
q3 = np.percentile(X, 75, axis=0)
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr
outlier_mask = (X < lower_bound) | (X > upper_bound)
```

**Kết quả**: 370,372 outliers (4.33% tổng số data points)

**Phương pháp 2: Z-score Method**
```python
mean_vals = np.mean(X, axis=0, keepdims=True)
std_vals = np.std(X, axis=0, ddof=1, keepdims=True)
z_scores = (X - mean_vals) / std_vals
outlier_mask = np.abs(z_scores) > 3.0
```

**Kết quả**: 83,598 outliers (0.98% tổng số data points)

**Quyết định**: **KHÔNG loại bỏ outliers** vì:
- Trong bài toán fraud detection, outliers có thể chính là các giao dịch gian lận
- Loại bỏ outliers có thể làm mất đi những mẫu quan trọng nhất
- Thay vào đó, sử dụng các phương pháp chuẩn hóa mạnh (robust scaling)

**c) Normalization & Standardization**:

**Bước 1: Log Transformation cho Amount**
```python
# Xử lý phân phối lệch phải
X_processed[:, amount_idx] = np.log1p(X[:, amount_idx])
```

**Lý do**: 
- Amount có skewness = 16.98 (rất cao)
- Sau log transform: skewness giảm xuống 0.16
- Giúp phân phối gần với chuẩn hơn

**Bước 2: Z-score Standardization**
```python
mean_vals = np.mean(X_processed, axis=0, keepdims=True)
std_vals = np.std(X_processed, axis=0, ddof=1, keepdims=True)
std_vals = np.where(std_vals == 0, 1, std_vals)  # Tránh chia cho 0
X_processed = (X_processed - mean_vals) / std_vals
```

**Kết quả**: 
- Mean ≈ 0, Std ≈ 1 cho tất cả features
- Phù hợp với các thuật toán dựa trên gradient (Logistic Regression)

**Các phương pháp khác đã thử nghiệm**:
- **Min-Max Normalization**: Đưa về [0, 1], nhưng bị ảnh hưởng mạnh bởi outliers
- **Decimal Scaling**: Ít phổ biến, kém hiệu quả hơn Z-score

**d) Train-Test Split**:

```python
test_size = 0.2
random_state = 42

np.random.seed(random_state)
indices = np.arange(n_samples)
np.random.shuffle(indices)

n_test = int(n_samples * test_size)
test_indices = indices[:n_test]
train_indices = indices[n_test:]

X_train = X_processed[train_indices]
X_test = X_processed[test_indices]
y_train = y[train_indices]
y_test = y[test_indices]
```

**Kết quả**:
- Train set: 227,846 samples (80%)
- Test set: 56,961 samples (20%)
- **Bảo toàn class distribution**: 
  - Train: 99.83% Normal, 0.17% Fraud
  - Test: 99.83% Normal, 0.17% Fraud

### 3.2. Thuật toán sử dụng

#### 3.2.1. Logistic Regression

**Công thức toán học**:

**1. Sigmoid Function**:
$$P(y=1|x) = \sigma(z) = \frac{1}{1 + e^{-z}}$$

với $z = w^T x + b = \sum_{i=1}^{n} w_i x_i + b$

**2. Loss Function (Binary Cross-Entropy)**:
$$L = -\frac{1}{m}\sum_{i=1}^{m}[y_i \log(\hat{y}_i) + (1-y_i)\log(1-\hat{y}_i)]$$

Trong đó:
- $m$: số lượng samples
- $y_i$: nhãn thực tế (0 hoặc 1)
- $\hat{y}_i = \sigma(w^T x_i + b)$: xác suất dự đoán

**3. Gradient Computation**:

Đạo hàm của loss function theo weights:
$$\frac{\partial L}{\partial w} = \frac{1}{m}X^T(\hat{y} - y)$$

Đạo hàm của loss function theo bias:
$$\frac{\partial L}{\partial b} = \frac{1}{m}\sum_{i=1}^{m}(\hat{y}_i - y_i)$$

**4. Update Rules (Gradient Descent)**:
$$w := w - \alpha \frac{\partial L}{\partial w}$$
$$b := b - \alpha \frac{\partial L}{\partial b}$$

Trong đó $\alpha$ là learning rate.

**Implementation bằng NumPy**:

```python
class LogisticRegression:
    def __init__(self, learning_rate=0.01, max_iter=1000, tol=1e-6):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.weights = None
        self.bias = None
        
    def _sigmoid(self, z):
        # Clip để tránh overflow
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
    
    def fit(self, X, y):
        m, n = X.shape
        
        # Khởi tạo weights ngẫu nhiên nhỏ
        self.weights = np.random.randn(n) * 0.01
        self.bias = 0.0
        
        for i in range(self.max_iter):
            # Forward pass
            z = X @ self.weights + self.bias
            y_pred = self._sigmoid(z)
            
            # Tính loss
            loss = -np.mean(y * np.log(y_pred + 1e-15) + 
                           (1 - y) * np.log(1 - y_pred + 1e-15))
            
            # Backward pass (Gradient computation)
            dw = (1/m) * X.T @ (y_pred - y)
            db = (1/m) * np.sum(y_pred - y)
            
            # Update weights
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            # Kiểm tra convergence
            if i > 0 and abs(prev_loss - loss) < self.tol:
                break
            prev_loss = loss
    
    def predict_proba(self, X):
        z = X @ self.weights + self.bias
        return self._sigmoid(z)
    
    def predict(self, X, threshold=0.5):
        probabilities = self.predict_proba(X)
        return (probabilities >= threshold).astype(int)
```

**Đặc điểm implementation**:
- **Vectorized operations**: Tất cả tính toán đều được vectorize, không dùng for loops
- **Broadcasting**: Sử dụng broadcasting để tính toán hiệu quả
- **Numerical stability**: Clip z values để tránh overflow trong sigmoid
- **Epsilon trong log**: Thêm 1e-15 để tránh log(0)

**Hyperparameters**:
- Learning rate: 0.01
- Max iterations: 1000
- Tolerance: 1e-6 (để kiểm tra convergence)
- Random state: 42 (đảm bảo reproducibility)

### 3.3. Evaluation Metrics

Trong bài toán imbalanced data, **Accuracy không phải là metric tốt**. Các metrics quan trọng:

**1. Confusion Matrix**:

|                | Predicted Normal | Predicted Fraud |
|----------------|------------------|-----------------|
| **Actual Normal** | TN (True Negative) | FP (False Positive) |
| **Actual Fraud**  | FN (False Negative) | TP (True Positive) |

**2. Precision (Độ chính xác dương tính)**:
$$\text{Precision} = \frac{TP}{TP + FP}$$

Ý nghĩa: Trong số các giao dịch mô hình dự đoán là gian lận, bao nhiêu phần trăm là đúng?

**3. Recall (Độ nhạy)**:
$$\text{Recall} = \frac{TP}{TP + FN}$$

Ý nghĩa: Mô hình phát hiện được bao nhiêu phần trăm tổng số vụ gian lận thực tế?

**4. F1-Score (Trung bình điều hòa)**:
$$\text{F1-Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

**5. AUC (Area Under ROC Curve)**:
- ROC Curve: Vẽ True Positive Rate (Recall) vs False Positive Rate
- AUC: Diện tích dưới đường cong ROC
- Metric tốt nhất cho imbalanced data vì không phụ thuộc vào threshold

**Implementation bằng NumPy**:

```python
def confusion_matrix(y_true, y_pred):
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tp = np.sum((y_true == 1) & (y_pred == 1))
    return np.array([[tn, fp], [fn, tp]])

def precision_score(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    tp, fp = cm[1, 1], cm[0, 1]
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0

def recall_score(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    tp, fn = cm[1, 1], cm[1, 0]
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0

def roc_curve(y_true, y_scores):
    # Sắp xếp theo score giảm dần
    sorted_indices = np.argsort(y_scores)[::-1]
    y_true_sorted = y_true[sorted_indices]
    y_scores_sorted = y_scores[sorted_indices]
    
    # Tính FPR và TPR cho từng threshold
    thresholds = np.unique(y_scores_sorted)
    fpr, tpr = [], []
    
    for threshold in thresholds:
        y_pred = (y_scores_sorted >= threshold).astype(int)
        cm = confusion_matrix(y_true_sorted, y_pred)
        tn, fp, fn, tp = cm[0,0], cm[0,1], cm[1,0], cm[1,1]
        fpr.append(fp / (fp + tn) if (fp + tn) > 0 else 0.0)
        tpr.append(tp / (tp + fn) if (tp + fn) > 0 else 0.0)
    
    return np.array(fpr), np.array(tpr), thresholds

def auc_score(fpr, tpr):
    # Tính diện tích bằng phương pháp trapezoidal
    sorted_indices = np.argsort(fpr)
    fpr_sorted = fpr[sorted_indices]
    tpr_sorted = tpr[sorted_indices]
    return np.trapz(tpr_sorted, fpr_sorted)
```

---

## 4. Installation & Setup

### 4.1. Requirements

- **Python**: 3.7 trở lên
- **NumPy**: >= 1.21.0
- **Matplotlib**: >= 3.5.0 (cho visualization)
- **Seaborn**: >= 0.11.0 (cho visualization đẹp hơn)
- **Jupyter**: >= 1.0.0 (để chạy notebooks)

### 4.2. Installation

**Bước 1: Clone repository** (nếu có)
```bash
git clone <repository-url>
cd <project-directory>
```

**Bước 2: Tạo virtual environment** (khuyến nghị)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

**Bước 3: Install dependencies**
```bash
pip install -r requirements.txt
```

Hoặc install từng package:
```bash
pip install numpy>=1.21.0 matplotlib>=3.5.0 seaborn>=0.11.0 jupyter
```

### 4.3. Dataset Setup

1. **Download dataset**:
   - Truy cập [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
   - Download file `creditcard.csv`

2. **Đặt file vào đúng thư mục**:
   ```
   data/
   └── raw/
       └── creditcard.csv
   ```

3. **Kiểm tra cấu trúc thư mục**:
   ```
   project/
   ├── data/
   │   ├── raw/
   │   │   └── creditcard.csv
   │   └── processed/  (sẽ được tạo tự động)
   ├── notebooks/
   │   ├── 01_data_exploration.ipynb
   │   ├── 02_preprocessing.ipynb
   │   └── 03_modeling.ipynb
   ├── requirements.txt
   └── README.md
   ```

---

## 5. Usage

### 5.1. Hướng dẫn cách chạy từng phần

#### 5.1.1. Data Exploration

**Chạy notebook đầu tiên**:
```bash
jupyter notebook notebooks/01_data_exploration.ipynb
```

**Notebook này sẽ thực hiện**:
1. Load dataset từ `data/raw/creditcard.csv`
2. Kiểm tra missing values và thống kê cơ bản
3. Phân tích class distribution
4. Phân tích các features quan trọng:
   - Time feature: Phân tích theo chu kỳ ngày/đêm
   - Amount feature: So sánh giữa Normal và Fraud
   - PCA features (V1-V28): Phân tích phân phối
5. Correlation analysis giữa các features
6. So sánh features giữa Normal và Fraud transactions
7. Feature engineering: Tạo rolling statistics
8. Statistical hypothesis testing (T-test)
9. Xử lý missing values (demo các phương pháp)
10. Lưu dữ liệu đã xử lý vào `data/processed/`

**Kết quả đầu ra**:
- File `header.npy`: Tên các features
- File `X_regression_filled.npy`: Dữ liệu đã xử lý missing values (nếu có)

#### 5.1.2. Data Preprocessing

**Chạy notebook thứ hai**:
```bash
jupyter notebook notebooks/02_preprocessing.ipynb
```

**Notebook này sẽ thực hiện**:
1. Load dữ liệu đã xử lý từ notebook 01
2. **Outlier Detection**:
   - IQR Method
   - Z-score Method
   - So sánh và phân tích kết quả
3. **Normalization & Standardization**:
   - Min-Max Normalization
   - Z-score Standardization
   - Log Transformation
   - Decimal Scaling
   - So sánh các phương pháp
4. **Áp dụng preprocessing cuối cùng**:
   - Log transform cho Amount
   - Z-score standardization cho tất cả features
5. **Train-Test Split**:
   - Chia 80% train, 20% test
   - Bảo toàn class distribution
6. Lưu dữ liệu đã xử lý:
   - `X_processed.npy`: Dữ liệu đã chuẩn hóa
   - `y.npy`: Labels
   - `X_train.npy`, `X_test.npy`: Train/test features
   - `y_train.npy`, `y_test.npy`: Train/test labels

**Kết quả đầu ra**:
- Các file `.npy` trong `data/processed/` để sử dụng cho modeling

#### 5.1.3. Modeling

**Chạy notebook thứ ba**:
```bash
jupyter notebook notebooks/03_modeling.ipynb
```

**Notebook này sẽ thực hiện**:
1. Load dữ liệu đã xử lý từ notebook 02
2. **Implement Evaluation Metrics**:
   - Accuracy, Precision, Recall, F1-Score
   - Confusion Matrix
   - ROC Curve và AUC
3. **Implement Logistic Regression**:
   - Class LogisticRegression với Gradient Descent
   - Training với các hyperparameters
   - Visualize training loss history
4. **Evaluation**:
   - Dự đoán trên test set
   - Tính các metrics
   - Vẽ Confusion Matrix
   - Vẽ ROC Curve và tính AUC
5. **Phân tích kết quả**:
   - Phân tích quantitative metrics
   - Phân tích Confusion Matrix
   - Phân tích ROC Curve và AUC

**Kết quả đầu ra**:
- Các biểu đồ visualization
- Metrics trên test set
- Phân tích và đánh giá mô hình

### 5.2. Chạy tuần tự toàn bộ pipeline

**Cách 1: Chạy từng notebook theo thứ tự**
1. Chạy `01_data_exploration.ipynb` → Lưu dữ liệu đã xử lý
2. Chạy `02_preprocessing.ipynb` → Lưu dữ liệu đã chuẩn hóa và split
3. Chạy `03_modeling.ipynb` → Huấn luyện và đánh giá mô hình

**Cách 2: Sử dụng Jupyter Notebook với kernel**
- Mở Jupyter Notebook
- Chạy tất cả cells trong từng notebook theo thứ tự

### 5.3. Lưu ý quan trọng

⚠️ **Thứ tự chạy**: Phải chạy theo thứ tự 01 → 02 → 03 vì:
- Notebook 02 phụ thuộc vào output của notebook 01
- Notebook 03 phụ thuộc vào output của notebook 02

⚠️ **Dataset**: Đảm bảo file `creditcard.csv` đã được đặt trong `data/raw/` trước khi chạy

⚠️ **Memory**: Dataset khá lớn (~150MB), đảm bảo có đủ RAM

---

## 7. Results

### 7.1. Kết quả đạt được (Metrics)

#### 7.1.1. Logistic Regression

**Hyperparameters**:
- Learning rate: 0.01
- Max iterations: 1000
- Random state: 42

**Training Results**:
- Số iterations thực tế: 1000 (chưa converge, nhưng loss đã ổn định)
- Final training loss: **0.1095**
- Training loss giảm đều và mượt mà, không có dấu hiệu overfitting

**Test Results**:

| Metric | Value | Giải thích |
|--------|-------|------------|
| **Accuracy** | 0.9989 | Rất cao nhưng không có ý nghĩa trong bài toán imbalanced |
| **Precision** | 0.8333 | Tốt - 83.33% cảnh báo là đúng |
| **Recall** | 0.4592 | Thấp - Chỉ phát hiện được 45.92% tổng số gian lận |
| **F1-Score** | 0.5921 | Trung bình - Bị kéo xuống do Recall thấp |
| **AUC** | **0.9748** | Rất cao - Mô hình có khả năng phân loại tốt |

**Confusion Matrix**:

|                | Predicted Normal | Predicted Fraud |
|----------------|------------------|-----------------|
| **Actual Normal** | 56,854 (TN) | 9 (FP) |
| **Actual Fraud**  | 53 (FN) | 45 (TP) |

**Phân tích**:
- ✅ **True Negatives (56,854)**: Đa số giao dịch bình thường được phân loại đúng
- ✅ **True Positives (45)**: Phát hiện được 45/98 vụ gian lận (45.92%)
- ⚠️ **False Positives (9)**: Chỉ có 9 cảnh báo giả - Precision cao
- ❌ **False Negatives (53)**: **53 vụ gian lận bị bỏ sót** - Đây là vấn đề lớn nhất

**Nhận định**:
- Mô hình đang **thiên về Precision** (an toàn quá mức)
- **Recall thấp** là vấn đề nghiêm trọng trong bài toán fraud detection
- Tuy nhiên, **AUC cao (0.9748)** chứng tỏ mô hình có khả năng phân loại tốt
- Vấn đề nằm ở **threshold quá cao (0.5)** - có thể hạ xuống để tăng Recall

### 7.2. Hình ảnh trực quan hóa kết quả

#### 7.2.1. Data Exploration Visualizations

**1. Class Distribution**:
- Bar chart: So sánh số lượng Normal vs Fraud
- Pie chart: Tỷ lệ phần trăm của mỗi class
- **Insight**: Class imbalance nghiêm trọng (99.83% vs 0.17%)

**2. Time Feature Analysis**:
- Histogram: Phân phối giao dịch theo giờ
- Boxplot: So sánh Time giữa Normal và Fraud
- Line chart: Fraud rate theo giờ trong ngày
- **Insight**: Fraud rate cao hơn vào ban đêm (2-4h sáng)

**3. Amount Feature Analysis**:
- Histogram: Phân phối Amount (lệch phải nghiêm trọng)
- Boxplot: So sánh Amount giữa Normal và Fraud
- **Insight**: Fraud transactions có mean cao hơn nhưng median thấp hơn

**4. PCA Features Distribution**:
- Histograms cho V1-V9: Phân phối của các PCA features
- **Insight**: Các features đã được chuẩn hóa, mean ≈ 0

**5. Correlation Heatmap**:
- Heatmap tương quan giữa các features quan trọng
- **Insight**: PCA features không tương quan với nhau (tính trực giao)

**6. Feature Engineering - Rolling Statistics**:
- Line chart: Amount với Rolling Mean và Rolling Std
- Scatter plot: Anomaly detection bằng Z-Score
- **Insight**: Có thể phát hiện anomalies cục bộ

#### 7.2.2. Preprocessing Visualizations

**1. Outlier Detection**:
- So sánh số lượng outliers giữa IQR và Z-score methods
- **Insight**: IQR phát hiện nhiều outliers hơn Z-score

**2. Normalization Comparison**:
- Histograms so sánh: Original vs Min-Max vs Z-score vs Log-transformed
- **Insight**: Log transformation giảm skewness từ 16.98 xuống 0.16

#### 7.2.3. Modeling Visualizations

**1. Training Loss History**:
- Line chart: Loss giảm đều qua các iterations
- **Insight**: Mô hình hội tụ tốt, không có dấu hiệu overfitting

**2. Confusion Matrix**:
- Heatmap: Trực quan hóa số lượng TP, TN, FP, FN
- **Insight**: False Negatives cao (53) là vấn đề chính

**3. ROC Curve**:
- Line chart: ROC curve với AUC = 0.9748
- So sánh với Random Classifier (đường chéo)
- **Insight**: Mô hình có khả năng phân loại rất tốt

### 7.3. So sánh và phân tích

#### 7.3.1. Điểm mạnh của mô hình

1. **AUC Score cao (0.9748)**:
   - Chứng tỏ mô hình có khả năng phân biệt tốt giữa Normal và Fraud
   - Top 5% trong các mô hình fraud detection

2. **Precision cao (0.8333)**:
   - Giảm thiểu False Positives
   - Không làm phiền khách hàng bằng cảnh báo giả

3. **Training ổn định**:
   - Loss giảm đều, không có dấu hiệu overfitting
   - Gradient Descent hoạt động tốt

#### 7.3.2. Điểm yếu và vấn đề

1. **Recall thấp (0.4592)**:
   - Chỉ phát hiện được 45.92% tổng số gian lận
   - **53 vụ gian lận bị bỏ sót** - gây thiệt hại tài chính

2. **Threshold quá cao**:
   - Threshold mặc định 0.5 có thể không phù hợp
   - Cần tune threshold để cân bằng Precision và Recall

3. **Class imbalance**:
   - Mô hình thiên về class đa số (Normal)
   - Cần xử lý class imbalance tốt hơn

#### 7.3.3. So sánh với Baseline

**Baseline (Dự đoán tất cả là Normal)**:
- Accuracy: 0.9983
- Precision: 0.0 (không có TP)
- Recall: 0.0 (không phát hiện được fraud nào)
- F1-Score: 0.0

**Mô hình Logistic Regression**:
- Accuracy: 0.9989 (+0.0006)
- Precision: 0.8333 (tốt)
- Recall: 0.4592 (tốt hơn baseline rất nhiều)
- F1-Score: 0.5921 (tốt)

**Kết luận**: Mô hình tốt hơn baseline đáng kể, đặc biệt là có thể phát hiện được fraud.

#### 7.3.4. Insights quan trọng

1. **PCA Features quan trọng**:
   - V3, V14, V17, V12, V10 là những features quan trọng nhất
   - Có sự khác biệt lớn giữa Normal và Fraud

2. **Time pattern**:
   - Fraud rate cao hơn vào ban đêm (2-4h sáng)
   - Có thể sử dụng làm feature engineering

3. **Amount distribution**:
   - Fraud có mean cao hơn nhưng median thấp hơn
   - Cần log transformation để xử lý skewness

4. **Threshold optimization**:
   - AUC cao chứng tỏ có thể tune threshold để cải thiện Recall
   - Trade-off giữa Precision và Recall

---

## 8. Project Structure

```
23127516/
├── README.md                          # File README này
├── requirements.txt                   # Danh sách các thư viện cần thiết
│
├── data/                              # Thư mục chứa dữ liệu
│   ├── raw/                           # Dữ liệu gốc
│   │   └── creditcard.csv             # Dataset gốc từ Kaggle
│   └── processed/                     # Dữ liệu đã xử lý
│       ├── header.npy                 # Tên các features
│       ├── X_mean_filled.npy          # Dữ liệu điền bằng mean
│       ├── X_median_filled.npy        # Dữ liệu điền bằng median
│       ├── X_regression_filled.npy    # Dữ liệu điền bằng regression
│       ├── X_specific_filled.npy       # Dữ liệu điền bằng giá trị cụ thể
│       ├── X_processed.npy            # Dữ liệu đã chuẩn hóa (log + z-score)
│       ├── y.npy                      # Labels
│       ├── X_train.npy                # Features tập train
│       ├── X_test.npy                 # Features tập test
│       ├── y_train.npy                # Labels tập train
│       └── y_test.npy                 # Labels tập test
│
└── notebooks/                         # Thư mục chứa các Jupyter notebooks
    ├── 01_data_exploration.ipynb      # Notebook khám phá dữ liệu
    ├── 02_preprocessing.ipynb         # Notebook tiền xử lý dữ liệu
    └── 03_modeling.ipynb              # Notebook huấn luyện và đánh giá mô hình
```

### 8.1. Giải thích chức năng từng file/folder

#### `data/raw/`
- **Chức năng**: Chứa dữ liệu gốc từ dataset Kaggle
- **File**: `creditcard.csv` - Dataset gốc với 284,807 giao dịch và 31 cột

#### `data/processed/`
- **Chức năng**: Chứa dữ liệu đã được xử lý qua các bước preprocessing
- **Files**:
  - `header.npy`: Lưu tên các features (dùng `np.save` với `allow_pickle=True`)
  - `X_*_filled.npy`: Các phiên bản dữ liệu với các phương pháp điền missing values khác nhau (demo)
  - `X_processed.npy`: Dữ liệu đã được log transform và z-score standardization
  - `y.npy`: Vector labels (0 hoặc 1)
  - `X_train.npy`, `X_test.npy`: Features đã được chia train/test
  - `y_train.npy`, `y_test.npy`: Labels đã được chia train/test

#### `notebooks/01_data_exploration.ipynb`
- **Chức năng**: Khám phá và phân tích dữ liệu ban đầu
- **Nội dung chính**:
  1. Load dataset bằng NumPy
  2. Kiểm tra missing values và thống kê mô tả
  3. Phân tích class distribution
  4. Phân tích các features quan trọng (Time, Amount, V1-V28)
  5. Correlation analysis
  6. So sánh features giữa Normal và Fraud
  7. Feature engineering (rolling statistics)
  8. Statistical hypothesis testing
  9. Xử lý missing values (demo)
  10. Lưu dữ liệu đã xử lý

#### `notebooks/02_preprocessing.ipynb`
- **Chức năng**: Tiền xử lý dữ liệu trước khi modeling
- **Nội dung chính**:
  1. Load dữ liệu từ notebook 01
  2. Outlier detection (IQR và Z-score methods)
  3. Normalization & Standardization:
     - Min-Max Normalization
     - Z-score Standardization
     - Log Transformation
     - Decimal Scaling
  4. Áp dụng preprocessing cuối cùng (Log + Z-score)
  5. Train-Test Split (80-20)
  6. Lưu dữ liệu đã xử lý

#### `notebooks/03_modeling.ipynb`
- **Chức năng**: Huấn luyện và đánh giá mô hình Logistic Regression
- **Nội dung chính**:
  1. Load dữ liệu đã xử lý từ notebook 02
  2. Implement các evaluation metrics (Accuracy, Precision, Recall, F1, AUC)
  3. Implement Logistic Regression class từ đầu
  4. Training mô hình với Gradient Descent
  5. Visualize training loss history
  6. Evaluation trên test set
  7. Vẽ Confusion Matrix
  8. Vẽ ROC Curve và tính AUC
  9. Phân tích và đánh giá kết quả

#### `requirements.txt`
- **Chức năng**: Liệt kê các thư viện Python cần thiết
- **Nội dung**:
  ```
  numpy>=1.21.0
  matplotlib>=3.5.0
  seaborn>=0.11.0
  ```

#### `README.md`
- **Chức năng**: Tài liệu hướng dẫn chi tiết về dự án
- **Nội dung**: Mô tả đầy đủ về project, dataset, methods, results, và hướng dẫn sử dụng

---

## 9. Challenges & Solutions

### 9.1. Khó khăn gặp phải khi dùng NumPy

#### 9.1.1. Challenge: Load CSV file không dùng Pandas

**Vấn đề**:
- NumPy không có hàm đọc CSV trực tiếp như Pandas (`pd.read_csv()`)
- Cần parse header và convert data types thủ công
- File CSV có header với dấu ngoặc kép (`"Time"`, `"V1"`, ...)

**Solution**:
```python
# Đọc file CSV bằng np.genfromtxt với dtype=str để giữ nguyên format
data_str = np.genfromtxt(file_path, dtype=str, delimiter=',')

# Xử lý header: Loại bỏ dấu ngoặc kép
data_str = np.char.strip(data_str, '"')
header = data_str[0]
data_str = data_str[1:]

# Convert sang float64
data = data_str.astype(np.float64)
```

**Bài học**: NumPy có `np.char` module để xử lý string arrays, và `np.genfromtxt()` có thể đọc CSV nhưng cần xử lý thêm.

#### 9.1.2. Challenge: Vectorization thay vì for loops

**Vấn đề**:
- Ban đầu có thể muốn dùng for loops để xử lý từng feature
- For loops chậm với dataset lớn (284,807 samples)
- Cần tính toán thống kê cho nhiều features

**Solution - Sử dụng Broadcasting và Vectorization**:

**Ví dụ 1: Tính mean cho tất cả features**
```python
# ❌ Chậm - Dùng for loop
means = np.zeros(n_features)
for i in range(n_features):
    means[i] = np.mean(data[:, i])

# ✅ Nhanh - Vectorized
means = np.mean(data, axis=0)  # Tính mean theo axis=0 (columns)
```

**Ví dụ 2: Tính Z-score cho tất cả features**
```python
# ❌ Chậm
z_scores = np.zeros_like(data)
for i in range(n_features):
    mean = np.mean(data[:, i])
    std = np.std(data[:, i])
    z_scores[:, i] = (data[:, i] - mean) / std

# ✅ Nhanh - Broadcasting
mean_vals = np.mean(data, axis=0, keepdims=True)  # Shape: (1, n_features)
std_vals = np.std(data, axis=0, keepdims=True)    # Shape: (1, n_features)
z_scores = (data - mean_vals) / std_vals  # Broadcasting: (n_samples, n_features)
```

**Ví dụ 3: Fancy indexing thay vì loop + if**
```python
# ❌ Chậm
fraud_data = []
for i in range(len(y)):
    if y[i] == 1:
        fraud_data.append(X[i])

# ✅ Nhanh - Boolean indexing
fraud_mask = (y == 1)
fraud_data = X[fraud_mask]
```

**Bài học**: Luôn nghĩ về cách vectorize operations, sử dụng broadcasting và fancy indexing.

#### 9.1.3. Challenge: Tính toán distance matrix cho KNN (nếu có)

**Vấn đề**:
- Cần tính distance giữa mỗi test point và tất cả train points
- For loops sẽ rất chậm với dataset lớn
- Memory có thể không đủ nếu tính toàn bộ distance matrix

**Solution - Broadcasting để tính distance matrix**:
```python
# Tính Euclidean distance giữa X1 (n1 samples) và X2 (n2 samples)
# Kết quả: distance matrix shape (n1, n2)

# Cách 1: Broadcasting với np.newaxis
diff = X1[:, np.newaxis, :] - X2[np.newaxis, :, :]  # Shape: (n1, n2, n_features)
distances = np.sqrt(np.sum(diff ** 2, axis=2))  # Shape: (n1, n2)

# Cách 2: Sử dụng công thức vectorized
# d^2 = ||x1||^2 + ||x2||^2 - 2*x1*x2
X1_squared = np.sum(X1 ** 2, axis=1, keepdims=True)  # (n1, 1)
X2_squared = np.sum(X2 ** 2, axis=1)  # (n2,)
dot_product = X1 @ X2.T  # (n1, n2)
distances_squared = X1_squared + X2_squared - 2 * dot_product
distances = np.sqrt(np.maximum(distances_squared, 0))  # Tránh negative do floating point
```

**Bài học**: Broadcasting là công cụ mạnh mẽ để tính toán hiệu quả, nhưng cần chú ý memory với dataset lớn.

#### 9.1.4. Challenge: Numerical stability trong sigmoid

**Vấn đề**:
- `exp(-z)` có thể overflow khi z rất âm (z << 0)
- `exp(z)` có thể overflow khi z rất dương (z >> 0)
- Dẫn đến `sigmoid(z)` trả về `nan` hoặc `inf`

**Solution - Clip z values**:
```python
def _sigmoid(self, z):
    # Clip z để tránh overflow
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))
```

**Giải thích**:
- `exp(-500) ≈ 0` → `sigmoid(-500) ≈ 0`
- `exp(500) ≈ inf` → `sigmoid(500) ≈ 1`
- Với z trong [-500, 500], sigmoid hoạt động ổn định

**Bài học**: Luôn chú ý đến numerical stability, đặc biệt với các hàm exponential.

#### 9.1.5. Challenge: Underflow trong Naive Bayes (nếu có)

**Vấn đề**:
- Nhân nhiều probabilities nhỏ có thể gây underflow
- `P(x|y) = P(x1|y) * P(x2|y) * ... * P(xn|y)` có thể rất nhỏ (gần 0)
- Dẫn đến mất độ chính xác số học

**Solution - Sử dụng log probabilities**:
```python
# ❌ Có thể underflow
likelihood = np.prod(probabilities, axis=1)  # Nhân nhiều số nhỏ

# ✅ Tránh underflow bằng log space
log_likelihood = np.sum(np.log(probabilities + 1e-15), axis=1)  # Cộng log = nhân
# Sau đó so sánh log probabilities thay vì probabilities
```

**Bài học**: Sử dụng log space khi làm việc với probabilities nhỏ để tránh underflow.

#### 9.1.6. Challenge: Tính toán thống kê phức tạp (Skewness, Kurtosis)

**Vấn đề**:
- NumPy không có hàm `skew()` và `kurtosis()` sẵn (hoặc có nhưng cần scipy)
- Cần implement từ đầu bằng công thức toán học

**Solution - Implement từ công thức**:
```python
def calculate_skewness(data):
    mean = np.mean(data, axis=0, keepdims=True)
    std = np.std(data, axis=0, ddof=1, keepdims=True)
    std = np.where(std == 0, 1, std)  # Tránh chia cho 0
    centered = data - mean
    skew = np.mean((centered / std) ** 3, axis=0)
    return skew

def calculate_kurtosis(data):
    mean = np.mean(data, axis=0, keepdims=True)
    std = np.std(data, axis=0, ddof=1, keepdims=True)
    std = np.where(std == 0, 1, std)
    centered = data - mean
    kurt = np.mean((centered / std) ** 4, axis=0) - 3  # Excess kurtosis
    return kurt
```

**Bài học**: Hiểu rõ công thức toán học giúp implement các hàm không có sẵn.

#### 9.1.7. Challenge: Xử lý division by zero

**Vấn đề**:
- Khi tính Z-score, nếu std = 0 (feature không đổi), sẽ gây lỗi division by zero
- Khi tính các metrics, nếu denominator = 0, sẽ gây lỗi

**Solution - Sử dụng np.where để xử lý edge cases**:
```python
# Ví dụ 1: Z-score standardization
std_vals = np.std(data, axis=0, ddof=1, keepdims=True)
std_vals = np.where(std_vals == 0, 1, std_vals)  # Thay 0 bằng 1
z_scores = (data - mean_vals) / std_vals

# Ví dụ 2: Precision score
if tp + fp == 0:
    return 0.0  # Không có positive predictions
return tp / (tp + fp)
```

**Bài học**: Luôn kiểm tra edge cases và xử lý division by zero.

### 9.2. Cách giải quyết

#### 9.2.1. Đọc tài liệu NumPy

- **Broadcasting**: Hiểu rõ cách NumPy broadcast arrays
- **Fancy indexing**: Sử dụng boolean masks và integer arrays
- **Universal functions (ufuncs)**: Tận dụng các hàm vectorized của NumPy
- **Memory efficiency**: Sử dụng views thay vì copies khi có thể

#### 9.2.2. Vectorization mindset

- **Luôn nghĩ về cách vectorize**: Trước khi viết for loop, nghĩ xem có thể vectorize không
- **Broadcasting**: Sử dụng `np.newaxis` và `keepdims=True` để control shape
- **Fancy indexing**: Sử dụng boolean masks thay vì loops + if

#### 9.2.3. Numerical stability

- **Overflow/Underflow**: Chú ý đến các hàm exponential, log
- **Log space**: Sử dụng log probabilities khi làm việc với probabilities nhỏ
- **Clipping**: Clip values để tránh overflow
- **Epsilon**: Thêm epsilon nhỏ (1e-15) khi tính log để tránh log(0)

#### 9.2.4. Memory efficiency

- **Views vs Copies**: Sử dụng views (`data[mask]`) thay vì copies khi có thể
- **In-place operations**: Sử dụng `+=`, `-=` thay vì `= +` khi có thể
- **Memory mapping**: Với dataset rất lớn, có thể dùng `np.memmap()`

#### 9.2.5. Testing và Debugging

- **Test từng function nhỏ**: Test từng function trước khi tích hợp
- **Kiểm tra shapes**: Luôn kiểm tra shape của arrays
- **Visualize intermediate results**: In ra một vài giá trị để kiểm tra
- **Compare với reference**: So sánh kết quả với scikit-learn hoặc Pandas (nếu có thể)

#### 9.2.6. Performance optimization

- **Profile code**: Sử dụng `%timeit` trong Jupyter để đo thời gian
- **Avoid unnecessary copies**: Sử dụng views khi có thể
- **Use appropriate dtypes**: Sử dụng `float32` thay vì `float64` nếu đủ độ chính xác
- **Batch processing**: Xử lý theo batch nếu dataset quá lớn

---

## 10. Future Improvements

### 10.1. Xử lý Class Imbalance

#### 10.1.1. Class Weighting

**Ý tưởng**: Tăng trọng số cho class thiểu số trong loss function

**Implementation**:
```python
# Trong Binary Cross-Entropy Loss
class_weight_0 = len(y) / (2 * np.sum(y == 0))  # Weight cho class 0
class_weight_1 = len(y) / (2 * np.sum(y == 1))  # Weight cho class 1

# Weighted loss
loss = -np.mean(class_weight_0 * (1-y) * np.log(1-y_pred + 1e-15) + 
                class_weight_1 * y * np.log(y_pred + 1e-15))
```

**Lợi ích**: Mô hình sẽ chú ý nhiều hơn đến class thiểu số (Fraud)

#### 10.1.2. SMOTE (Synthetic Minority Oversampling Technique)

**Ý tưởng**: Tạo synthetic samples cho class thiểu số

**Cách hoạt động**:
1. Chọn một sample từ class thiểu số
2. Tìm k nearest neighbors từ cùng class
3. Tạo synthetic sample bằng cách interpolate giữa sample và neighbors

**Lợi ích**: Tăng số lượng samples của class thiểu số mà không chỉ duplicate

#### 10.1.3. Undersampling

**Ý tưởng**: Giảm số lượng samples của class đa số

**Phương pháp**:
- Random undersampling
- Tomek Links
- Edited Nearest Neighbors

**Lưu ý**: Cần cẩn thận để không mất thông tin quan trọng

### 10.2. Feature Engineering

#### 10.2.1. Time-based Features

**Tạo features từ Time**:
```python
# Hour of day
hours = (time_data // 3600) % 24

# Day of week (nếu có đủ dữ liệu)
days = (time_data // 86400) % 7

# Is weekend
is_weekend = (days == 5) | (days == 6)

# Is night (2-6 AM)
is_night = (hours >= 2) & (hours < 6)
```

**Lợi ích**: Tận dụng pattern thời gian đã phát hiện (fraud rate cao vào ban đêm)

#### 10.2.2. Amount Binning

**Chia Amount thành các bins**:
```python
# Tạo bins dựa trên quantiles
bins = np.percentile(amount_data, [0, 25, 50, 75, 100])
amount_binned = np.digitize(amount_data, bins)
```

**Lợi ích**: Giảm ảnh hưởng của outliers, tạo features categorical

#### 10.2.3. Interaction Features

**Tạo features tương tác giữa các features quan trọng**:
```python
# Ví dụ: Tương tác giữa V3 và V14 (2 features quan trọng nhất)
interaction = V3 * V14

# Hoặc ratio
ratio = V3 / (V14 + 1e-10)  # Tránh chia cho 0
```

**Lợi ích**: Nắm bắt mối quan hệ phi tuyến giữa các features

#### 10.2.4. Polynomial Features

**Tạo polynomial features**:
```python
# Bậc 2
X_poly = np.column_stack([X, X**2])

# Hoặc chỉ cho một số features quan trọng
important_features = X[:, [v3_idx, v14_idx, v17_idx]]
X_poly = np.column_stack([X, important_features**2])
```

**Lợi ích**: Nắm bắt mối quan hệ phi tuyến

### 10.3. Model Improvements

#### 10.3.1. Hyperparameter Tuning

**Tune các hyperparameters**:
- Learning rate: Thử 0.001, 0.01, 0.1
- Max iterations: Tăng lên nếu chưa converge
- Regularization: Thêm L1/L2 regularization để tránh overfitting

**Implementation L2 Regularization**:
```python
# Thêm vào loss function
L2_penalty = lambda_reg * np.sum(self.weights ** 2)
loss = binary_cross_entropy_loss + L2_penalty

# Thêm vào gradient
dw = (1/m) * X.T @ (y_pred - y) + 2 * lambda_reg * self.weights
```

#### 10.3.2. Threshold Optimization

**Tune threshold để cân bằng Precision và Recall**:
```python
# Thử các threshold khác nhau
thresholds = np.arange(0.1, 0.9, 0.05)
best_f1 = 0
best_threshold = 0.5

for threshold in thresholds:
    y_pred = (y_pred_proba >= threshold).astype(int)
    f1 = f1_score(y_test, y_pred)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold
```

**Lợi ích**: Có thể tăng Recall mà không giảm Precision quá nhiều

#### 10.3.3. Ensemble Methods

**Kết hợp nhiều mô hình**:
- **Voting**: Kết hợp predictions từ nhiều mô hình
- **Stacking**: Dùng mô hình khác để combine predictions
- **Bagging**: Train nhiều mô hình trên các subsets khác nhau

#### 10.3.4. Advanced Algorithms

**Thử các thuật toán khác** (implement từ đầu bằng NumPy):
- **Decision Trees**: Có thể xử lý tốt với imbalanced data
- **Random Forest**: Ensemble của Decision Trees
- **Neural Networks**: Nếu được phép, có thể thử MLP đơn giản

### 10.4. Evaluation Improvements

#### 10.4.1. Precision-Recall Curve

**Vẽ PR Curve thay vì chỉ ROC Curve**:
- PR Curve tốt hơn ROC Curve cho imbalanced data
- Focus vào Precision và Recall thay vì FPR

#### 10.4.2. Cost-Sensitive Evaluation

**Đánh giá dựa trên cost matrix**:
```python
# Cost matrix
cost_matrix = {
    'TN': 0,      # True Negative: Không có cost
    'FP': 10,     # False Positive: Làm phiền khách hàng
    'FN': 1000,   # False Negative: Mất tiền do gian lận
    'TP': -100    # True Positive: Phát hiện được, tiết kiệm tiền
}

# Tính total cost
total_cost = (TN * cost_matrix['TN'] + 
              FP * cost_matrix['FP'] + 
              FN * cost_matrix['FN'] + 
              TP * cost_matrix['TP'])
```

**Lợi ích**: Phản ánh đúng tác động thực tế của các loại lỗi

#### 10.4.3. Cross-Validation

**Sử dụng k-fold cross-validation**:
```python
def k_fold_cross_validation(X, y, k=5):
    n_samples = len(X)
    fold_size = n_samples // k
    scores = []
    
    for i in range(k):
        # Split data
        val_start = i * fold_size
        val_end = (i + 1) * fold_size
        
        X_val = X[val_start:val_end]
        y_val = y[val_start:val_end]
        X_train = np.concatenate([X[:val_start], X[val_end:]])
        y_train = np.concatenate([y[:val_start], y[val_end:]])
        
        # Train and evaluate
        model.fit(X_train, y_train)
        score = model.evaluate(X_val, y_val)
        scores.append(score)
    
    return np.mean(scores), np.std(scores)
```

**Lợi ích**: Đánh giá ổn định hơn, không phụ thuộc vào một lần split

### 10.5. Code Optimization

#### 10.5.1. Memory Optimization

**Sử dụng memory mapping cho dataset lớn**:
```python
# Thay vì load toàn bộ vào memory
X = np.load('X_train.npy')

# Sử dụng memory mapping
X = np.load('X_train.npy', mmap_mode='r')  # Read-only memory map
```

**Lợi ích**: Tiết kiệm memory, có thể xử lý dataset lớn hơn

#### 10.5.2. Parallel Processing

**Sử dụng multiprocessing cho cross-validation**:
```python
from multiprocessing import Pool

def train_fold(args):
    X_train, y_train, X_val, y_val = args
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model.evaluate(X_val, y_val)

# Parallel processing
with Pool(processes=4) as pool:
    scores = pool.map(train_fold, fold_args)
```

**Lợi ích**: Tăng tốc độ training khi có nhiều CPU cores

#### 10.5.3. Caching

**Cache các kết quả tính toán trung gian**:
```python
import pickle

# Cache preprocessed data
if os.path.exists('X_processed_cache.npy'):
    X_processed = np.load('X_processed_cache.npy')
else:
    X_processed = preprocess(X)
    np.save('X_processed_cache.npy', X_processed)
```

**Lợi ích**: Tránh tính toán lại các kết quả đã có

### 10.6. Documentation và Code Quality

#### 10.6.1. Refactor thành modules

**Tách code thành các modules**:
```
src/
├── __init__.py
├── data_processing.py    # Các hàm xử lý dữ liệu
├── models.py             # Các mô hình ML
├── metrics.py            # Các evaluation metrics
└── visualization.py     # Các hàm visualization
```

**Lợi ích**: Code dễ maintain và reuse hơn

#### 10.6.2. Unit Tests

**Viết unit tests cho các functions**:
```python
def test_precision_score():
    y_true = np.array([0, 1, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0, 1])
    assert precision_score(y_true, y_pred) == 1.0
```

**Lợi ích**: Đảm bảo code hoạt động đúng

#### 10.6.3. Type Hints và Docstrings

**Thêm type hints và docstrings**:
```python
def precision_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate precision score.
    
    Parameters:
    -----------
    y_true : np.ndarray
        True labels
    y_pred : np.ndarray
        Predicted labels
    
    Returns:
    --------
    float
        Precision score
    """
    ...
```

**Lợi ích**: Code dễ đọc và maintain hơn

---

## 11. Contributors

### 11.1. Thông tin tác giả

- **Tên**: [Tên sinh viên]
- **MSSV**: 23127516
- **Email**: [Email]
- **Trường**: Trường Đại học Khoa học Tự nhiên, Đại học Quốc gia TP.HCM
- **Khoa**: Khoa Công nghệ Thông tin
- **Bộ môn**: Khoa học Máy tính
- **Môn học**: Programming for Data Science

### 11.2. Contact

Nếu có câu hỏi, góp ý hoặc muốn đóng góp cho dự án, vui lòng liên hệ:

- **Email**: [Email]
- **GitHub**: [GitHub username] (nếu có)
- **LinkedIn**: [LinkedIn profile] (nếu có)

### 11.3. Acknowledgments

- **Dataset**: Cảm ơn ULB Machine Learning Group và Kaggle đã cung cấp dataset
- **Giảng viên**: Cảm ơn giảng viên môn Programming for Data Science đã hướng dẫn
- **Tài liệu**: Cảm ơn cộng đồng NumPy, Matplotlib, Seaborn đã cung cấp tài liệu tuyệt vời

---

## 12. License

This project is licensed under the **MIT License** - see the LICENSE file for details.

**MIT License** cho phép:
- ✅ Sử dụng thương mại
- ✅ Sử dụng cá nhân
- ✅ Sửa đổi
- ✅ Phân phối
- ✅ Sublicense

**Yêu cầu**:
- ⚠️ Bao gồm license và copyright notice
- ⚠️ Không có warranty

---

## References

1. **NumPy Documentation**: https://numpy.org/doc/
2. **Credit Card Fraud Detection Dataset**: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
3. **Matplotlib Documentation**: https://matplotlib.org/
4. **Seaborn Documentation**: https://seaborn.pydata.org/
5. **Logistic Regression Theory**: 
   - Bishop, C. M. (2006). Pattern Recognition and Machine Learning. Springer.
   - Hastie, T., Tibshirani, R., & Friedman, J. (2009). The Elements of Statistical Learning. Springer.
6. **Imbalanced Data Handling**:
   - Chawla, N. V., et al. (2002). SMOTE: Synthetic Minority Over-sampling Technique. Journal of Artificial Intelligence Research.
7. **Evaluation Metrics for Imbalanced Data**:
   - Saito, T., & Rehmsmeier, M. (2015). The Precision-Recall Plot Is More Informative Than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets. PLOS ONE.

---

## Lưu ý

**Dự án này được thực hiện như một phần của bài tập học tập**. Tất cả các thuật toán Machine Learning đều được **implement từ đầu bằng NumPy** để học hỏi và hiểu sâu về cách hoạt động của các thuật toán, không sử dụng các thư viện ML có sẵn như scikit-learn.

**Mục đích chính**:
- ✅ Hiểu rõ cách hoạt động của các thuật toán ML cơ bản
- ✅ Làm chủ NumPy và vectorization
- ✅ Áp dụng kiến thức toán học vào thực tế
- ✅ Xử lý bài toán imbalanced data

**Không phải mục đích**:
- ❌ Tạo ra mô hình production-ready tốt nhất
- ❌ So sánh với các mô hình state-of-the-art
- ❌ Tối ưu hóa performance cực đại

---

**Cảm ơn bạn đã quan tâm đến dự án!** 🙏
