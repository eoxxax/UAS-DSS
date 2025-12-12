import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')

# ==================== CONFIG ====================
st.set_page_config(
    page_title="Customer Churn Prediction DSS",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== STYLING ====================
st.markdown("""
<style>
    .main-header {
        font-size: 2.8rem;
        color: #1f77b4;
        text-align: center;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.3rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .insight-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# ==================== SIDEBAR ====================
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/user-shield.png", width=80)
    
    st.markdown("### Anggota Tim")
    st.markdown("""
    - Azmi Naifah Iftinah (013)
    - Audrey Shaina Tjandra (026)
    - Siti Nailah Eko P. A. (059)
    """)
    
    st.markdown("---")
    
    st.markdown("### Tentang Aplikasi")
    st.info("""
    Decision Support System untuk memprediksi pelanggan yang berisiko churn 
    dan memberikan rekomendasi retensi berbasis data.
    
    Model: Random Forest Classifier  
    Accuracy: ~80%  
    Dataset: Telco Customer Churn
    """)
    
    st.markdown("---")
    
    st.markdown("### Fitur Utama")
    st.markdown("""
    - Prediksi Churn Individual
    - What-If Analysis
    - Batch Prediction
    - Rekomendasi Bisnis
    - Financial Impact Analysis
    """)

# ==================== HEADER ====================
st.markdown('<p class="main-header">Customer Churn Prediction DSS</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Sistem Pendukung Keputusan untuk Prediksi & Strategi Churn Pelanggan</p>', unsafe_allow_html=True)

# ==================== FUNCTIONS ====================

@st.cache_data
def load_and_train_model():
    """Load data and train model"""
    try:
        # Load data dari folder
        df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
        
        # Preprocessing
        data = df.copy()
        
        if 'customerID' in data.columns:
            data = data.drop('customerID', axis=1)
        
        # Fix TotalCharges
        data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
        data['TotalCharges'].fillna(data['TotalCharges'].median(), inplace=True)
        
        # Binary encoding
        binary_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
        for col in binary_cols:
            if col in data.columns:
                le = LabelEncoder()
                data[col] = le.fit_transform(data[col].astype(str))
        
        # SeniorCitizen
        if data['SeniorCitizen'].dtype == 'object':
            data['SeniorCitizen'] = data['SeniorCitizen'].map({'Yes': 1, 'No': 0})
        
        # Target
        data['Churn'] = data['Churn'].map({'Yes': 1, 'No': 0})
        
        # One-hot encoding
        categorical_cols = ['MultipleLines', 'InternetService', 'OnlineSecurity', 
                           'OnlineBackup', 'DeviceProtection', 'TechSupport',
                           'StreamingTV', 'StreamingMovies', 'Contract', 'PaymentMethod']
        
        for col in categorical_cols:
            if col in data.columns:
                data = pd.get_dummies(data, columns=[col], prefix=col, drop_first=True)
        
        # Split
        X = data.drop('Churn', axis=1)
        y = data['Churn']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train model dengan hyperparameter yang lebih sensitif untuk high risk
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=12,  # Lebih shallow untuk lebih sensitif
            min_samples_split=10,
            min_samples_leaf=5,
            class_weight='balanced',  # PENTING: balance class weights
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred)
        }
        
        return model, X.columns.tolist(), metrics, df
        
    except FileNotFoundError:
        st.error("File 'WA_Fn-UseC_-Telco-Customer-Churn.csv' tidak ditemukan!")
        st.info("Pastikan file dataset ada di folder yang sama dengan script ini.")
        return None, None, None, None

def preprocess_batch(df, feature_names):
    """Preprocess batch data - FAST & VECTORIZED"""

    data = df.copy()

    # Drop kolom tidak terpakai
    if "customerID" in data.columns:
        data = data.drop("customerID", axis=1)
    if "Churn" in data.columns:
        data = data.drop("Churn", axis=1)

    # Kolom numerik
    numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'SeniorCitizen']
    for col in numeric_cols:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")
            data[col].fillna(0, inplace=True)

    # Mengisi nilai untuk kolom kategorikal
    categorical_cols = [
        'gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling',
        'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
        'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
        'Contract', 'PaymentMethod'
    ]

    for col in categorical_cols:
        if col in data.columns:
            data[col] = data[col].astype(str).replace("", "No").fillna("No")

    # Label encoding
    binary_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
    for col in binary_cols:
        if col in data.columns:
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col])

    # One-hot encoding
    onehot_cols = [
        'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
        'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
        'Contract', 'PaymentMethod'
    ]
    for col in onehot_cols:
        if col in data.columns:
            data = pd.get_dummies(data, columns=[col], prefix=col, drop_first=True)

    # Menambahkan missing kolom
    for col in feature_names:
        if col not in data.columns:
            data[col] = 0

    # Reorder
    data = data[feature_names]

    return data

def preprocess_input(input_dict, feature_names):
    """Preprocess single input untuk prediction"""
    
    # Create dataframe
    df = pd.DataFrame([input_dict])
    
    # Bersihkan kolom numerik (string kosong dan missing values)
    numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    for col in numeric_cols:
        if col in df.columns:
            # Mengubah ke numerik
            df[col] = pd.to_numeric(df[col], errors='coerce')
            # Isi nilai dengan median atau nol
            if df[col].isna().any():
                df[col].fillna(0, inplace=True)
    
    # Memastikan SeniorCitizen numerik
    if 'SeniorCitizen' in df.columns:
        if df['SeniorCitizen'].dtype == 'object':
            df['SeniorCitizen'] = pd.to_numeric(df['SeniorCitizen'], errors='coerce')
            df['SeniorCitizen'].fillna(0, inplace=True)
    
    # Binary encoding
    binary_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
    for col in binary_cols:
        if col in df.columns:
            # Tangani string kosong
            df[col] = df[col].fillna('No').astype(str)
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
    
    # One-hot encoding
    categorical_cols = ['MultipleLines', 'InternetService', 'OnlineSecurity', 
                       'OnlineBackup', 'DeviceProtection', 'TechSupport',
                       'StreamingTV', 'StreamingMovies', 'Contract', 'PaymentMethod']
    
    for col in categorical_cols:
        if col in df.columns:
            # Tangani string kosong dan nilai NaN
            df[col] = df[col].fillna('No').astype(str)
            df = pd.get_dummies(df, columns=[col], prefix=col, drop_first=True)
    
    # Sesuaikan dengan fitur yang digunakan saat training
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0
    
    # Reorder kolom
    df = df[feature_names]
    
    # Memastikan semua nilai numerik
    df = df.apply(pd.to_numeric, errors='coerce').fillna(0)
    
    return df

def get_risk_level(probability):
    """Menentukan risk level"""
    if probability >= 0.6:
        return "HIGH RISK", "red"
    elif probability >= 0.35:
        return "MEDIUM RISK", "orange"
    else:
        return "LOW RISK", "green"

def analyze_churn_factors(input_dict, probability):
    """Analisis Faktor Penting key Churn"""
    factors = []
    
    # Contract
    if input_dict['Contract'] == "Month-to-month":
        factors.append({
            'factor': 'Month-to-Month Contract',
            'impact': 'HIGH',
            'detail': 'No commitment = 5x higher churn risk'
        })
    
    # Payment method
    if input_dict['PaymentMethod'] == "Electronic check":
        factors.append({
            'factor': 'Electronic Check Payment',
            'impact': 'HIGH',
            'detail': 'Inconvenient payment method increases churn by 45%'
        })
    
    # Tenure
    tenure = input_dict['tenure']
    if tenure < 6:
        factors.append({
            'factor': f'Very New Customer ({tenure} months)',
            'impact': 'CRITICAL',
            'detail': 'First 6 months = highest churn period (50%+ risk)'
        })
    elif tenure < 12:
        factors.append({
            'factor': f'New Customer ({tenure} months)',
            'impact': 'HIGH',
            'detail': 'Still in critical period (35%+ risk)'
        })
    
    # Charges
    monthly = input_dict['MonthlyCharges']
    if monthly > 80:
        factors.append({
            'factor': f'High Monthly Cost (${monthly:.2f})',
            'impact': 'HIGH',
            'detail': 'Premium pricing increases price sensitivity'
        })
    
    # Services
    if input_dict['OnlineSecurity'] == "No":
        factors.append({
            'factor': 'No Online Security',
            'impact': 'MEDIUM',
            'detail': 'Missing value-added service'
        })
    
    if input_dict['TechSupport'] == "No":
        factors.append({
            'factor': '🛠 No Tech Support',
            'impact': 'MEDIUM',
            'detail': 'Lack of support increases frustration'
        })
    
    # Internet + Fiber
    if input_dict['InternetService'] == "Fiber optic":
        if input_dict['OnlineSecurity'] == "No" and input_dict['TechSupport'] == "No":
            factors.append({
                'factor': 'Fiber Optic + No Services',
                'impact': 'HIGH',
                'detail': 'Paying premium price with minimal benefits'
            })
    
    # Senior citizen
    if input_dict['SeniorCitizen'] == "Yes":
        factors.append({
            'factor': 'Senior Citizen',
            'impact': 'MEDIUM',
            'detail': 'Higher sensitivity to service issues'
        })
    
    return factors

def generate_recommendations(input_dict, probability, factors):
    """Generate actionable recommendations"""
    recommendations = []
    
    monthly = input_dict['MonthlyCharges']
    tenure = input_dict['tenure']
    
    # Priority 1: Contract upgrade
    if input_dict['Contract'] == "Month-to-month":
        discount_1yr = monthly * 0.20 * 12
        discount_2yr = monthly * 0.25 * 24
        recommendations.append({
            'priority': 'CRITICAL',
            'action': 'Kampanye Upgrade Kontrak',
            'detail': f"""
        - Tawarkan kontrak 1 tahun dengan diskon 20% (Hemat ${discount_1yr:.0f}/tahun)
        - Tawarkan kontrak 2 tahun dengan diskon 25% (Hemat ${discount_2yr:.0f} selama 2 tahun)
        - Berikan bonus loyalitas: 3 bulan pertama layanan streaming GRATIS (nilai $30)
        - Target tingkat keberhasilan: 60-70%
            """,
            'impact': f'High - Mengunci pelanggan, menurunkan churn hingga 65%',
            'cost': f'${monthly * 2:.0f} (biaya diskon)',
            'roi': f'${monthly * 24:.0f} (nilai retensi untuk 24 bulan)'
        })
    
    # Priority 2: Payment method
    if input_dict['PaymentMethod'] == "Electronic check":
        recommendations.append({
            'priority': 'HIGH',
            'action': 'Migrasi ke Auto-Pay',
            'detail': f"""
        - Segera: Berikan kredit **$50** untuk pelanggan yang berpindah ke auto-pay
        - Berkelanjutan: Diskon **$5/bulan** untuk pembayaran via kartu kredit/transfer bank
        - Bonus: Bebaskan seluruh denda keterlambatan selama 12 bulan
        - Komunikasi: Telepon personal dari account manager
            """,
            'impact': 'Medium-High - Mengurangi churn sebesar 30-40%',
            'cost': f'${110:.0f}/tahun (insentif)',
            'roi': f'${monthly * 12:.0f} (nilai retensi tahunan)'
        })
    
    # Priority 3: New customer retention
    if tenure < 12:
        recommendations.append({
            'priority': 'CRITICAL',
            'action': 'Perawatan Intensif untuk Pelanggan Baru',
            'detail': f"""
        - Minggu 1-4: Pemeriksaan mingguan melalui panggilan dari account manager khusus
        - Bulan 2-6: Survei kepuasan dua minggu sekali dengan penyelesaian masalah secara langsung
        - Insentif: Kredit loyalitas sebesar **${tenure * 10}** (bertambah seiring lamanya berlangganan)
        - Bonus: Insentif upgrade kontrak awal (diskon tambahan 10%)
        - Dukungan: Jalur dukungan teknis prioritas
            """,
            'impact': 'Critical - Mengurangi churn pelanggan baru hingga 45%',
            'cost': f'${200:.0f} (biaya program)',
            'roi': f'${monthly * 36:.0f} (LTV 3 tahun jika pelanggan bertahan)'
        })
    
    # Priority 4: Price optimization
    if monthly > 70:
        new_price = monthly * 0.85
        savings = monthly - new_price
        recommendations.append({
            'priority': 'HIGH',
            'action': 'Tinjauan Harga yang Dipersonalisasi',
            'detail': f"""
        - Saat ini: ${monthly:.2f}/bulan
        - Usulan: ${new_price:.2f}/bulan (pengurangan 15%)
        - Penghematan: ${savings:.2f}/bulan = **${savings * 12:.0f}/tahun
        - Alternatif: Harga tetap, tetapi tambahkan 2-3 layanan premium GRATIS
        - Alasan: Diskon loyalitas setelah {tenure} bulan berlangganan
            """,
            'impact': 'High - Mengurangi churn yang sensitif terhadap harga hingga 40%',
            'cost': f'${savings * 12:.0f}/tahun (pengurangan pendapatan)',
            'roi': f'Lebih baik daripada kehilangan ${monthly * 12:.0f}/tahun sepenuhnya'
        })
    
    # Priority 5: Service enhancement
    if input_dict['OnlineSecurity'] == "No" or input_dict['TechSupport'] == "No":
        recommendations.append({
            'priority': 'MEDIUM',
            'action': 'Paket Layanan Bernilai Tambah',
            'detail': f"""
        - Penawaran: 3 bulan **GRATIS Online Security + Tech Support
        - Nilai: $30/bulan = **$90 total
        - Setelah masa trial: Lanjutkan dengan diskon 50% selama 6 bulan berikutnya
        - Komunikasi: "Terima kasih telah menjadi pelanggan yang berharga bagi kami"
        - Tujuan: Meningkatkan persepsi nilai dan biaya perpindahan (switching cost)
            """,
            'impact': 'Medium - Mengurangi churn sebesar 25-30%',
            'cost': '$90 (biaya trial)',
            'roi': f'${monthly * 12:.0f} (nilai retensi tahunan)'
        })
    
    # Financial summary
    total_cost = 0
    total_benefit = monthly * 24  
    
    if input_dict['Contract'] == "Month-to-month":
        total_cost += monthly * 2
    if input_dict['PaymentMethod'] == "Electronic check":
        total_cost += 110
    if tenure < 12:
        total_cost += 200
    
    roi = ((total_benefit - total_cost) / total_cost * 100) if total_cost > 0 else 0
    
    recommendations.append({
        'priority': 'FINANCIAL SUMMARY',
        'action': 'Investasi vs. Keuntungan',
        'detail': f"""
    Total Investasi yang Diperlukan: ${total_cost:.0f}
    LTV Pelanggan 2 Tahun: ${total_benefit:.0f}
    Keuntungan Bersih: ${total_benefit - total_cost:.0f}
    ROI: {roi:.0f}%

    Keputusan: {"SANGAT DIREKOMENDASIKAN untuk berinvestasi pada retensi" if roi > 200 else "⚠ Pantau dan optimalkan pengeluaran retensi"}
        """,
        'impact': f'Perkiraan tingkat retensi: 70-75%',
        'cost': f'${total_cost:.0f}',
        'roi': f'{roi:.0f}% pengembalian dari investasi retensi'
    })

    
    return recommendations

# ==================== LOAD MODEL ====================
with st.spinner(" Loading model..."):
    model, feature_names, metrics, original_data = load_and_train_model()

if model is None:
    st.stop()

st.success(f"Model loaded! Accuracy: {metrics['accuracy']:.1%} | Recall: {metrics['recall']:.1%}")

# ==================== MAIN TABS ====================
tab1, tab2, tab3 = st.tabs(["User Guide", "Individual Prediction", "Batch Analysis"])

# ==================== TAB 1: USER GUIDE ====================
with tab1:
    st.markdown("## Panduan Penggunaan Sistem")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### Apa itu Churn?
        
        Churn adalah ketika pelanggan berhenti menggunakan layanan perusahaan. 
        Dalam konteks telco, ini berarti pelanggan membatalkan langganan mereka.
        
        Mengapa Churn Penting?
        - Biaya akuisisi pelanggan baru 5-25x lebih mahal daripada mempertahankan pelanggan existing
        - Meningkatkan retensi 5% dapat meningkatkan profit 25-95%
        
        ---
        
        ### Tujuan Sistem Ini
        
        Sistem ini membantu perusahaan untuk:
        1. Identifikasi pelanggan berisiko tinggi churn
        2. Prediksi probabilitas churn dengan akurasi ~80%
        3. Analisis faktor-faktor penyebab churn
        4. Rekomendasi strategi retensi
        """)
    
    with col2:
        st.markdown("""
        ### Cara Menggunakan
        
        #### Individual Prediction
        Untuk analisis pelanggan per individu:
        
        Step 1: Input data pelanggan
        - Profile (gender, age, tenure, dll)
        - Contract & billing info
        - Services yang digunakan
        
        Step 2: Klik "Predict"
        - Sistem akan menampilkan risk level
        - Analisis faktor penyebab churn
        - Rekomendasi retensi spesifik
        
        Step 3: Test "What-If" scenarios
        - Coba ubah contract type
        - Test perubahan payment method
        - Lihat impact perubahan harga
        
        ---
        
        #### Batch Analysis
        Untuk analisis massal:
        
        Step 1: Upload CSV file
        - Format harus sama dengan training data
        - Dapat menggunakan ratusan/ribuan records
        
        Step 2: Klik "Analyze All"
        - Sistem prediksi semua pelanggan
        - Segment berdasarkan risk level
        - Identifikasi top high-risk customers
        
        Step 3: Review insights
        - Executive summary
        - Financial impact analysis
        - Strategic recommendations
        
        Step 4: Download results
        - CSV dengan semua predictions
        - Churn reasons per customer
        - Ready untuk CRM integration
        """)
    
    st.markdown("---")
    
    st.markdown("## Memahami Risk Levels")
    
    risk_col1, risk_col2, risk_col3 = st.columns(3)
    
    with risk_col1:
        st.error("""
        ### HIGH RISK (60%+)
        
        Karakteristik:
        - Month-to-month contract
        - Electronic check payment
        - Tenure < 6 bulan
        - High monthly charges
        - No value-added services
        
        Action Required:
        - Segera lakukan pendekatan
        - Berikan penawaran retensi yang agresif
        - Tugaskan account manager khusus
        - Lakukan panggilan personal dalam 24 jam
        
        Budget: $50-$150 per customer
        """)
    
    with risk_col2:
        st.warning("""
        ### MEDIUM RISK (35-60%)
        
        Karakteristik:
        - Mix of risky factors
        - Moderate tenure (6-18 months)
        - One-year contract
        - Some services missing
        
        Action Required:
        - Kampanye email proaktif
        - Hadiah/Reward loyalitas
        - Survei kepuasan
        - Tindak lanjut dalam 1 minggu
        
        Budget: $25-$50 per customer
        """)
    
    with risk_col3:
        st.success("""
        ### LOW RISK (<35%)
        
        Karakteristik:
        - Long-term contract (2-year)
        - Auto-pay enabled
        - High tenure (>24 months)
        - Multiple services
        - Partner & dependents
        
        Action Required:
        - Peluang upsell
        - Program loyalitas
        - Insentif referral
        - Inisiatif pertumbuhan
        
        Budget: Investasikan kembali untuk pertumbuhan
        """)
    
    st.markdown("---")
    
    st.markdown("## Key Insights dari Data")
    
    if original_data is not None:
        insight_col1, insight_col2 = st.columns(2)
        
        with insight_col1:
            # Churn by contract
            st.markdown("### Contract Type Impact")
            contract_churn = original_data.groupby('Contract')['Churn'].apply(
                lambda x: (x == 'Yes').sum() / len(x) * 100
            ).sort_values(ascending=False)
            
            fig = px.bar(
                x=contract_churn.values,
                y=contract_churn.index,
                orientation='h',
                title='Churn Rate by Contract Type',
                labels={'x': 'Churn Rate (%)', 'y': 'Contract'},
                color=contract_churn.values,
                color_continuous_scale='Reds'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.info(f"""
            Key Finding:
            - Month-to-month: {contract_churn['Month-to-month']:.1f}% churn
            - One year: {contract_churn.get('One year', 0):.1f}% churn
            - Two year: {contract_churn.get('Two year', 0):.1f}% churn
            
            Action: Prioritaskan kampanye upgrade kontrak!
            """)
        
        with insight_col2:
            # Churn by tenure
            st.markdown("### Tenure Impact")
            original_data['TenureGroup'] = pd.cut(
                original_data['tenure'],
                bins=[0, 12, 24, 48, 100],
                labels=['0-12m', '12-24m', '24-48m', '48m+']
            )
            tenure_churn = original_data.groupby('TenureGroup')['Churn'].apply(
                lambda x: (x == 'Yes').sum() / len(x) * 100
            )
            
            fig = px.bar(
                x=tenure_churn.index,
                y=tenure_churn.values,
                title='Churn Rate by Tenure',
                labels={'x': 'Tenure', 'y': 'Churn Rate (%)'},
                color=tenure_churn.values,
                color_continuous_scale='Reds'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.info(f"""
            Key Finding:
            - First year: {tenure_churn.iloc[0]:.1f}% churn (CRITICAL!)
            - After 4 years: {tenure_churn.iloc[-1]:.1f}% churn
            
            Action: Program perawatan intensif untuk pelanggan baru!
            """)

# ==================== TAB 2: INDIVIDUAL PREDICTION ====================
with tab2:
    st.markdown("## Individual Customer Churn Prediction")
    
    # Input form
    with st.form("prediction_form"):
        st.markdown("### Input Customer Data")
        
        # Demographics
        with st.expander("Demographics & Profile", expanded=True):
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                gender = st.selectbox("Gender", ["Female", "Male"])
            with col2:
                senior = st.selectbox("Senior Citizen", ["No", "Yes"])
            with col3:
                partner = st.selectbox("Has Partner", ["No", "Yes"])
            with col4:
                dependents = st.selectbox("Has Dependents", ["No", "Yes"])
        
        # Tenure & Billing
        with st.expander("Billing & Tenure", expanded=True):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                tenure = st.slider("Tenure (months)", 0, 72, 1, help="How long customer has been with us")
            with col2:
                monthly = st.number_input("Monthly Charges ($)", 18.0, 120.0, 70.0, 5.0)
            with col3:
                total = st.number_input("Total Charges ($)", 0.0, 10000.0, monthly * tenure, 50.0)
        
        # Contract & Payment
        with st.expander("Contract & Payment", expanded=True):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                contract = st.selectbox(
                    "Contract Type",
                    ["Month-to-month", "One year", "Two year"],
                    help="⚠ Month-to-month = HIGHEST RISK!"
                )
            with col2:
                payment = st.selectbox(
                    "Payment Method",
                    ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
                    help="⚠ Electronic check = HIGH RISK!"
                )
            with col3:
                paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
        
        # Services
        with st.expander("Services & Features", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("Phone & Internet")
                phone = st.selectbox("Phone Service", ["Yes", "No"])
                lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
                internet = st.selectbox(
                    "Internet Service",
                    ["DSL", "Fiber optic", "No"],
                    help="Fiber optic without services = HIGH RISK"
                )
            
            with col2:
                st.markdown("Security & Support")
                security = st.selectbox(
                    "Online Security",
                    ["No", "Yes", "No internet service"],
                    help="⚠ No = Risk factor"
                )
                backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
                protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
                support = st.selectbox(
                    "Tech Support",
                    ["No", "Yes", "No internet service"],
                    help="⚠ No = Risk factor"
                )
            
            col3, col4 = st.columns(2)
            with col3:
                st.markdown("Streaming")
                tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
            with col4:
                st.markdown("Movies")
                movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
        
        submit = st.form_submit_button("PREDICT CHURN RISK", type="primary", use_container_width=True)
    
    # Logika Prediksi
    if submit:
        # Data input
        input_dict = {
            "gender": gender,
            "SeniorCitizen": 1 if senior == "Yes" else 0,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone,
            "MultipleLines": lines,
            "InternetService": internet,
            "OnlineSecurity": security,
            "OnlineBackup": backup,
            "DeviceProtection": protection,
            "TechSupport": support,
            "StreamingTV": tv,
            "StreamingMovies": movies,
            "Contract": contract,
            "PaperlessBilling": paperless,
            "PaymentMethod": payment,
            "MonthlyCharges": monthly,
            "TotalCharges": total
        }
        
        # Preprocess
        processed = preprocess_input(input_dict, feature_names)
        
        # Prediksi
        prediction = model.predict(processed)[0]
        probability = model.predict_proba(processed)[0, 1]
        
        # Mendapatkan risk level
        risk_label, risk_color = get_risk_level(probability)
        
        # Display Hasil
        st.markdown("---")
        st.markdown("## Prediction Results")
        
        # Metrics
        met1, met2, met3 = st.columns(3)
        
        with met1:
            if "HIGH" in risk_label:
                st.error(f"### {risk_label}")
            elif "MEDIUM" in risk_label:
                st.warning(f"### {risk_label}")
            else:
                st.success(f"### {risk_label}")
        
        with met2:
            st.metric("Churn Probability", f"{probability:.1%}", 
                     help="Probability this customer will churn")
        
        with met3:
            confidence = max(probability, 1 - probability)
            st.metric("Model Confidence", f"{confidence:.1%}",
                     help="How confident the model is in this prediction")
        
        # Indikator Risiko
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=probability * 100,
            title={'text': "Churn Risk Score", 'font': {'size': 24}},
            delta={'reference': 50, 'increasing': {'color': "red"}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1},
                'bar': {'color': risk_color},
                'bgcolor': "white",
                'steps': [
                    {'range': [0, 35], 'color': 'lightgreen'},
                    {'range': [35, 60], 'color': 'lightyellow'},
                    {'range': [60, 100], 'color': 'lightcoral'}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 60
                }
            }
        ))
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
        
        # Analisis Faktor Churn
        st.markdown("---")
        st.markdown("## Analisis Akar Masalah")
        
        factors = analyze_churn_factors(input_dict, probability)
        
        if factors:
            for factor in factors:
                if factor['impact'] == 'CRITICAL':
                    st.error(f"{factor['factor']}\n\n{factor['detail']}")
                elif factor['impact'] == 'HIGH':
                    st.warning(f"{factor['factor']}\n\n{factor['detail']}")
                else:
                    st.info(f"{factor['factor']}\n\n{factor['detail']}")
        else:
            st.success("No major risk factors detected!")
        
        # Rekomendasi
        st.markdown("---")
        st.markdown("## Rekomendasi Tindak Lanjut")
        
        recommendations = generate_recommendations(input_dict, probability, factors)
        
        for i, rec in enumerate(recommendations):
            with st.expander(f"{rec['priority']}: {rec['action']}", expanded=(i < 2)):
                st.markdown(rec['detail'])
                
                if 'impact' in rec and 'cost' in rec and 'roi' in rec:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Impact", rec['impact'])
                    with col2:
                        st.metric("Investment", rec['cost'])
                    with col3:
                        st.metric("Potential ROI", rec['roi'])
        
        # What-if analysis
        st.markdown("---")
        st.markdown("##  What-If Analysis")
        st.info("Coba berbagai skenario untuk melihat bagaimana perubahan memengaruhi risiko churn!")
        
        whatif_col1, whatif_col2 = st.columns(2)
        
        with whatif_col1:
            st.markdown("### Scenario 1: Contract Upgrade")
            
            # Simulasi Perubahan Kontrak
            test_scenarios = []
            
            for new_contract in ["Month-to-month", "One year", "Two year"]:
                test_input = input_dict.copy()
                test_input['Contract'] = new_contract
                test_processed = preprocess_input(test_input, feature_names)
                test_prob = model.predict_proba(test_processed)[0, 1]
                
                test_scenarios.append({
                    'Contract': new_contract,
                    'Probability': test_prob,
                    'Change': test_prob - probability
                })
            
            scenario_df = pd.DataFrame(test_scenarios)
            
            fig = px.bar(
                scenario_df,
                x='Contract',
                y='Probability',
                title='Impact of Contract Type',
                labels={'Probability': 'Churn Probability'},
                color='Probability',
                color_continuous_scale='RdYlGn_r',
                text=scenario_df['Probability'].apply(lambda x: f'{x:.1%}')
            )
            fig.add_hline(y=probability, line_dash="dash", line_color="red",
                         annotation_text="Current")
            st.plotly_chart(fig, use_container_width=True)
            
            # Opsi terbaik
            best_contract = scenario_df.loc[scenario_df['Probability'].idxmin()]
            improvement = (probability - best_contract['Probability']) * 100
            
            if improvement > 10:
                st.success(f"""
                Recommendation: Switch to **{best_contract['Contract']}
                - Reduces risk by {improvement:.1f} percentage points
                - New probability: {best_contract['Probability']:.1%}
                """)
        
        with whatif_col2:
            st.markdown("### Scenario 2: mengubah Payment Method")
            
            # Simulasi peruabahan payment method
            payment_scenarios = []
            
            for new_payment in ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]:
                test_input = input_dict.copy()
                test_input['PaymentMethod'] = new_payment
                test_processed = preprocess_input(test_input, feature_names)
                test_prob = model.predict_proba(test_processed)[0, 1]
                
                payment_scenarios.append({
                    'Payment': new_payment.replace(' (automatic)', ''),
                    'Probability': test_prob
                })
            
            payment_df = pd.DataFrame(payment_scenarios)
            
            fig = px.bar(
                payment_df,
                x='Payment',
                y='Probability',
                title='Impact of Payment Method',
                labels={'Probability': 'Churn Probability'},
                color='Probability',
                color_continuous_scale='RdYlGn_r',
                text=payment_df['Probability'].apply(lambda x: f'{x:.1%}')
            )
            fig.add_hline(y=probability, line_dash="dash", line_color="red",
                         annotation_text="Current")
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
            
            # Opsi terbaik
            best_payment = payment_df.loc[payment_df['Probability'].idxmin()]
            improvement = (probability - best_payment['Probability']) * 100
            
            if improvement > 5:
                st.success(f"""
                Recommendation: Switch to **{best_payment['Payment']}
                - Reduces risk by {improvement:.1f} percentage points
                - New probability: {best_payment['Probability']:.1%}
                """)
        
        # Skenario Gabungan
        st.markdown("### Optimal Scenario")
        
        optimal_input = input_dict.copy()
        optimal_input['Contract'] = "Two year"
        optimal_input['PaymentMethod'] = "Credit card (automatic)"
        optimal_input['OnlineSecurity'] = "Yes"
        optimal_input['TechSupport'] = "Yes"
        
        optimal_processed = preprocess_input(optimal_input, feature_names)
        optimal_prob = model.predict_proba(optimal_processed)[0, 1]
        
        improvement = (probability - optimal_prob) * 100
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Current Risk", f"{probability:.1%}")
        with col2:
            st.metric("Optimal Risk", f"{optimal_prob:.1%}", 
                     delta=f"-{improvement:.1f}pp", delta_color="inverse")
        with col3:
            reduction = (1 - optimal_prob/probability) * 100
            st.metric("Risk Reduction", f"{reduction:.0f}%")
        
        st.info("""
        Optimal Package:
        - Kontrak 2 tahun dengan diskon 25%
        - Auto-pay (Kartu kredit)
        - Online Security + Tech Support GRATIS selama 3 bulan
        - Bonus loyalitas: Kredit $50
        
        Hasil yang Diharapkan: Risiko churn berkurang sebesar {:.0f}%
        """.format(reduction))

# ==================== TAB 3: BATCH ANALYSIS ====================
with tab3:
    st.markdown("## Batch Customer Analysis")
    
    st.info("""
    Upload CSV file dengan kolom yang sama dengan training data.
    Sistem akan memprediksi churn risk untuk semua pelanggan sekaligus dan memberikan insights.
    """)
    
    # Required columns information
    with st.expander("Format Dataset yang Diperlukan", expanded=False):
        st.markdown("""
        ### Kolom Wajib dalam CSV:
        
        **Demographic & Account Info:**
        - `gender` - Gender pelanggan (Female/Male)
        - `SeniorCitizen` - Apakah senior citizen (0/1 atau Yes/No)
        - `Partner` - Memiliki partner (Yes/No)
        - `Dependents` - Memiliki tanggungan (Yes/No)
        - `tenure` - Lama berlangganan dalam bulan (angka)
        
        **Billing Info:**
        - `Contract` - Jenis kontrak (Month-to-month/One year/Two year)
        - `PaperlessBilling` - Paperless billing (Yes/No)
        - `PaymentMethod` - Metode pembayaran (Electronic check/Mailed check/Bank transfer (automatic)/Credit card (automatic))
        - `MonthlyCharges` - Biaya bulanan (angka)
        - `TotalCharges` - Total biaya (angka)
        
        **Phone Services:**
        - `PhoneService` - Layanan telepon (Yes/No)
        - `MultipleLines` - Multiple lines (Yes/No/No phone service)
        
        **Internet Services:**
        - `InternetService` - Layanan internet (DSL/Fiber optic/No)
        - `OnlineSecurity` - Online security (Yes/No/No internet service)
        - `OnlineBackup` - Online backup (Yes/No/No internet service)
        - `DeviceProtection` - Device protection (Yes/No/No internet service)
        - `TechSupport` - Tech support (Yes/No/No internet service)
        - `StreamingTV` - Streaming TV (Yes/No/No internet service)
        - `StreamingMovies` - Streaming movies (Yes/No/No internet service)
        
        **Optional:**
        - `customerID` - ID pelanggan (opsional, akan diabaikan saat prediksi)
        - `Churn` - Status churn aktual (opsional, untuk perbandingan)
        
        ---
        
        ### Contoh Format:
        ```
        customerID,gender,SeniorCitizen,Partner,Dependents,tenure,PhoneService,...
        7590-VHVEG,Female,0,Yes,No,1,No,...
        5575-GNVDE,Male,0,No,No,34,Yes,...
        ```
        
        📥 **Download template:** Gunakan dataset WA_Fn-UseC_-Telco-Customer-Churn.csv sebagai referensi format.
        """)
    
    st.markdown("---")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload Customer Data (CSV)",
        type=['csv'],
        help="File harus memiliki kolom: gender, SeniorCitizen, Partner, tenure, MonthlyCharges, Contract, dll."
    )
    
    if uploaded_file is not None:
        # Load file
        try:
            batch_df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Gagal membaca file CSV: {str(e)}")
            st.info("Pastikan file adalah CSV yang valid dengan encoding UTF-8.")
            st.stop()
        
        # ==================== VALIDASI DATASET ====================
        required_columns = [
            'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
            'PhoneService', 'MultipleLines', 'InternetService', 
            'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
            'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
            'PaymentMethod', 'MonthlyCharges', 'TotalCharges'
        ]
        
        # Cek kolom yang hilang
        missing_columns = [col for col in required_columns if col not in batch_df.columns]
        
        if missing_columns:
            st.error("❌ **Dataset tidak sesuai format!**")
            st.markdown(f"### Kolom yang hilang ({len(missing_columns)}):")
            
            # Tampilkan dalam 3 kolom
            missing_col1, missing_col2, missing_col3 = st.columns(3)
            
            for idx, col in enumerate(missing_columns):
                if idx % 3 == 0:
                    missing_col1.markdown(f"- `{col}`")
                elif idx % 3 == 1:
                    missing_col2.markdown(f"- `{col}`")
                else:
                    missing_col3.markdown(f"- `{col}`")
            
            st.warning("""
            ### Cara Memperbaiki:
            
            1. **Pastikan semua kolom wajib ada** dalam CSV Anda
            2. **Periksa ejaan** - kolom harus persis sama (case-sensitive)
            3. **Download template** dari dokumentasi atau gunakan format dataset training
            4. **Periksa struktur** - pastikan tidak ada baris kosong di awal file
            
            Klik expander "Format Dataset yang Diperlukan" di atas untuk melihat daftar lengkap kolom yang diperlukan.
            """)
            st.stop()
        
        # Validasi tipe data dan nilai
        validation_errors = []
        
        # Cek kolom numerik
        numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
        for col in numeric_cols:
            if col in batch_df.columns:
                # Coba konversi ke numerik
                batch_df[col] = pd.to_numeric(batch_df[col], errors='coerce')
                
                # Hitung berapa banyak nilai yang tidak valid
                invalid_count = batch_df[col].isna().sum()
                if invalid_count > 0:
                    validation_errors.append(f"⚠ `{col}`: {invalid_count} nilai tidak valid (harus numerik)")
        
        # Cek nilai SeniorCitizen
        if 'SeniorCitizen' in batch_df.columns:
            unique_values = batch_df['SeniorCitizen'].unique()
            valid_values = [0, 1, '0', '1', 'Yes', 'No', 'yes', 'no']
            invalid_seniors = [v for v in unique_values if v not in valid_values and pd.notna(v)]
            if invalid_seniors:
                validation_errors.append(f"⚠ `SeniorCitizen`: Nilai tidak valid ditemukan {invalid_seniors}. Harus: 0/1 atau Yes/No")
        
        # Cek nilai Contract
        if 'Contract' in batch_df.columns:
            valid_contracts = ['Month-to-month', 'One year', 'Two year']
            invalid_contracts = batch_df[~batch_df['Contract'].isin(valid_contracts + [np.nan])]['Contract'].unique()
            if len(invalid_contracts) > 0:
                validation_errors.append(f"⚠ `Contract`: Nilai tidak valid {list(invalid_contracts)}. Harus: {valid_contracts}")
        
        # Tampilkan peringatan validasi jika ada
        if validation_errors:
            st.warning("⚠ **Peringatan Validasi Data:**")
            for error in validation_errors:
                st.markdown(f"- {error}")
            st.info("""
            Dataset akan tetap diproses, tetapi hasil prediksi mungkin kurang akurat untuk baris dengan data tidak valid.
            Sebaiknya perbaiki data terlebih dahulu untuk hasil optimal.
            """)

        st.success(f"✅ File berhasil dimuat: **{len(batch_df):,}** pelanggan")
        
        # Tampilkan ringkasan dataset
        summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)
        
        with summary_col1:
            st.metric("Total Baris", f"{len(batch_df):,}")
        with summary_col2:
            st.metric("Total Kolom", f"{len(batch_df.columns)}")
        with summary_col3:
            has_customer_id = "customerID" in batch_df.columns
            st.metric("Customer ID", "Ada" if has_customer_id else "Tidak ada")
        with summary_col4:
            has_churn = "Churn" in batch_df.columns
            st.metric("Label Churn", "Ada" if has_churn else "Tidak ada")
        
        with st.expander("Preview Data", expanded=False):
            st.dataframe(batch_df.head(20), use_container_width=True)

        # Button Analisis
        if st.button("Run Batch Analysis", use_container_width=True, type="primary"):
            try:
                progress = st.progress(0)
                status = st.empty()
                
                # ---------- PREPROCESSING ----------
                status.text("⚙ Preprocessing data...")
                progress.progress(25)

                processed = preprocess_batch(batch_df, feature_names)

                # ---------- PREDICTION ----------
                status.text("Predicting churn...")
                progress.progress(55)

                preds = model.predict(processed)

                # ---------- PROBABILITIES ----------
                status.text("Calculating probabilities...")
                progress.progress(80)

                probs = model.predict_proba(processed)[:, 1]

                # ------------- FINAL ---------------
                status.text("Finalizing results...")
                progress.progress(100)

                # Gabungkan ke dataframe asli
                batch_df['ChurnPrediction'] = ['CHURN' if p == 1 else 'RETAINED' for p in preds]
                batch_df['ChurnProbability'] = probs
                batch_df['RiskLevel'] = pd.cut(
                    probs,
                    bins=[0, 0.35, 0.60, 1],
                    labels=['LOW', 'MEDIUM', 'HIGH']
                )
                original_batch = batch_df.copy()

                status.empty()
                progress.empty()

                st.success("Batch analysis completed!")

            except Exception as e:
                st.error(f"Error during batch processing: {str(e)}")
        
            # ==================== EXECUTIVE SUMMARY ====================
            st.markdown("---")
            st.markdown("## Executive Summary")
            
            total_customers = len(original_batch)
            predicted_churners = (original_batch['ChurnPrediction'] == 'CHURN').sum()
            churn_rate = (predicted_churners / total_customers) * 100
            
            high_risk = (original_batch['RiskLevel'] == 'HIGH').sum()
            medium_risk = (original_batch['RiskLevel'] == 'MEDIUM').sum()
            low_risk = (original_batch['RiskLevel'] == 'LOW').sum()
            
            avg_prob = original_batch['ChurnProbability'].mean()
            
            # Key metrics
            metric1, metric2, metric3, metric4 = st.columns(4)
            
            with metric1:
                st.metric("Total Customers", f"{total_customers:,}")
            with metric2:
                st.metric("⚠ Predicted Churners", f"{predicted_churners:,}", 
                         f"{churn_rate:.1f}%")
            with metric3:
                st.metric("High Risk", f"{high_risk:,}",
                         f"{high_risk/total_customers*100:.1f}%")
            with metric4:
                st.metric("Avg Churn Probability", f"{avg_prob:.1%}")
            
            # Risk distribution
            st.markdown("### Risk Distribution")
            
            dist_col1, dist_col2 = st.columns([2, 1])
            
            with dist_col1:
                # Histogram
                fig = px.histogram(
                    original_batch,
                    x='ChurnProbability',
                    color='RiskLevel',
                    nbins=40,
                    title='Churn Probability Distribution',
                    labels={'ChurnProbability': 'Churn Probability'},
                    color_discrete_map={'LOW': 'green', 'MEDIUM': 'orange', 'HIGH': 'red'}
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            with dist_col2:
                # Pie chart
                risk_counts = original_batch['RiskLevel'].value_counts()
                fig = px.pie(
                    values=risk_counts.values,
                    names=risk_counts.index,
                    title='Risk Segmentation',
                    color=risk_counts.index,
                    color_discrete_map={'LOW': 'green', 'MEDIUM': 'orange', 'HIGH': 'red'}
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            # ==================== FINANCIAL IMPACT ====================
            st.markdown("---")
            st.markdown("## Financial Impact Analysis")
            
            if 'MonthlyCharges' in original_batch.columns:
                # Revenue at risk
                high_risk_customers = original_batch[original_batch['RiskLevel'] == 'HIGH']
                medium_risk_customers = original_batch[original_batch['RiskLevel'] == 'MEDIUM']
                
                revenue_at_risk_high = high_risk_customers['MonthlyCharges'].sum()
                revenue_at_risk_medium = medium_risk_customers['MonthlyCharges'].sum()
                total_revenue_at_risk = revenue_at_risk_high + revenue_at_risk_medium
                
                revenue_at_risk_annual = total_revenue_at_risk * 12
                
                fin1, fin2, fin3 = st.columns(3)
                
                with fin1:
                    st.metric("Monthly Revenue at Risk", f"${total_revenue_at_risk:,.0f}")
                with fin2:
                    st.metric("Annual Revenue at Risk", f"${revenue_at_risk_annual:,.0f}")
                with fin3:
                    retention_budget = total_revenue_at_risk * 1.5
                    st.metric("Suggested Retention Budget", f"${retention_budget:,.0f}")
                
                # ROI calculation
                st.markdown("### Retention Investment ROI")
                
                # Asumsikan tingkat keberhasilan retensi 70% dengan rekomendasi kami
                retained_revenue = revenue_at_risk_annual * 0.70
                roi_pct = ((retained_revenue - retention_budget) / retention_budget * 100)
                
                roi1, roi2, roi3, roi4 = st.columns(4)
                
                with roi1:
                    st.metric("Investment", f"${retention_budget:,.0f}")
                with roi2:
                    st.metric("Potential Savings", f"${retained_revenue:,.0f}",
                             help="70% retention success rate")
                with roi3:
                    st.metric("Net Benefit", f"${retained_revenue - retention_budget:,.0f}")
                with roi4:
                    st.metric("ROI", f"{roi_pct:.0f}%")
                
                st.success(f"""
                ### Investment Recommendation
                
                Investasikan ${retention_budget:,.0f} dalam program retensi yang terarah:
                - Fokus pada {high_risk} pelanggan berisiko tinggi (tindakan segera)
                - Pendekatan proaktif untuk {medium_risk} pelanggan berisiko menengah
                - Perkiraan tingkat retensi: 70%
                - Potensi penghematan tahunan: ${retained_revenue:,.0f}
                - Return on investment: {roi_pct:.0f}%

                Ini adalah investasi yang jelas sangat menguntungkan!

                """)
            
            # ==================== TOP HIGH-RISK CUSTOMERS ====================
            st.markdown("---")
            st.markdown("## Top 20 High-Risk Customers - Immediate Action Required")
            
            top_risk = original_batch[original_batch['RiskLevel'] == 'HIGH'].sort_values(
                'ChurnProbability', ascending=False
            ).head(20)
            
            # Select display columns
            display_cols = []
            if 'customerID' in top_risk.columns:
                display_cols.append('customerID')
            
            important_cols = ['tenure', 'Contract', 'MonthlyCharges', 'PaymentMethod', 
                            'ChurnProbability', 'RiskLevel']
            
            for col in important_cols:
                if col in top_risk.columns:
                    display_cols.append(col)
            
            # Format display
            display_df = top_risk[display_cols].copy()
            if 'ChurnProbability' in display_df.columns:
                display_df['ChurnProbability'] = display_df['ChurnProbability'].apply(lambda x: f"{x:.1%}")
            if 'MonthlyCharges' in display_df.columns:
                display_df['MonthlyCharges'] = display_df['MonthlyCharges'].apply(lambda x: f"${x:.2f}")
            
            st.dataframe(
                display_df,
                use_container_width=True,
                height=400
            )
            
            st.warning("""
            URGENT: Pelanggan ini membutuhkan penanganan segera dalam 24–48 jam.
            - Tugaskan account manager khusus
            - Siapkan penawaran retensi yang dipersonalisasi
            - Lakukan panggilan prioritas dari tim customer
            """)
            
            # ==================== STRATEGIC INSIGHTS ====================
            st.markdown("---")
            st.markdown("## Strategic Insights & Recommendations")
            
            insight1, insight2 = st.columns(2)
            
            with insight1:
                st.markdown("### Contract Analysis")
                
                if 'Contract' in original_batch.columns:
                    # Churn berdasarkan contract
                    contract_risk = original_batch.groupby('Contract').agg({
                        'ChurnProbability': 'mean',
                        'customerID' if 'customerID' in original_batch.columns else 'tenure': 'count'
                    }).round(3)
                    contract_risk.columns = ['Avg Churn Prob', 'Count']
                    contract_risk = contract_risk.sort_values('Avg Churn Prob', ascending=False)
                    
                    fig = px.bar(
                        contract_risk.reset_index(),
                        x='Contract',
                        y='Avg Churn Prob',
                        title='Average Churn Risk by Contract Type',
                        color='Avg Churn Prob',
                        color_continuous_scale='Reds',
                        text=contract_risk['Avg Churn Prob'].apply(lambda x: f'{x:.1%}')
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Jumlah pelanggan berisiko churn tinggi berdasarkan jenis kontrak
                    mtm_high_risk = original_batch[
                        (original_batch['Contract'] == 'Month-to-month') &
                        (original_batch['RiskLevel'] == 'HIGH')
                    ].shape[0]
                    
                    if mtm_high_risk > 0:
                        st.error(f"""
                        Critical Finding:
                        - {mtm_high_risk} pelanggan berisiko tinggi dengan kontrak month-to-month
                        - Tindakan: Lakukan kampanye upgrade kontrak besar 
                        - Penawaran: Diskon 25% untuk yang kontrak 2 tahun
                        - Target: Konversi 60% = {int(mtm_high_risk * 0.6)} pelanggan

                        """)
            
            with insight2:
                st.markdown("### Payment Method Analysis")
                
                if 'PaymentMethod' in original_batch.columns:
                    # Churn berdasarkan payment
                    payment_risk = original_batch.groupby('PaymentMethod').agg({
                        'ChurnProbability': 'mean',
                        'customerID' if 'customerID' in original_batch.columns else 'tenure': 'count'
                    }).round(3)
                    payment_risk.columns = ['Avg Churn Prob', 'Count']
                    payment_risk = payment_risk.sort_values('Avg Churn Prob', ascending=False)
                    
                    fig = px.bar(
                        payment_risk.reset_index(),
                        x='PaymentMethod',
                        y='Avg Churn Prob',
                        title='Average Churn Risk by Payment Method',
                        color='Avg Churn Prob',
                        color_continuous_scale='Reds',
                        text=payment_risk['Avg Churn Prob'].apply(lambda x: f'{x:.1%}')
                    )
                    fig.update_layout(xaxis_tickangle=-30)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Count of high-risk by payment
                    echeck_high_risk = original_batch[
                        (original_batch['PaymentMethod'] == 'Electronic check') &
                        (original_batch['RiskLevel'] == 'HIGH')
                    ].shape[0]
                    
                    if echeck_high_risk > 0:
                        st.warning(f"""
                        ⚠ Key Finding:
                        - {echeck_high_risk} pelanggan berisiko tinggi yang menggunakan electronic check
                        - Tindakan: Program migrasi ke auto-pay
                        - Insentif: Kredit $50 + diskon $5/bulan
                        - Perkiraan konversi: 75% = {int(echeck_high_risk * 0.75)} pelanggan
                        """)
            
            # Tenure analysis
            st.markdown("### Customer Tenure Impact")
            
            if 'tenure' in original_batch.columns:
                # Create tenure group
                original_batch['TenureGroup'] = pd.cut(
                    original_batch['tenure'],
                    bins=[0, 6, 12, 24, 48, 100],
                    labels=['0-6m', '6-12m', '12-24m', '24-48m', '48m+']
                )
                
                tenure_risk = original_batch.groupby('TenureGroup').agg({
                    'ChurnProbability': 'mean',
                    'customerID' if 'customerID' in original_batch.columns else 'tenure': 'count'
                }).round(3)
                tenure_risk.columns = ['Avg Churn Prob', 'Count']
                
                fig = px.bar(
                    tenure_risk.reset_index(),
                    x='TenureGroup',
                    y='Avg Churn Prob',
                    title='Churn Risk by Customer Tenure',
                    color='Avg Churn Prob',
                    color_continuous_scale='Reds',
                    text=tenure_risk['Avg Churn Prob'].apply(lambda x: f'{x:.1%}')
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # New customer count
                new_customer_high_risk = original_batch[
                    (original_batch['tenure'] < 12) &
                    (original_batch['RiskLevel'].isin(['HIGH', 'MEDIUM']))
                ].shape[0]
                
                if new_customer_high_risk > 0:
                    st.error(f"""
                    Critical Priority:
                    - {new_customer_high_risk} pelanggan berisiko dengan tenure < 12 bulan
                    - Tindakan: Program perawatan pelanggan baru secara langsung
                    - Strategi:
                    - Panggilan check-in mingguan (3 bulan pertama)
                    - Account manager khusus
                    - Kredit loyalitas: $10/bulan selama 12 bulan pertama
                    - Bonus upgrade kontrak lebih awal
                    - Investasi: ${new_customer_high_risk * 150:,}
                    - Perkiraan penghematan: ${new_customer_high_risk * 800:,} (mencegah churn)
                    """)
            
            # ==================== ACTION PLAN ====================
            st.markdown("---")
            st.markdown("## 90-Day Action Plan")
            
            plan_col1, plan_col2, plan_col3 = st.columns(3)
            
            with plan_col1:
                st.markdown("### Week 1-2: Immediate")
                st.error(f"""
                High-Risk Customers ({high_risk})
                
                - [ ] Ambil 20 pelanggan teratas untuk panggilan segera
                - [ ] Tetapkan account manager
                - [ ] Siapkan penawaran retensi
                - [ ] Lakukan kampanye outreach
                - [ ] Pantau konversi harian

                Budget: ${high_risk * 100:,}  
                Goal: Retensi 70% = {int(high_risk * 0.7)}
                """)
            
            with plan_col2:
                st.markdown("### Week 3-6: Proactive")
                st.warning(f"""
                Medium-Risk Customers ({medium_risk})
                
                - [ ] Kampanye survei kepuasan
                - [ ] Pendaftaran program loyalitas
                - [ ] Penawaran peningkatan layanan
                - [ ] Insentif upgrade kontrak
                - [ ] Pantau metrik keterlibatan pelanggan

                Busget: ${medium_risk * 40:,}  
                Goal: Mencegah peningkatan ke kategori berisiko tinggi
                """)
            
            with plan_col3:
                st.markdown("### Week 7-12: Growth")
                st.success(f"""
                Low-Risk Customers ({low_risk})
                
                - [ ] Identifikasi peluang upsell
                - [ ] Lakukan program referral
                - [ ] Manfaat tingkat VIP
                - [ ] Membangun komunitas
                - [ ] Mengumpulkan testimoni

                Budget: Investasikan kembali penghematan  
                Goal: Meningkatkan CLTV sebesar 30%

                """)
            
            # Download results
            st.markdown("---")
            st.markdown("##Download Hasil")
            
            # Prepare CSV
            download_df = original_batch.copy()
            if 'ChurnProbability' in download_df.columns:
                download_df['ChurnProbability'] = download_df['ChurnProbability'].round(4)
            
            csv = download_df.to_csv(index=False)
            
            st.download_button(
                label="⬇ Download Complete Analysis (CSV)",
                data=csv,
                file_name=f"churn_analysis_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True,
                type="primary"
            )
            
            st.success("""
            Analysis complete! 
            
            CSV ini berisi:
            - Seluruh data pelanggan asli
            - Prediksi churn (CHURN/RETAINED)
            - Skor probabilitas churn
            - Segmentasi tingkat risiko (HIGH/MEDIUM/LOW)

            Siap untuk integrasi dengan CRM atau analisis lanjutan di Excel/Tableau!
            """)
