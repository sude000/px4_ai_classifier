import streamlit as st
import pandas as pd
import numpy as np
import torch
import joblib
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from datetime import datetime

# app.py kök dizinde olduğu için src modüllerini doğrudan import edebiliriz!
from src import config
from src.models.architectures import get_model

# =========================================================================
# 1. MERKEZİ YAPILANDIRMA
# =========================================================================
ACTIVE_MODEL_TYPE = config.ACTIVE_MODEL_TYPE
ACTIVE_MODEL_WEIGHTS = config.ACTIVE_MODEL_WEIGHTS
WINDOW_SIZE = config.WINDOW_SIZE
STEP_SIZE = config.STEP_SIZE
CLASS_NAMES = config.CLASS_NAMES
COLUMNS_TO_DROP = config.COLUMNS_TO_DROP
SCALER_PATH = config.SCALER_PATH

# Arıza Renk Haritası (Temporal Highlighting için)
FAULT_COLORS = {
    0: {"name": "Normal", "color": "rgba(34, 197, 94, 0.2)", "solid": "#22C55E", "emoji": "✅"},
    1: {"name": "Engine Failure", "color": "rgba(239, 68, 68, 0.4)", "solid": "#EF4444", "emoji": "🔴"},      
    2: {"name": "Elevator Failure", "color": "rgba(249, 115, 22, 0.4)", "solid": "#F97316", "emoji": "🟠"},   
    3: {"name": "Rudder Failure", "color": "rgba(234, 179, 8, 0.4)", "solid": "#EAB308", "emoji": "🟡"},     
    4: {"name": "Aileron Failure", "color": "rgba(22, 163, 74, 0.4)", "solid": "#16A34A", "emoji": "🟢"},      
    5: {"name": "Multi-Fault", "color": "rgba(147, 51, 234, 0.4)", "solid": "#9333EA", "emoji": "🟣"}         
}

# =========================================================================
# 2. MODEL VE SCALER YÜKLEME
# =========================================================================
@st.cache_resource
def load_model_and_scaler():
    """Dinamik Model ve Scaler Yükleme"""
    if not os.path.exists(SCALER_PATH):
        st.error(f"❌ Scaler dosyası bulunamadı: {SCALER_PATH}")
        st.stop()
    
    scaler = joblib.load(SCALER_PATH)
    
    if not os.path.exists(ACTIVE_MODEL_WEIGHTS):
        st.error(f"❌ Model dosyası bulunamadı: {ACTIVE_MODEL_WEIGHTS}")
        st.stop()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    num_features = scaler.n_features_in_
    num_classes = config.NUM_CLASSES
    
    model = get_model(ACTIVE_MODEL_TYPE, num_features, num_classes, device)
    model.load_state_dict(torch.load(ACTIVE_MODEL_WEIGHTS, map_location=device))
    model.eval()
    
    return scaler, model, device

# =========================================================================
# 3. VERİ İŞLEME VE TAHMİN
# =========================================================================
def preprocess_and_predict(uploaded_file, scaler, model, device):
    """
    Canlı Özellik Ölçekleme ve Kayan Pencere Tahmini
    Journal Part 1 & preprocess_and_split.py mantığıyla birebir uyumlu.
    """
    df = pd.read_csv(uploaded_file)
    
    # 1. Zaman sütununu bul
    time_cols = [col for col in df.columns if 'time' in col.lower()]
    if not time_cols:
        st.error("❌ Veri setinde 'time' veya '%time' içeren bir sütun bulunamadı.")
        st.stop()
    time_col = time_cols[0]
    timestamps = df[time_col].values
    
    # 2. preprocess_and_split.py'deki COLS_TO_DROP listesini birebir uygula
    RAW_COLS_TO_DROP = [
        "mavros-imu-data_raw.field.angular_velocity.x",
        "mavros-imu-data_raw.field.angular_velocity.y",
        "mavros-imu-data_raw.field.angular_velocity.z",
        "mavros-imu-data_raw.field.linear_acceleration.x",
        "mavros-imu-data_raw.field.linear_acceleration.y",
        "mavros-imu-data_raw.field.linear_acceleration.z"
    ]
    
    all_cols_to_drop = [col for col in COLUMNS_TO_DROP if col in df.columns]
    all_cols_to_drop.extend([col for col in RAW_COLS_TO_DROP if col in df.columns])
    
    # 3. Feature sütunlarını belirle
    feature_cols = [col for col in df.columns if col not in all_cols_to_drop]
    feature_cols = [col for col in feature_cols if np.issubdtype(df[col].dtype, np.number)]
    
    X = df[feature_cols].values
    
    # 4. Güvenlik kontrolü
    if X.shape[1] != scaler.n_features_in_:
        st.error(f"❌ Özellik sayısı uyuşmazlığı!")
        st.error(f"Beklenen (Scaler): {scaler.n_features_in_}")
        st.error(f"Gelen (CSV): {X.shape[1]}")
        st.warning(f"Düşürülen sütunlar: {all_cols_to_drop}")
        st.warning(f"Kalan feature sütunları: {feature_cols}")
        st.stop()
    
    # 5. Canlı Ölçekleme
    X_scaled = scaler.transform(X)
    
    predictions = []
    pred_timestamps = []
    confidences = []
    
    # 6. Kayan Pencere Tahmini
    for i in range(0, len(X_scaled) - WINDOW_SIZE + 1, STEP_SIZE):
        window_data = X_scaled[i : i + WINDOW_SIZE]
        window_time = timestamps[i + WINDOW_SIZE - 1]
        
        tensor_data = torch.FloatTensor(window_data).unsqueeze(0).to(device)
        
        with torch.no_grad():
            logits = model(tensor_data)
            probs = torch.softmax(logits, dim=1)
            confidence, pred_class = torch.max(probs, dim=1)
            
        predictions.append(pred_class.item())
        confidences.append(confidence.item())
        pred_timestamps.append(window_time)
    
    return df, timestamps, pred_timestamps, predictions, confidences, time_col, feature_cols

# =========================================================================
# 4. STREAMLIT ARAYÜZÜ
# =========================================================================
st.set_page_config(page_title="PX4 AI Fault Classifier", page_icon="🚁", layout="wide")

st.title("🚁 PX4 AI Arıza Sınıflandırıcı")
st.markdown("Ham drone telemetri verisini yükleyin. Yapay zeka modeli, uçuş boyunca oluşabilecek arızaları **zamansal olarak** tespit edip görselleştirecektir.")

# Sidebar
st.sidebar.header("⚙️ Model Yapılandırması")
st.sidebar.info(f"""
**Aktif Model:** {ACTIVE_MODEL_TYPE.upper()}  
**Ağırlıklar:** `{ACTIVE_MODEL_WEIGHTS}`  
**Pencere Boyutu:** {WINDOW_SIZE} timestep  
**Adım Boyutu:** {STEP_SIZE} timestep  
""")

st.sidebar.markdown("---")
st.sidebar.subheader("📂 Veri Yükleme")
uploaded_file = st.sidebar.file_uploader("Bir ROS Bag CSV dosyası yükleyin", type=["csv"])

# =========================================================================
# 5. ANA İŞLEM AKIŞI
# =========================================================================
if uploaded_file is not None:
    st.success("✅ Dosya başarıyla yüklendi! İşlem başlatılıyor...")
    
    with st.spinner("Model ve scaler belleğe yükleniyor, veriler işleniyor..."):
        scaler, model, device = load_model_and_scaler()
        original_df, timestamps, pred_timestamps, predictions, confidences, time_col, feature_cols = preprocess_and_predict(
            uploaded_file, scaler, model, device
        )
    
    pred_df = pd.DataFrame({
    'timestamp': pred_timestamps,
    'prediction': predictions,
    'confidence': confidences
})

# 🛡️ NaN Koruması: Bazı pencereler için confidence NaN olabilir
    pred_df['confidence'] = pred_df['confidence'].fillna(0.5)  # NaN ise 0.5 (ortalama güven) varsay
    pred_df['prediction'] = pred_df['prediction'].fillna(0).astype(int)  # NaN ise 0 (Normal) varsay
    
    fault_periods = pred_df[pred_df['prediction'] > 0]
    
    # =========================================================================
    # 6. İNTERAKTİF SENSÖR SEÇİMİ (A Seçeneği)
    # =========================================================================
    st.subheader("🔍 Detaylı Zaman Serisi Analizi")
    
    # Timestamp'i saniyeye çevir (nanosecond -> second)
    TIME_SCALE = 1e9
    timestamps_sec = timestamps / TIME_SCALE
    pred_timestamps_sec = np.array(pred_timestamps) / TIME_SCALE
    
    # Uçuş süresini hesapla
    flight_duration = timestamps_sec[-1] - timestamps_sec[0]
    
    # Kullanıcıdan sensör seçimi al
    sensor_cols = [col for col in feature_cols if col in original_df.columns]
    
    # IMU sensörlerini önceliklendir
    imu_cols = [col for col in sensor_cols if 'angular_velocity' in col or 'acceleration' in col]
    default_col = imu_cols[0] if imu_cols else sensor_cols[0]
    
    selected_sensor = st.selectbox(
        "📡 Gösterilecek Sensör",
        options=sensor_cols,
        format_func=lambda x: x.split('.')[-1] if '.' in x else x,
        index=sensor_cols.index(default_col) if default_col in sensor_cols else 0
    )
    
    # Window görselleştirme ayarları
    col1, col2 = st.columns(2)
    with col1:
        show_windows = st.checkbox("🪟 Kayan Pencereleri Renkli Şeritlerle Göster", value=True)
    with col2:
        show_confidence = st.checkbox("📊 Güven Skorunu Alt Grafikte Göster", value=True)
    
    # =========================================================================
    # 7. ANA GRAFİK: Sensör Sinyali + Window Şeritleri
    # =========================================================================
    num_rows = 3 if show_confidence else 2
    row_heights = [0.5, 0.3, 0.2] if show_confidence else [0.6, 0.4]
    
    fig = make_subplots(
        rows=num_rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        row_heights=row_heights,
        subplot_titles=(
            f"📈 {selected_sensor.split('.')[-1]} - Sinyal",
            "🤖 AI Tahminleri (Window Bazlı)",
            "📊 Güven Skoru"
        ) if show_confidence else (
            f"📈 {selected_sensor.split('.')[-1]} - Sinyal",
            "🤖 AI Tahminleri (Window Bazlı)"
        )
    )
    
    # A) Ana Sinyal Grafiği
    sensor_values = original_df[selected_sensor].values
    fig.add_trace(
        go.Scatter(
            x=timestamps_sec,
            y=sensor_values,
            mode='lines',
            name='Sinyal',
            line=dict(color='#1e40af', width=1.5),
            hovertemplate='Zaman: %{x:.2f}s<br>Değer: %{y:.4f}<extra></extra>'
        ),
        row=1, col=1
    )
    
    # B) Window'ları renkli şeritler olarak göster (Yöntem A)
    if show_windows:
        # Her prediction için bir dikdörtgen çiz
        for idx, row in pred_df.iterrows():
            pred_class = row['prediction']
            conf = row['confidence']
            time_sec = row['timestamp'] / TIME_SCALE
            
            # Dikdörtgenin başlangıç ve bitiş zamanı
            # Her pencere STEP_SIZE timestep kapsar (yaklaşık 0.1 saniye)
            start_time = time_sec - (STEP_SIZE / 100.0)  # 100Hz varsayımı
            end_time = time_sec
            
            # Renk ve opaklık
            color_info = FAULT_COLORS[pred_class]
            base_color = color_info["color"]
            
            # 🛡️ Güvenlik: NaN veya geçersiz değerleri temizle
            if pd.isna(conf) or conf < 0 or conf > 1:
                conf = 0.5  # Varsayılan

            # Opaklığı confidence'a göre ayarla (0.2 - 0.6 arası)
            opacity = 0.2 + (conf * 0.4)
            
            # RGBA formatında renk oluştur
            if pred_class == 0:
                fill_color = f"rgba(34, 197, 94, {opacity:.2f})"
            elif pred_class == 1:
                fill_color = f"rgba(239, 68, 68, {opacity:.2f})"
            elif pred_class == 2:
                fill_color = f"rgba(249, 115, 22, {opacity:.2f})"
            elif pred_class == 3:
                fill_color = f"rgba(234, 179, 8, {opacity:.2f})"
            elif pred_class == 4:
                fill_color = f"rgba(22, 163, 74, {opacity:.2f})"
            else:
                fill_color = f"rgba(147, 51, 234, {opacity:.2f})"
            
            # Üst grafik (sensör) için şerit
            fig.add_shape(
                type="rect",
                x0=start_time, x1=end_time,
                y0=0, y1=1,
                fillcolor=fill_color,
                line=dict(width=0),
                layer="below",
                row=1, col=1
            )
            
            # Alt grafik (tahmin) için şerit
            fig.add_shape(
                type="rect",
                x0=start_time, x1=end_time,
                y0=0, y1=1,
                fillcolor=fill_color,
                line=dict(width=0),
                layer="below",
                row=2, col=1
            )
            
            # Confidence grafiği için de şerit
            if show_confidence:
                fig.add_shape(
                    type="rect",
                    x0=start_time, x1=end_time,
                    y0=0, y1=100,
                    fillcolor=fill_color,
                    line=dict(width=0),
                    layer="below",
                    row=3, col=1
                )
    
    # C) Alt grafik: Window tahminleri (scatter)
    fig.add_trace(
        go.Scatter(
            x=pred_timestamps_sec,
            y=pred_df['prediction'],
            mode='markers',
            name='Window Tahminleri',
            marker=dict(
                size=8,
                color=pred_df['prediction'],
                colorscale=[
                    [0, '#22C55E'],
                    [0.2, '#EF4444'],
                    [0.4, '#F97316'],
                    [0.6, '#EAB308'],
                    [0.8, '#16A34A'],
                    [1.0, '#9333EA']
                ],
                line=dict(width=1, color='white'),
                showscale=True,
                colorbar=dict(
                    title="Sınıf",
                    tickvals=[0, 1, 2, 3, 4, 5],
                    ticktext=["Normal", "Engine", "Elevator", "Rudder", "Aileron", "Multi"]
                )
            ),
            hovertemplate='Zaman: %{x:.2f}s<br>Sınıf: %{y}<extra></extra>'
        ),
        row=2, col=1
    )
    
    # D) Güven skoru grafiği
    if show_confidence:
        fig.add_trace(
            go.Scatter(
                x=pred_timestamps_sec,
                y=pred_df['confidence'] * 100,
                mode='lines+markers',
                name='Güven',
                line=dict(color='#10b981', width=2),
                marker=dict(size=4, color='#10b981'),
                fill='tozeroy',
                fillcolor='rgba(16, 185, 129, 0.2)',
                hovertemplate='Zaman: %{x:.2f}s<br>Güven: %{y:.1f}%<extra></extra>'
            ),
            row=3, col=1
        )
    
    # Layout ayarları
    fig.update_layout(
        height=800 if show_confidence else 600,
        title_text=f"🔍 Detaylı Analiz: {selected_sensor.split('.')[-1]}",
        hovermode="x unified",
        plot_bgcolor='white',
        paper_bgcolor='white',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    fig.update_xaxes(title_text="⏱️ Zaman (Saniye)", row=num_rows, col=1, gridcolor='lightgray')
    fig.update_yaxes(title_text="Sensör Değeri", row=1, col=1, gridcolor='lightgray')
    fig.update_yaxes(title_text="Tahmin Sınıfı", row=2, col=1, gridcolor='lightgray')
    if show_confidence:
        fig.update_yaxes(title_text="Güven (%)", row=3, col=1, range=[0, 100], gridcolor='lightgray')
    
    st.plotly_chart(fig, use_container_width=True)
    
    # =========================================================================
    # 8. KAYAN PENCERE GÖRSELLEŞTİRMESİ (Journal Concept)
    # =========================================================================
    with st.expander("📐 Kayan Pencere (Sliding Window) Konsepti", expanded=False):
        st.markdown(f"""
        **Konsept:** {WINDOW_SIZE} timestep'lik pencereler {STEP_SIZE} adım ile kaydırılıyor.
        
        **Gerçek Zaman Karşılığı:**
        - Her pencere ≈ **{WINDOW_SIZE/100:.2f} saniye** (100Hz'de)
        - Her adım ≈ **{STEP_SIZE/100:.2f} saniye**
        
        > *"120 saniyelik tüm uçuşu modele beslemek yerine, üst üste binen 50 zaman adımlık pencerelere bölüyoruz. 
        > Her pencere bir sınıflandırma alır. Bu bize ince taneli zamansal çözünürlük sağlar."*
        
        **Görselleştirme:**
        - Üst grafikteki renkli şeritler, her pencerenin tahmin ettiği zaman aralığını gösterir
        - Renk = Tahmin edilen arıza sınıfı
        - Opaklık = Modelin güven skoru (daha yüksek güven = daha opak)
        """)
        
        # İlk 5 pencereyi detaylı göster
        sample_windows = min(5, len(pred_df))
        
        if sample_windows > 0:
            st.markdown(f"**İlk {sample_windows} Pencerenin Detayları:**")
            
            window_data = []
            for i in range(sample_windows):
                start_idx = i * STEP_SIZE
                end_idx = start_idx + WINDOW_SIZE
                
                if end_idx > len(timestamps_sec):
                    break
                
                pred_class = pred_df.iloc[i]['prediction']
                conf = pred_df.iloc[i]['confidence'] * 100
                start_time = timestamps_sec[start_idx]
                end_time = timestamps_sec[end_idx - 1]
                
                window_data.append({
                    "Pencere #": i + 1,
                    "Zaman Aralığı": f"{start_time:.2f}s - {end_time:.2f}s",
                    "Süre (sn)": f"{end_time - start_time:.2f}",
                    "Tahmin": FAULT_COLORS[pred_class]['emoji'] + " " + FAULT_COLORS[pred_class]['name'],
                    "Güven": f"%{conf:.1f}"
                })
            
            st.dataframe(pd.DataFrame(window_data), use_container_width=True, hide_index=True)
    
    # =========================================================================
    # 9. PERFORMANS METRİKLERİ
    # =========================================================================
    st.subheader("📈 Model Performans Metrikleri")
    
    col1, col2, col3, col4 = st.columns(4)
    
    total_predictions = len(pred_df)
    fault_predictions = len(fault_periods)
    normal_predictions = total_predictions - fault_predictions
    avg_confidence = pred_df['confidence'].mean() if len(pred_df) > 0 else 0
    
    col1.metric(
        "📊 Toplam Pencere", 
        f"{total_predictions:,}",
        delta=f"{fault_predictions} arıza, {normal_predictions} normal"
    )
    
    col2.metric(
        "🎯 Ortalama Güven", 
        f"%{avg_confidence * 100:.1f}",
        delta=f"Std: %{pred_df['confidence'].std() * 100:.1f}"
    )
    
    col3.metric(
        "⚠️ Arıza Oranı", 
        f"%{(fault_predictions/total_predictions)*100:.1f}" if total_predictions > 0 else "%0.0",
        delta=f"{fault_periods['prediction'].nunique()} farklı arıza tipi" if fault_predictions > 0 else "Arıza yok"
    )
    
    col4.metric(
        "⏱️ Uçuş Süresi", 
        f"{flight_duration:.1f} sn",
        delta=f"{total_predictions/flight_duration:.1f} tahmin/sn" if flight_duration > 0 else "0 tahmin/sn"
    )
    
    # =========================================================================
    # 10. SINIF BAZLI PERFORMANS TABLOSU
    # =========================================================================
    st.markdown("### 📋 Sınıf Bazlı Performans Özeti")
    
    if len(pred_df) > 0:
        summary_df = []
        for fault_class in range(config.NUM_CLASSES):
            class_data = pred_df[pred_df['prediction'] == fault_class]
            if len(class_data) > 0:
                fault_name = FAULT_COLORS[fault_class]['emoji'] + " " + FAULT_COLORS[fault_class]['name']
                
                summary_df.append({
                    "Arıza Tipi": fault_name,
                    "Tahmin Sayısı": len(class_data),
                    "Ort. Güven (%)": f"%{class_data['confidence'].mean()*100:.1f}",
                    "Min-Max Güven": f"%{class_data['confidence'].min()*100:.0f} - %{class_data['confidence'].max()*100:.0f}",
                    "İlk Tespit (sn)": f"{class_data['timestamp'].min()/TIME_SCALE:.2f}",
                    "Son Tespit (sn)": f"{class_data['timestamp'].max()/TIME_SCALE:.2f}",
                    "Kapsam (sn)": f"{(class_data['timestamp'].max() - class_data['timestamp'].min())/TIME_SCALE:.2f}"
                })
        
        if summary_df:
            st.dataframe(pd.DataFrame(summary_df), use_container_width=True, hide_index=True)
        else:
            st.info("✅ Tüm uçuş boyunca sadece Normal sınıfı tespit edildi.")
    
    # =========================================================================
    # 11. DETAY TABLOSU & EXPORT
    # =========================================================================
    if len(fault_periods) > 0:
        st.write("**🔍 Tespit Edilen Arıza Ayrıntıları:**")
        fault_details = fault_periods.copy()
        fault_details['Arıza Tipi'] = fault_details['prediction'].map(
            lambda x: FAULT_COLORS[x]["emoji"] + " " + FAULT_COLORS[x]["name"]
        )
        fault_details['Güven Skoru'] = fault_details['confidence'].apply(lambda x: f"%{x * 100:.1f}")
        fault_details['Zaman (sn)'] = fault_details['timestamp'].apply(lambda x: f"{x / TIME_SCALE:.2f}")
        
        st.dataframe(
            fault_details[['Zaman (sn)', 'Arıza Tipi', 'Güven Skoru']],
            use_container_width=True,
            hide_index=True
        )
        
        # CSV Export
        csv = fault_details[['Zaman (sn)', 'Arıza Tipi', 'Güven Skoru']].to_csv(index=False).encode('utf-8-sig')
        st.download_button(
            label="📥 Arıza Raporunu CSV Olarak İndir",
            data=csv,
            file_name=f"fault_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

else:
    st.info("👈 Sol menüden analiz etmek istediğiniz `.csv` dosyasını yükleyerek başlayın.")
    
    st.markdown("""
    ### 💡 Beklenen Çıktı
    
    Dosya yüklendiğinde:
    - **İnteraktif sensör seçimi**: Dropdown ile istediğiniz sensörü seçebilirsiniz
    - **Üst grafik**: Seçilen sensörün sürekli sinyali (mavi çizgi)
    - **Renkli şeritler**: Her kayan pencerenin tahmin ettiği zaman aralığı
      - Renk = Tahmin edilen arıza sınıfı
      - Opaklık = Modelin güven skoru
    - **Alt grafik**: AI tahminleri (renkli noktalar)
    - **Güven grafiği**: Her pencere için güven skoru (yeşil alan)
    - **Kayan Pencere Konsepti**: İlk 5 pencerenin detaylı tablosu
    - **Performans Metrikleri**: Toplam pencere, ortalama güven, arıza oranı
    - **Sınıf Bazlı Tablo**: Her arıza tipi için detaylı istatistikler
    """)