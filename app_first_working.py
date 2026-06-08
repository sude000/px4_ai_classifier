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
    0: {"name": "Normal", "color": "rgba(34, 197, 94, 0.15)", "solid": "#22C55E", "emoji": "✅"},
    1: {"name": "Engine Failure", "color": "rgba(239, 68, 68, 0.35)", "solid": "#EF4444", "emoji": "🔴"},      
    2: {"name": "Elevator Failure", "color": "rgba(249, 115, 22, 0.35)", "solid": "#F97316", "emoji": "🟠"},   
    3: {"name": "Rudder Failure", "color": "rgba(234, 179, 8, 0.35)", "solid": "#EAB308", "emoji": "🟡"},     
    4: {"name": "Aileron Failure", "color": "rgba(22, 163, 74, 0.35)", "solid": "#16A34A", "emoji": "🟢"},      
    5: {"name": "Multi-Fault", "color": "rgba(147, 51, 234, 0.35)", "solid": "#9333EA", "emoji": "🟣"}         
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
    
    return df, timestamps, pred_timestamps, predictions, confidences, time_col

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
        original_df, timestamps, pred_timestamps, predictions, confidences, time_col = preprocess_and_predict(
            uploaded_file, scaler, model, device
        )
    
    pred_df = pd.DataFrame({
        'timestamp': pred_timestamps,
        'prediction': predictions,
        'confidence': confidences
    })
    
    fault_periods = pred_df[pred_df['prediction'] > 0]
    
    # =========================================================================
    # 6. ZAMANSAL VURGULAMA & SÜREKLİ SİNYAL GÖRSELLEŞTİRMESİ
    # =========================================================================
    st.subheader("📊 Uçuş Analizi ve Kayan Pencere Tahminleri")
    
    # Timestamp'i saniyeye çevir (nanosecond -> second)
    TIME_SCALE = 1e9
    timestamps_sec = timestamps / TIME_SCALE
    pred_timestamps_sec = np.array(pred_timestamps) / TIME_SCALE
    
    # Uçuş süresini hesapla
    flight_duration = timestamps_sec[-1] - timestamps_sec[0]
    
    # IMU sensör sütununu bul
    imu_cols = [col for col in original_df.columns if 'angular_velocity.x' in col and 'raw' not in col]
    if imu_cols:
        sensor_col = imu_cols[0]
    else:
        numeric_cols = [col for col in original_df.select_dtypes(include=[np.number]).columns 
                       if col not in [time_col, 'flight_id', 'target_label']]
        sensor_col = numeric_cols[0] if numeric_cols else None
    
    sensor_data = original_df[sensor_col].values if sensor_col else None
    
    # =========================================================================
    # A) ANA GRAFİK: Sürekli Sinyal + Renkli Arka Plan Bölgeleri
    # =========================================================================
    fig = make_subplots(
        rows=2, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.06,
        row_heights=[0.6, 0.4],
        subplot_titles=(
            f"📡 Telemetri Sinyali ({sensor_col.split('.')[-1] if sensor_col else 'Sensor'}) - Süre: {flight_duration:.1f} sn",
            f"🤖 AI Tahmin Güven Skoru (Kayan Pencere: {WINDOW_SIZE} ts, Adım: {STEP_SIZE})"
        )
    )
    
    # 1. Üst Grafik: Sürekli Sensör Sinyali
    if sensor_data is not None:
        fig.add_trace(
            go.Scatter(
                x=timestamps_sec, 
                y=sensor_data, 
                mode='lines', 
                name='IMU X',
                line=dict(color='#1e40af', width=1.5),
                hovertemplate='Zaman: %{x:.2f}s<br>Angular Velocity: %{y:.4f}<extra></extra>'
            ),
            row=1, col=1
        )
    
    # 2. Arıza bölgelerini arka plana renkli dikdörtgenler olarak ekle
    # Ardışık aynı tahminleri birleştirerek "bölge" oluştur
    fault_regions = []
    current_fault = None
    region_start = None
    
    for idx, row in pred_df.sort_values('timestamp').iterrows():
        fault_type = row['prediction']
        fault_time = row['timestamp'] / TIME_SCALE
        
        if fault_type != current_fault:
            # Önceki bölgeyi kapat
            if current_fault is not None and region_start is not None:
                fault_regions.append({
                    'start': region_start,
                    'end': fault_time,
                    'fault': current_fault
                })
            # Yeni bölge başlat
            current_fault = fault_type
            region_start = fault_time
    
    # Son bölgeyi ekle
    if current_fault is not None and region_start is not None:
        fault_regions.append({
            'start': region_start,
            'end': pred_timestamps_sec[-1],
            'fault': current_fault
        })
    
    # Renkli arka planları ekle
    for region in fault_regions:
        color = FAULT_COLORS[region['fault']]["color"]
        fault_name = FAULT_COLORS[region['fault']]['name']
        
        # Üst grafik (sensör) için
        fig.add_shape(
            type="rect",
            x0=region['start'], x1=region['end'],
            y0=0, y1=1,
            fillcolor=color,
            line=dict(width=0),
            layer="below",
            row=1, col=1
        )
        
        # Alt grafik (tahmin) için
        fig.add_shape(
            type="rect",
            x0=region['start'], x1=region['end'],
            y0=0, y1=1,
            fillcolor=color,
            line=dict(width=0),
            layer="below",
            row=2, col=1
        )
        
        # Arıza bölgelerine etiket ekle (sadece üst grafikte)
        if region['fault'] != 0:
            fig.add_annotation(
                x=(region['start'] + region['end']) / 2,
                y=1.02,
                text=FAULT_COLORS[region['fault']]['emoji'] + " " + fault_name,
                showarrow=False,
                font=dict(size=10, color=FAULT_COLORS[region['fault']]['solid']),
                row=1, col=1
            )
    
    # 3. Alt Grafik: Her sınıf için güven skorunu çizgi olarak göster
    # Önce her zaman adımı için "baskın sınıfın güven skorunu" gösteren sürekli bir çizgi
    fig.add_trace(
        go.Scatter(
            x=pred_timestamps_sec,
            y=[c * 100 for c in pred_df['confidence']],
            mode='lines',
            name='Güven Skoru',
            line=dict(color='#6366f1', width=2),
            fill='tozeroy',
            fillcolor='rgba(99, 102, 241, 0.1)',
            hovertemplate='Zaman: %{x:.2f}s<br>Güven: %{y:.1f}%<extra></extra>'
        ),
        row=2, col=1
    )
    
    # Arıza noktalarını vurgulu marker'larla göster
    if len(fault_periods) > 0:
        fault_times_sec = (fault_periods['timestamp'].values / TIME_SCALE)
        fault_confs = fault_periods['confidence'].values * 100
        fault_classes = fault_periods['prediction'].values
        
        # Her fault tipi için ayrı scatter
        for fault_class in range(1, config.NUM_CLASSES):
            mask = fault_classes == fault_class
            if mask.sum() > 0:
                fig.add_trace(
                    go.Scatter(
                        x=fault_times_sec[mask],
                        y=fault_confs[mask],
                        mode='markers',
                        name=FAULT_COLORS[fault_class]['emoji'] + ' ' + FAULT_COLORS[fault_class]['name'],
                        marker=dict(
                            color=FAULT_COLORS[fault_class]['solid'],
                            size=8,
                            symbol='diamond',
                            line=dict(width=1.5, color='white')
                        ),
                        hovertemplate=f'{FAULT_COLORS[fault_class]["name"]}<br>Zaman: %{{x:.2f}}s<br>Güven: %{{y:.1f}}%<extra></extra>'
                    ),
                    row=2, col=1
                )
    
    # Layout ayarları
    fig.update_layout(
        height=750,
        title_text="🔍 Kayan Pencere ile Zamansal Arıza Tespiti",
        title_font_size=22,
        hovermode="x unified",
        legend=dict(
            orientation="h", 
            yanchor="bottom", 
            y=1.02, 
            xanchor="right", 
            x=1,
            bgcolor='rgba(255,255,255,0.9)',
            bordercolor='lightgray',
            borderwidth=1
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        showlegend=True
    )
    
    fig.update_xaxes(title_text="⏱️ Zaman (Saniye)", row=2, col=1, gridcolor='lightgray', showgrid=True)
    fig.update_yaxes(title_text="Angular Velocity (rad/s)", row=1, col=1, gridcolor='lightgray', showgrid=True)
    fig.update_yaxes(title_text="AI Güven Skoru (%)", row=2, col=1, gridcolor='lightgray', range=[0, 105], showgrid=True)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # =========================================================================
    # B) KAYAN PENCERE GÖSTERİMİ (Journal Concept Visualization)
    # =========================================================================
    with st.expander("📐 Kayan Pencere (Sliding Window) Görselleştirmesi", expanded=False):
        st.markdown(f"""
        **Konsept:** {WINDOW_SIZE} timestep'lik pencereler {STEP_SIZE} adım ile kaydırılıyor.  
        **Gerçek Zaman Karşılığı:** Her pencere ≈ **{WINDOW_SIZE/100:.2f} saniye**, her adım ≈ **{STEP_SIZE/100:.2f} saniye**.
        
        > *"120 saniyelik tüm uçuşu modele beslemek yerine, üst üste binen 50 zaman adımlık pencerelere bölüyoruz. 
        > Her pencere bir sınıflandırma alır. Bu bize ince taneli zamansal çözünürlük sağlar."*
        """)
        
        # İlk 5 pencereyi görselleştir
        sample_windows = min(5, len(pred_df))
        
        if sensor_data is not None:
            window_viz = go.Figure()
            
            # Arka planda tüm sinyal (gri)
            window_viz.add_trace(go.Scatter(
                x=timestamps_sec,
                y=sensor_data,
                mode='lines',
                name='Tüm Sinyal',
                line=dict(color='lightgray', width=1)
            ))
            
            # İlk 5 pencereyi renkli göster
            colors = ['#EF4444', '#F97316', '#EAB308', '#22C55E', '#3B82F6']
            
            for i in range(sample_windows):
                window_start_idx = i * STEP_SIZE
                window_end_idx = window_start_idx + WINDOW_SIZE
                
                if window_end_idx > len(timestamps_sec):
                    break
                
                pred_class = pred_df.iloc[i]['prediction']
                conf_score = pred_df.iloc[i]['confidence'] * 100
                
                window_viz.add_trace(go.Scatter(
                    x=timestamps_sec[window_start_idx:window_end_idx],
                    y=sensor_data[window_start_idx:window_end_idx],
                    mode='lines',
                    name=f'Pencere {i+1}: {FAULT_COLORS[pred_class]["emoji"]} {FAULT_COLORS[pred_class]["name"]} (%{conf_score:.1f})',
                    line=dict(color=colors[i % len(colors)], width=3)
                ))
            
            window_viz.update_layout(
                title="İlk 5 Pencerenin Zaman Ekseninde Konumu",
                xaxis_title="Zaman (Saniye)",
                yaxis_title="Sensör Değeri",
                height=400,
                hovermode="x unified",
                plot_bgcolor='white'
            )
            
            st.plotly_chart(window_viz, use_container_width=True)
    
    # =========================================================================
    # C) PERFORMANS METRİKLERİ
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
    # D) SINIF BAZLI PERFORMANS TABLOSU
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
    # E) DETAY TABLOSU & EXPORT
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
    - **Üst grafik**: Sürekli IMU sensör sinyali (mavi çizgi)
    - **Alt grafik**: AI güven skoru (mor alan) + arıza noktaları (renkli elmaslar)
    - **Arka plan bölgeleri**: Her arıza tipi için renkli şeritler
      - 🔴 Kırmızı: Engine Failure
      - 🟠 Turuncu: Elevator Failure
      - 🟡 Sarı: Rudder Failure
      - 🟢 Yeşil: Aileron Failure
      - 🟣 Mor: Multi-Fault
    - **Kayan Pencere Görselleştirmesi**: İlk 5 pencerenin sinyal üzerindeki konumu
    - **Performans Metrikleri**: Toplam pencere, ortalama güven, arıza oranı
    - **Sınıf Bazlı Tablo**: Her arıza tipi için detaylı istatistikler
    """)