import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

import logic

st.set_page_config(page_title="Kargo Dağıtım Optimizasyonu", layout="wide")

st.title("🚚 Kargo Dağıtım Ağı Optimizasyonu")
st.markdown(
    "Gerçek sürüş ağı üzerinde en kısa yol algoritmalarıyla depo çıkışlı rotalar planlayın, "
    "mesafe ve süre maliyetlerini kıyaslayın, teslimat dosyaları yükleyin."
)

DEFAULT_START = (40.9900, 29.0200)
DEFAULT_END = (40.9800, 29.0300)
ALGORITHM_LABELS = {
    "dijkstra": "Dijkstra",
    "bellman-ford": "Bellman-Ford",
}
METRIC_LABELS = {"distance": "Mesafe (metre)", "duration": "Süre (saniye)"}
DEFAULT_DATASET_ID = "6a5b3185-80f7-4d81-9e9e-afef4e894c64"
DEFAULT_RESOURCE_ID = "9788606b-19f1-41c8-bd08-97f4aa952d90"


def parse_drop_points(uploaded_file) -> list:
    df = pd.read_csv(uploaded_file)
    df.columns = [col.strip().lower() for col in df.columns]
    if not {"lat", "lon"}.issubset(df.columns):
        raise ValueError("CSV dosyası 'name (opsiyonel)', 'lat' ve 'lon' kolonlarını içermelidir.")

    points = []
    for idx, row in df.iterrows():
        lat = row["lat"]
        lon = row["lon"]
        if pd.isna(lat) or pd.isna(lon):
            continue
        name = str(row["name"]).strip() if "name" in df.columns else f"Teslimat {idx + 1}"
        points.append({"name": name or f"Teslimat {idx + 1}", "coords": (float(lat), float(lon))})

    if not points:
        raise ValueError("Geçerli teslimat satırı bulunamadı.")
    return points


st.sidebar.header("Ayarlar")

# Popüler bölgeler listesi
POPULAR_PLACES = [
    "Kadıköy, Istanbul, Turkey",
    "Beşiktaş, Istanbul, Turkey",
    "Şişli, Istanbul, Turkey",
    "Üsküdar, Istanbul, Turkey",
    "Fatih, Istanbul, Turkey",
    "Beyoğlu, Istanbul, Turkey",
    "Bakırköy, Istanbul, Turkey",
    "Ataşehir, Istanbul, Turkey",
    "Maltepe, Istanbul, Turkey",
    "Pendik, Istanbul, Turkey",
    "Sarıyer, Istanbul, Turkey",
    "Eyüpsultan, Istanbul, Turkey",
    "Ankara, Turkey",
    "İzmir, Turkey",
    "Bursa, Turkey",
]

# Bölge seçimi: selectbox veya özel giriş
place_option = st.sidebar.selectbox(
    "Bölge Seçimi",
    options=["Hazır Bölgeler"] + POPULAR_PLACES + ["Özel Bölge Gir"],
    index=1,  # Varsayılan: Kadıköy
)

if place_option == "Özel Bölge Gir":
    place_name = st.sidebar.text_input("Özel Bölge Adı", value="Kadıköy, Istanbul, Turkey")
elif place_option == "Hazır Bölgeler":
    place_name = "Kadıköy, Istanbul, Turkey"  # Varsayılan
else:
    place_name = place_option

use_cache = st.sidebar.checkbox("OSM verilerini önbellekten yükle", value=True)

if "graph" not in st.session_state or st.session_state.get("place_name") != place_name or st.session_state.get("use_cache") != use_cache:
    with st.spinner(f"{place_name} için harita verisi hazırlanıyor..."):
        st.session_state.graph = logic.get_graph(place_name, use_cache=use_cache)
        st.session_state.place_name = place_name
        st.session_state.use_cache = use_cache
        # Yeni harita yüklendiğinde eski rota sonuçlarını temizle
        st.session_state.route_result = None
        st.session_state.multi_result = None
    if st.session_state.graph is None:
        st.error("Harita verisi alınamadı. Lütfen geçerli bir bölge ismi girin.")
        st.stop()

graph = st.session_state.graph

# Bölge değiştiğinde koordinatları otomatik güncelle
if "current_place" not in st.session_state or st.session_state.current_place != place_name:
    st.session_state.current_place = place_name
    # Graf merkezini hesapla
    nodes = list(graph.nodes(data=True))
    if nodes:
        center_lat = sum(data['y'] for _, data in nodes) / len(nodes)
        center_lon = sum(data['x'] for _, data in nodes) / len(nodes)
        # Başlangıç ve bitiş noktalarını merkeze yakın ayarla
        st.session_state.default_start = (center_lat + 0.001, center_lon + 0.001)
        st.session_state.default_end = (center_lat - 0.001, center_lon - 0.001)
    else:
        st.session_state.default_start = DEFAULT_START
        st.session_state.default_end = DEFAULT_END

# Varsayılan koordinatları session state'ten al
default_start = st.session_state.get("default_start", DEFAULT_START)
default_end = st.session_state.get("default_end", DEFAULT_END)

st.sidebar.subheader("Rota Planlama")
st.sidebar.caption(f"🗺️ Seçili bölge: {place_name}")

# Haritadan konum seçme modu
use_map_selection = st.sidebar.checkbox("Haritadan konum seç (tıkla)", value=False)

if use_map_selection:
    st.sidebar.info("📍 Haritaya tıklayarak başlangıç ve bitiş noktalarını seçin")
    if "map_start" not in st.session_state:
        st.session_state.map_start = default_start
    if "map_end" not in st.session_state:
        st.session_state.map_end = default_end

    start_lat, start_lon = st.session_state.map_start
    end_lat, end_lon = st.session_state.map_end

    col1, col2 = st.sidebar.columns(2)
    if col1.button("Başlangıcı seç"):
        st.session_state.selecting = "start"
    if col2.button("Bitişi seç"):
        st.session_state.selecting = "end"

    st.sidebar.text(f"Başlangıç: {start_lat:.4f}, {start_lon:.4f}")
    st.sidebar.text(f"Bitiş: {end_lat:.4f}, {end_lon:.4f}")
else:
    start_lat = st.sidebar.number_input("Başlangıç Enlemi", value=default_start[0], format="%.4f")
    start_lon = st.sidebar.number_input("Başlangıç Boylamı", value=default_start[1], format="%.4f")
    end_lat = st.sidebar.number_input("Varış Enlemi", value=default_end[0], format="%.4f")
    end_lon = st.sidebar.number_input("Varış Boylamı", value=default_end[1], format="%.4f")

st.sidebar.markdown("---")

# Algoritma karşılaştırma modu
compare_mode = st.sidebar.checkbox("Algoritmaları karşılaştır", value=False)

if compare_mode:
    st.sidebar.info("📊 Tüm algoritmalar aynı anda çalışacak")
    algorithm_choice = "dijkstra"  # Varsayılan
else:
    algorithm_choice = st.sidebar.selectbox("Algoritma", options=list(ALGORITHM_LABELS.keys()), format_func=lambda key: ALGORITHM_LABELS[key])

metric_choice = st.sidebar.selectbox("Optimizasyon Hedefi", options=list(METRIC_LABELS.keys()), format_func=lambda key: METRIC_LABELS[key])
traffic_multiplier = st.sidebar.slider("Trafik Katsayısı (gerçek zamanlı veriye göre ayarlayın)", min_value=0.5, max_value=3.0, value=1.0, step=0.1)

st.sidebar.markdown("---")
st.sidebar.subheader("Teslimat Noktaları")

# Teslimat ekleme seçenekleri
delivery_mode = st.sidebar.radio(
    "Teslimat noktası ekleme yöntemi:",
    ["Manuel (Haritadan seç)", "CSV Dosyası", "Hızlı Test (3-4 nokta)"]
)

# Session state için teslimat noktaları
if "manual_deliveries" not in st.session_state:
    st.session_state.manual_deliveries = []

drop_points = []

if delivery_mode == "Manuel (Haritadan seç)":
    st.sidebar.info("📦 Haritaya tıklayarak teslimat noktası ekleyin")

    if st.sidebar.button("➕ Yeni teslimat noktası ekle"):
        st.session_state.selecting = "delivery"

    if st.sidebar.button("🗑️ Tüm teslimatları temizle"):
        st.session_state.manual_deliveries = []
        st.rerun()

    drop_points = st.session_state.manual_deliveries
    if drop_points:
        st.sidebar.success(f"✅ {len(drop_points)} teslimat noktası eklendi")
        for idx, point in enumerate(drop_points, 1):
            st.sidebar.text(f"{idx}. {point['name']}")

elif delivery_mode == "CSV Dosyası":
    st.sidebar.caption("Format örneği: name,lat,lon")
    sample_csv = "name,lat,lon\nMüşteri 1,40.9855,29.0251\nMüşteri 2,40.9750,29.0400"
    st.sidebar.download_button("Örnek CSV indir", sample_csv, file_name="ornek_teslimat.csv", mime="text/csv")
    uploaded_file = st.sidebar.file_uploader("Teslimat dosyası yükle", type="csv")

    if uploaded_file is not None:
        try:
            drop_points = parse_drop_points(uploaded_file)
            uploaded_file.seek(0)
            st.sidebar.success(f"{len(drop_points)} teslimat yüklendi.")
        except Exception as exc:
            st.sidebar.error(f"Teslimatlar okunamadı: {exc}")

elif delivery_mode == "Hızlı Test (3-4 nokta)":
    st.sidebar.info("🚀 Demo için örnek teslimat noktaları")
    # Kadıköy çevresinde örnek noktalar
    center_lat = default_start[0]
    center_lon = default_start[1]

    drop_points = [
        {"name": "Teslimat 1", "coords": (center_lat + 0.002, center_lon + 0.003)},
        {"name": "Teslimat 2", "coords": (center_lat - 0.003, center_lon + 0.002)},
        {"name": "Teslimat 3", "coords": (center_lat + 0.001, center_lon - 0.004)},
        {"name": "Teslimat 4", "coords": (center_lat - 0.002, center_lon - 0.002)},
    ]
    st.sidebar.success(f"✅ {len(drop_points)} örnek teslimat oluşturuldu")

optimize_order = st.sidebar.checkbox("Teslimat sırasını optimize et", value=True)
return_to_depot = st.sidebar.checkbox("Teslimat sonrası depoya dön", value=False)

start_coords = (start_lat, start_lon)
end_coords = (end_lat, end_lon)

# Session state'te rota sonuçlarını sakla
if "route_result" not in st.session_state:
    st.session_state.route_result = None
if "multi_result" not in st.session_state:
    st.session_state.multi_result = None
if "comparison_results" not in st.session_state:
    st.session_state.comparison_results = None

if st.sidebar.button("Rotayı Hesapla"):
    with st.spinner("Rota hesaplanıyor..."):
        if compare_mode and drop_points:
            # Çoklu teslimat için algoritma karşılaştırması
            st.session_state.comparison_results = {}
            for alg in ALGORITHM_LABELS.keys():
                result = logic.plan_multi_stop_route(
                    graph,
                    start_coords,
                    drop_points,
                    algorithm=alg,
                    metric=metric_choice,
                    traffic_multiplier=traffic_multiplier,
                    optimize_order=optimize_order,
                    return_to_depot=return_to_depot,
                )
                if result:
                    st.session_state.comparison_results[alg] = result
            st.session_state.route_result = None
            st.session_state.multi_result = None
        elif compare_mode and not drop_points:
            # Tek nokta için algoritma karşılaştırması
            st.session_state.comparison_results = {}
            for alg in ALGORITHM_LABELS.keys():
                result = logic.calculate_route(
                    graph,
                    start_coords,
                    end_coords,
                    algorithm=alg,
                    metric=metric_choice,
                    traffic_multiplier=traffic_multiplier,
                )
                if result:
                    st.session_state.comparison_results[alg] = result
            st.session_state.route_result = None
            st.session_state.multi_result = None
        elif drop_points:
            st.session_state.multi_result = logic.plan_multi_stop_route(
                graph,
                start_coords,
                drop_points,
                algorithm=algorithm_choice,
                metric=metric_choice,
                traffic_multiplier=traffic_multiplier,
                optimize_order=optimize_order,
                return_to_depot=return_to_depot,
            )
            st.session_state.route_result = None
            st.session_state.route_result = None
            st.session_state.comparison_results = None
        else:
            st.session_state.route_result = logic.calculate_route(
                graph,
                start_coords,
                end_coords,
                algorithm=algorithm_choice,
                metric=metric_choice,
                traffic_multiplier=traffic_multiplier,
            )
            st.session_state.multi_result = None
            st.session_state.comparison_results = None

# Harita için rota ve koordinatları hazırla
map_route = None
if st.session_state.comparison_results:
    # Karşılaştırma modunda ilk algoritmayı göster
    first_alg = list(st.session_state.comparison_results.keys())[0]
    map_route = st.session_state.comparison_results[first_alg]["route"]
elif st.session_state.multi_result:
    map_route = st.session_state.multi_result["route"]
elif st.session_state.route_result:
    map_route = st.session_state.route_result["route"]

map_end_coords = None
if drop_points:
    if st.session_state.multi_result and st.session_state.multi_result["segments"]:
        map_end_coords = st.session_state.multi_result["segments"][-1]["end_coords"]
    else:
        map_end_coords = drop_points[-1]["coords"]
else:
    map_end_coords = end_coords

map_obj = logic.create_map(
    graph,
    route=map_route,
    start_coords=start_coords,
    end_coords=map_end_coords,
    drop_points=drop_points if drop_points else None,
)

# Haritayı göster ve tıklama olaylarını yakala
map_data = st_folium(map_obj, width=1000, height=600, key="main_map")

# Haritadan konum seçimi aktifse ve tıklama yapıldıysa
if map_data and map_data.get("last_clicked"):
    clicked_lat = map_data["last_clicked"]["lat"]
    clicked_lon = map_data["last_clicked"]["lng"]

    if use_map_selection and st.session_state.get("selecting") == "start":
        st.session_state.map_start = (clicked_lat, clicked_lon)
        st.session_state.selecting = None
        st.rerun()
    elif use_map_selection and st.session_state.get("selecting") == "end":
        st.session_state.map_end = (clicked_lat, clicked_lon)
        st.session_state.selecting = None
        st.rerun()
    elif st.session_state.get("selecting") == "delivery":
        # Yeni teslimat noktası ekle
        delivery_num = len(st.session_state.manual_deliveries) + 1
        st.session_state.manual_deliveries.append({
            "name": f"Teslimat {delivery_num}",
            "coords": (clicked_lat, clicked_lon)
        })
        st.session_state.selecting = None
        st.rerun()

st.markdown("---")

# Algoritma karşılaştırma sonuçları
if st.session_state.comparison_results:
    st.success("🔬 Algoritma Karşılaştırması Tamamlandı")

    # Karşılaştırma tablosu
    comparison_data = []
    for alg, result in st.session_state.comparison_results.items():
        row_data = {
            "Algoritma": ALGORITHM_LABELS[alg],
            "Mesafe (km)": f"{result['distance_m']/1000:.2f}",
            "Süre (dk)": f"{result['duration_s']/60:.1f}",
            "Node Sayısı": len(result['route']),
        }

        # Çoklu teslimat varsa teslimat sırasını da göster
        if 'segments' in result:
            row_data["Teslimat Sayısı"] = len(result['segments'])

        comparison_data.append(row_data)

    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df, use_container_width=True)

    # En iyi algoritmaları vurgula
    min_distance = min(float(row["Mesafe (km)"]) for row in comparison_data)
    min_time = min(float(row["Süre (dk)"]) for row in comparison_data)

    cols_metrics = st.columns(3)
    cols_metrics[0].metric("En Kısa Mesafe", f"{min_distance:.2f} km", "🏆")
    cols_metrics[1].metric("En Hızlı Süre", f"{min_time:.1f} dk", "⚡")

    # Algoritmaların farkını göster
    if len(comparison_data) > 1:
        max_distance = max(float(row["Mesafe (km)"]) for row in comparison_data)
        max_time = max(float(row["Süre (dk)"]) for row in comparison_data)
        distance_diff = ((max_distance - min_distance) / min_distance * 100) if min_distance > 0 else 0
        time_diff = ((max_time - min_time) / min_time * 100) if min_time > 0 else 0
        cols_metrics[2].metric("Maksimum Fark", f"{max(distance_diff, time_diff):.1f}%", "📊")

    # Her algoritma için harita göster
    st.subheader("Rotaların Görsel Karşılaştırması")
    cols = st.columns(len(st.session_state.comparison_results))

    for idx, (alg, result) in enumerate(st.session_state.comparison_results.items()):
        with cols[idx]:
            st.markdown(f"**{ALGORITHM_LABELS[alg]}**")

            # Teslimat noktalarını da göster
            if 'segments' in result:
                mini_map = logic.create_map(
                    graph,
                    route=result["route"],
                    start_coords=start_coords,
                    end_coords=result["segments"][-1]["end_coords"] if result["segments"] else end_coords,
                    drop_points=drop_points,
                )
            else:
                mini_map = logic.create_map(
                    graph,
                    route=result["route"],
                    start_coords=start_coords,
                    end_coords=end_coords,
                )
            st_folium(mini_map, width=300, height=300, key=f"map_{alg}")

            # Teslimat sırasını göster
            if 'order' in result:
                st.caption(f"Sıra: {' → '.join(result['order'][:3])}{'...' if len(result['order']) > 3 else ''}")

    st.info("💡 **İpucu**: Dijkstra genellikle en hızlı sonucu verirken her iki algoritma da optimal rotayı bulur. Bellman-Ford negatif ağırlıklı graflar için de kullanılabilir.")

elif st.session_state.route_result:
    st.success("Tek noktalı rota hazır.")
    cols = st.columns(3)
    cols[0].metric("Toplam Mesafe", f"{st.session_state.route_result['distance_m']/1000:.2f} km")
    cols[1].metric("Tahmini Süre", f"{st.session_state.route_result['duration_s']/60:.1f} dk")
    cols[2].metric("Algoritma", ALGORITHM_LABELS[algorithm_choice])

    route_df = pd.DataFrame(logic.route_to_coordinates(graph, st.session_state.route_result["route"]))
    st.download_button("Rota koordinatlarını CSV olarak indir", route_df.to_csv(index=False).encode("utf-8"), "rota.csv", "text/csv")

elif st.session_state.multi_result:
    st.success("Çok noktalı rota hazır.")
    cols = st.columns(3)
    cols[0].metric("Toplam Mesafe", f"{st.session_state.multi_result['distance_m']/1000:.2f} km")
    cols[1].metric("Tahmini Süre", f"{st.session_state.multi_result['duration_s']/60:.1f} dk")
    cols[2].metric("Ziyaret sayısı", f"{len(st.session_state.multi_result['segments'])}")

    detail_df = pd.DataFrame(
        [
            {
                "Nokta": seg["name"],
                "Mesafe (km)": seg["distance_m"] / 1000,
                "Süre (dk)": seg["duration_s"] / 60,
            }
            for seg in st.session_state.multi_result["segments"]
        ]
    )
    st.dataframe(detail_df, use_container_width=True)

    order_text = " ➜ ".join(st.session_state.multi_result["order"]) if "order" in st.session_state.multi_result else ""
    st.caption(f"Teslimat sırası: {order_text}")

    route_df = pd.DataFrame(logic.route_to_coordinates(graph, st.session_state.multi_result["route"]))
    st.download_button("Birleştirilmiş rota koordinatlarını indir", route_df.to_csv(index=False).encode("utf-8"), "cok_noktali_rota.csv", "text/csv")

else:
    st.info("Sol menüden parametreleri seçip 'Rotayı Hesapla' butonuna basın.")

st.markdown("### Nasıl Kullanılır?")
st.markdown("1. Bölgeyi girin ve depoyu temsil eden koordinatları set edin.")
st.markdown("2. CSV yükleyerek çoklu teslimat noktaları ekleyebilir veya tek hedef için koordinatlar girebilirsiniz.")
st.markdown("3. Algoritma ve optimizasyon hedefini seçerek trafiğe göre katsayı belirleyin.")
st.markdown("4. 'Rotayı Hesapla' seçeneğiyle rotayı ve performans metriklerini görüntüleyin.")

st.markdown("---")
st.subheader("İBB Trafik Yoğunluk Haritası Kaynağı")

with st.expander("CKAN metadata doğrulaması", expanded=False):
    dataset_id = st.text_input("Dataset (package) ID", value=DEFAULT_DATASET_ID, key="dataset_input")
    resource_id = st.text_input("Resource ID", value=DEFAULT_RESOURCE_ID, key="resource_input")

    if "ckan_dataset" not in st.session_state:
        st.session_state.ckan_dataset = None
    if "ckan_resource" not in st.session_state:
        st.session_state.ckan_resource = None

    if st.button("CKAN üzerinden getir", key="fetch_ckan"):
        try:
            with st.spinner("İBB CKAN üzerinden metadata alınıyor..."):
                st.session_state.ckan_dataset = logic.fetch_ckan_dataset(dataset_id)
                st.session_state.ckan_resource = logic.fetch_ckan_resource(resource_id)
            st.success("Metadata başarıyla alındı.")
        except Exception as exc:
            st.session_state.ckan_dataset = None
            st.session_state.ckan_resource = None
            st.error(f"CKAN isteği başarısız: {exc}")

    if st.session_state.get("ckan_dataset"):
        dataset = st.session_state.ckan_dataset
        st.markdown("#### Dataset Bilgileri")
        st.write(
            {
                "Başlık": dataset.get("title"),
                "Son Güncellenme": dataset.get("metadata_modified"),
                "Lisans": dataset.get("license_title"),
                "Organizasyon": dataset.get("organization", {}).get("title"),
                "Etiketler": [tag["name"] for tag in dataset.get("tags", [])],
            }
        )
        resources = dataset.get("resources", [])
        if resources:
            st.markdown("#### Mevcut Kaynaklar")
            resource_df = pd.DataFrame(
                [
                    {
                        "Ad": res.get("name"),
                        "Format": res.get("format"),
                        "Güncellenme": res.get("last_modified") or res.get("created"),
                        "URL": res.get("url"),
                    }
                    for res in resources
                ]
            )
            st.dataframe(resource_df, use_container_width=True)

    if st.session_state.get("ckan_resource"):
        resource = st.session_state.ckan_resource
        st.markdown("#### Seçili Resource")
        st.write(
            {
                "Ad": resource.get("name"),
                "Format": resource.get("format"),
                "Durum": resource.get("state"),
                "Son Güncellenme": resource.get("last_modified"),
                "İndirme URL": resource.get("url"),
            }
        )
        if resource.get("url"):
            st.markdown(f"[Kaynağı Aç]({resource['url']})")
