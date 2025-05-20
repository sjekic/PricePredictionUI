import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import json
import datetime

model = joblib.load("xgb_best_model.joblib")
feature_names = model.get_booster().feature_names

def save_feedback(prediction_value, accuracy_rating, feedback_text=None):
    """
    Save user feedback to a flat file on disk.
    
    Parameters:
    - prediction_value: The price that was predicted
    - accuracy_rating: Rating of prediction accuracy
    - feedback_text: Optional text feedback
    """
    timestamp = datetime.datetime.now().isoformat()
    
    feedback_data = {
        "timestamp": timestamp,
        "predicted_price": float(prediction_value),
        "accuracy_rating": accuracy_rating,
        "comments": feedback_text,
        "session_id": st.session_state.get("session_id", "unknown"),
        "input_data": {k: (float(v) if isinstance(v, (int, float, np.number)) else v) 
                    for k, v in input_data.items() if k in feature_names}
    }
    
    os.makedirs("feedback", exist_ok=True)


    feedback_file = "feedback/price_predictor_feedback.jsonl"
    with open(feedback_file, "a") as f:
        f.write(json.dumps(feedback_data) + "\n")
    
    return True

if 'show_feedback' not in st.session_state:
    st.session_state.show_feedback = False
if 'accuracy_rating' not in st.session_state:
    st.session_state.accuracy_rating = "Select an option"
if 'feedback_text' not in st.session_state:
    st.session_state.feedback_text = ""
if 'feedback_submitted' not in st.session_state:
    st.session_state.feedback_submitted = False
if 'session_id' not in st.session_state:
    st.session_state.session_id = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

if 'show_feedback' not in st.session_state:
    st.session_state.show_feedback = False
if 'accuracy_rating' not in st.session_state:
    st.session_state.accuracy_rating = "Select an option"
if 'feedback_text' not in st.session_state:
    st.session_state.feedback_text = ""
if 'feedback_submitted' not in st.session_state:
    st.session_state.feedback_submitted = False
if 'session_id' not in st.session_state:
    st.session_state.session_id = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

def update_accuracy(value):
    st.session_state.accuracy_rating = value

def update_feedback_text(value):
    st.session_state.feedback_text = value

def submit_feedback():
    if st.session_state.accuracy_rating == "Select an option":
        st.session_state.feedback_error = True
        return
    
    save_feedback(
        st.session_state.last_prediction,
        st.session_state.accuracy_rating,
        st.session_state.feedback_text
    )
    
    st.session_state.feedback_submitted = True
    st.session_state.feedback_error = False
    st.session_state.accuracy_rating = "Select an option"
    st.session_state.feedback_text = ""

st.title("💻 Computer Price Predictor")
st.text("Welcome to our final Machine Learning Foundations project! \n We have built a model that can predict the price of a laptop based on certain specifics like RAM, GPU, CPU, brand, color and so much more! \n Select the features your dream laptop would have and get a price prediction.")

input_data = {feature: 0 for feature in feature_names}

st.subheader("Device Type")
tipo_features = [
    "tipo_Desktop","tipo_Laptop","tipo_nan"
]

tipo_producto_features = [
    'tipo_producto_Barebone', 'tipo_producto_Chromebook', 'tipo_producto_Kit ampliación PC', 'tipo_producto_Mini PC', 'tipo_producto_Netbook', 'tipo_producto_PC completo', 'tipo_producto_PC de oficina', 'tipo_producto_PC gaming', 'tipo_producto_PC multimedia', 'tipo_producto_Portátil 3D', 'tipo_producto_Portátil convertible', 'tipo_producto_Portátil gaming', 'tipo_producto_Portátil multimedia', 'tipo_producto_Portátil profesional', 'tipo_producto_Thin Client', 'tipo_producto_Ultrabook', 'tipo_producto_Workstation', 'tipo_producto_nan'
]   

tipo_names = [f.replace("tipo_", "") for f in tipo_features]
selected_tipo = st.selectbox("Select type of device", options=tipo_names)
for t in tipo_features:
    input_data[t] = 0
input_data[f"tipo_{selected_tipo}"] = 1

tipo_producto_names = [f.replace("tipo_producto_", "") for f in tipo_producto_features]
selected_tipo_producto = st.selectbox("Select product type", options=tipo_producto_names)
for p in tipo_producto_features:
    input_data[p] = 0
input_data[f"tipo_producto_{selected_tipo_producto}"] = 1


st.subheader("Brand")
company_name_map = {
    0: "ASRock", 1: "ASUS", 2: "AWow", 3: "Abra", 4: "Acemagic", 5: "Acemagician", 6: "Acer", 7: "Actina",
    8: "Adonia", 9: "Advance", 10: "Alienware", 11: "Alurin", 12: "Ankermann", 13: "Apple", 14: "Arc", 
    15: "BEASTCOM", 16: "BMAX", 17: "Basic", 18: "Basilisk", 19: "Beelink", 20: "Blackview", 21: "Captiva",
    22: "Chuwi", 23: "Concept", 24: "Corsair", 25: "Deep", 26: "DeepGaming", 27: "Dell", 28: "Denver",
    29: "DeskMini", 30: "Edge", 31: "Elitegroup", 32: "Ernitec", 33: "F", 34: "Force", 35: "Fujitsu",
    36: "FutureNUC", 37: "GPD", 38: "GREED", 39: "Gaming", 40: "Geekom", 41: "GemiBook", 42: "GigaByte",
    43: "Gold", 44: "HP", 45: "Huawei", 46: "Hyrican", 47: "IT", 48: "Innjoo", 49: "Intel", 50: "Iox",
    51: "Ioxbook", 52: "Joule", 53: "Kiebel", 54: "LG", 55: "Lenovo", 56: "Leotec", 57: "Lite", 58: "MSI",
    59: "Mars", 60: "MeLE", 61: "Medion", 62: "Memory", 63: "Microsoft", 64: "Minisforum", 65: "Minix",
    66: "Neo", 67: "Nexus", 68: "NiPoGi", 69: "Nilox", 70: "Ninkear", 71: "Nitropc", 72: "NucBox", 73: "Nuke",
    74: "Office", 75: "Orbsmart", 76: "Ouvis", 77: "PC", 78: "PcCom", 79: "Plus", 80: "Prime", 81: "Primux",
    82: "Prixton", 83: "Pro", 84: "Quieter", 85: "Racing", 86: "Razer", 87: "Samsung", 88: "Schenker",
    89: "ScreenOn", 90: "Sedatech", 91: "Shuttle", 92: "Silver", 93: "Striker", 94: "SuperMicro", 95: "TB",
    96: "Techbite", 97: "Technologies", 98: "Tecra", 99: "Thomson", 100: "Toughline", 101: "Tulpar", 
    102: "U", 103: "VIST", 104: "Venom", 105: "Vibox", 106: "Viewsonic", 107: "Vision", 108: "X", 
    109: "Xiaomi", 110: "Yashi", 111: "Zenith", 112: "Zone", 113: "Zotac", 114: "iggual", 115: "iiyama"
}
brand_to_label = {v: k for k, v in company_name_map.items()}
selected_brand = st.selectbox("Select Brand", options=sorted(brand_to_label.keys()))
input_data["company_name_label"] = brand_to_label[selected_brand]


st.subheader("RAM")
st.text("RAM stands for Random Access Memory and it is a volatile type of memory, meaning its data gets deleted every time we turn off the computer.")
ram_features = [
    'ram_tipo_ram_DDR3', 'ram_tipo_ram_DDR3L', 'ram_tipo_ram_DDR4', 'ram_tipo_ram_DDR4L', 'ram_tipo_ram_DDR5', 'ram_tipo_ram_LPDDR3', 'ram_tipo_ram_LPDDR4', 'ram_tipo_ram_LPDDR4X', 'ram_tipo_ram_LPDDR5', 'ram_tipo_ram_LPDDR5X', 'ram_tipo_ram_nan'
]
ram_names = [f.replace("ram_tipo_ram_", "") for f in ram_features]
selected_ram = st.selectbox("Select type of RAM", options=ram_names)
for r in ram_features:
    input_data[r] = 0
input_data[f"ram_tipo_ram_{selected_ram}"] = 1

ram_capacity = st.number_input(
    "RAM capacity (GB)",
    min_value=0,
    max_value=256,
    step=4,
    value=16  
)
input_data["ram_memoria_ram_GB"] = int(ram_capacity)

ram_freq = st.number_input(
    "RAM frequency (MHz)",
    min_value=1600,
    max_value=8533,
    step=100,
    value=3200  
)
input_data["ram_frecuencia_memoria_MHz"] = int(ram_freq)


st.subheader("Operating System")
st.text("OS stands for Operating System. It is crucial to process, memory, file system and device management, as well as user interface and security and access")
os_features = [
    'os_DOS', 'os_No OS', 'os_Other OS', 'os_Windows', 'os_macOS'
]
os_names = [f.replace("os_", "") for f in os_features]
selected_os = st.selectbox("Select OS", options=os_names)
for o in os_features:
    input_data[o] = 0
input_data[f"os_{selected_os}"] = 1

st.subheader("Color")
color_features = [
    "color_azul", "color_blanco", "color_bronce", "color_dorado", "color_gris",
    "color_negro", "color_plateado", "color_rojo", "color_rosa", "color_verde", "color_nan"
]

color_translation = {
    "color_azul": "Blue",
    "color_blanco": "White",
    "color_bronce": "Bronze",
    "color_dorado": "Gold",
    "color_gris": "Gray",
    "color_negro": "Black",
    "color_plateado": "Silver",
    "color_rojo": "Red",
    "color_rosa": "Pink",
    "color_verde": "Green",
    "color_nan": "Unknown"
}

english_labels = [color_translation[feat] for feat in color_features]
reverse_translation = {v: k for k, v in color_translation.items()}

selected_color_english = st.selectbox("Choose a color:", english_labels)
selected_color_spanish = reverse_translation[selected_color_english]


for color_feature in color_features:
    input_data[color_feature] = 0
input_data[selected_color_spanish] = 1

st.subheader("Monitor")

screen_size_inch = st.number_input(
    "Screen Size (inches)", min_value=5.0, max_value=40.0, value=15.6, step=0.1, format="%.1f"
)
input_data["pantalla_tamaño_pantalla_pulgadas"] = float(screen_size_inch)

screen_diagonal_cm = st.number_input(
    "Screen Diagonal (cm)", min_value=12.0, max_value=100.0, value=39.6, step=0.1, format="%.1f"
)
input_data["pantalla_diagonal_pantalla_cm"] = float(screen_diagonal_cm)

brightness = st.number_input(
    "Brightness (cd/m²)", min_value=100, max_value=1500, value=300, step=50
)
input_data["pantalla_luminosidad_cd_m2"] = int(brightness)

resolution_map = {
    0: "2,2K",
    1: "2.5K",
    2: "2.8K",
    3: "2K",
    4: "3,2K",
    5: "3K",
    6: "4K",
    7: "FHD+",
    8: "Full HD",
    9: "HD Ready",
    10: "HD+",
    11: "Missing_value",
    12: "QHD",
    13: "QHD+",
    14: "Retina",
    15: "UHD+",
    16: "Ultra HD",
    17: "WQHD",
    18: "WQUXGA",
    19: "WQXGA",
    20: "WQXGA+",
    21: "WUXGA",
    22: "WUXGA+",
    23: "WXGA+"
}

screen_tech_label = st.selectbox("Screen Technology", list(resolution_map.values()))

screen_tech = list(resolution_map.keys())[list(resolution_map.values()).index(screen_tech_label)]
input_data["pantalla_tecnología_pantalla_label"] = int(screen_tech)


st.subheader("Hard Disk")
st.text("The hard disk (HDD) is a traditional storage device used in computers that uses spinning magnetic disks, while the solid state drive (SSD) doesn't have moving parts and uses integrated circuits to store data electronically")
disco_duro_features = [
    'disco_duro_tipo_disco_duro_PCIe SSD', 'disco_duro_tipo_disco_duro_SATA', 'disco_duro_tipo_disco_duro_SSD', 'disco_duro_tipo_disco_duro_disco duro HDD', 'disco_duro_tipo_disco_duro_disco duro M.2 SSD', 'disco_duro_tipo_disco_duro_disco duro SSD', 'disco_duro_tipo_disco_duro_disco híbrido (HHD)', 'disco_duro_tipo_disco_duro_memoria flash', 'disco_duro_tipo_disco_duro_sin disco duro', 'disco_duro_tipo_disco_duro_nan'
]
disco_duro_names = [f.replace("disco_duro_tipo_disco_duro_", "") for f in disco_duro_features]
selected_disco_duro = st.selectbox("Select type of hard disk", options=disco_duro_names)
for d in disco_duro_features:
    input_data[d] = 0
input_data[f"disco_duro_tipo_disco_duro_{selected_disco_duro}"] = 1

ssd_capacity = st.number_input(
    "SSD capacity (GB)",
    min_value=8,
    max_value=8000,
    step=128,
    value=1000  
)
input_data["disco_duro_capacidad_memoria_ssd_GB"] = float(ssd_capacity)

num_disks = st.number_input(
    "Number of installed hard drives", min_value=0, max_value=3, step=1, value=1
)
input_data["disco_duro_número_discos_duros_instalados"] = float(num_disks)

with st.expander("See more configuration options"):
    st.subheader("Processor")
    st.text("The processor is also known as the CPU and it's the brain of the computer.")
    
    procesador_name_map = {
    0: "AMD 3000", 1: "AMD A-Series", 2: "AMD A10", 3: "AMD Athlon", 4: "AMD Athlon 3150U",
    5: "AMD GX", 6: "AMD Ryzen", 7: "AMD Ryzen 3", 8: "AMD Ryzen 3 4300GE", 9: "AMD Ryzen 5",
    10: "AMD Ryzen 5 4600G", 11: "AMD Ryzen 5 7535U", 12: "AMD Ryzen 7", 13: "AMD Ryzen 7 3700U",
    14: "AMD Ryzen 7 5700U", 15: "AMD Ryzen 7 7435HS", 16: "AMD Ryzen 9", 17: "AMD Ryzen 9 5900X",
    18: "AMD Ryzen AI 7", 19: "AMD Ryzen AI 9", 20: "AMD Ryzen Embedded",
    21: "AMD Ryzen Threadripper PRO", 22: "AMD Ryzen Threadripper PRO 5955WX",
    23: "ARM Cortex", 24: "Apple M1", 25: "Apple M1 Pro", 26: "Apple M1 Ultra", 27: "Apple M2",
    28: "Apple M2 Max", 29: "Apple M2 Pro", 30: "Apple M2 Ultra", 31: "Apple M3",
    32: "Apple M3 Max", 33: "Apple M3 Pro", 34: "Apple M4", 35: "Apple M4 Max",
    36: "Apple M4 Pro", 37: "Intel Atom", 38: "Intel Atom D525", 39: "Intel Atom E3815",
    40: "Intel Celeron", 41: "Intel Celeron G1620", 42: "Intel Celeron N4000",
    43: "Intel Celeron N4120", 44: "Intel Core Ultra 5", 45: "Intel Core Ultra 7",
    46: "Intel Core Ultra 9", 47: "Intel Core i3", 48: "Intel Core i5", 49: "Intel Core i7",
    50: "Intel Core i9", 51: "Intel N", 52: "Intel Pentium", 53: "Intel Pentium 6405U",
    54: "Intel Pentium Gold", 55: "Intel Pentium N3700", 56: "Intel Pentium Silver",
    57: "Intel Xeon", 58: "Intel Xeon E5", 59: "Intel Xeon Silver 4210R",
    60: "MediaTek Kompanio", 61: "MediaTek MT8183", 62: "Missing_value", 63: "Qualcomm Kryo",
    64: "Qualcomm Snapdragon", 65: "Qualcomm Snapdragon 7180c",
    66: "Qualcomm Snapdragon 8cx Gen3", 67: "Qualcomm Snapdragon X Elite",
    68: "Qualcomm Snapdragon X Plus", 69: "Qualcomm Snapdragon X Plus X1P",
    70: "RockChip RK3368", 71: "VIA Eden"
    }
    procesador_to_label = {v: k for k, v in procesador_name_map.items()}
    selected_processor = st.selectbox("Select Processor", options=sorted(procesador_to_label.keys()))
    input_data["procesador_name_label"] = procesador_to_label[selected_processor]

    st.text("Processor Cache")

    # Cache presence selections with radio buttons to ensure one is selected
    cache_options = ["L2", "L3", "None"]
    selected_cache = st.radio("Cache Level", cache_options)
    
    # Set all cache features to 0
    input_data['procesador_nivel_caché_L2'] = 0
    input_data['procesador_nivel_caché_L3'] = 0
    input_data['procesador_nivel_caché_nan'] = 0
    
    # Set the selected cache feature to 1
    if selected_cache == "L2":
        input_data['procesador_nivel_caché_L2'] = 1
    elif selected_cache == "L3":
        input_data['procesador_nivel_caché_L3'] = 1
    else:  # None
        input_data['procesador_nivel_caché_nan'] = 1

    turbo_freq = st.number_input(
        "Max turbo frequency (GHz)",
        min_value=1.9,
        max_value=6.0,
        step=0.1,
        value=5.0  
    )
    input_data["procesador_frecuencia_turbo_máx__GHz"] = float(turbo_freq)

    threads = st.number_input(
        "Number of threads",
        min_value=2,
        max_value=32,
        step=2,
        value=12
    )
    input_data["procesador_número_hilos_ejecución"] = int(threads)

    tdp = st.number_input(
        "Processor TDP (W)",
        min_value=2,
        max_value=280,
        step=5,
        value=65
    )
    input_data["procesador_tdp_W"] = float(tdp)

    cores = st.number_input(
        "Number of cores",
        min_value=1,
        max_value=32,
        step=1,
        value=8
    )
    input_data["procesador_número_núcleos_procesador_cores"] = int(cores)

    base_freq = st.number_input(
        "Base clock frequency (GHz)",
        min_value=0.5,
        max_value=4.3,
        step=0.1,
        value=1.0 
    )
    input_data["procesador_frecuencia_reloj"] = float(base_freq)

    cache = st.number_input(
        "Processor cache (MB)",
        min_value=1,
        max_value=128,
        step=1,
        value=12
    )
    input_data["procesador_caché_MB"] = float(cache)

    proc_freq = st.number_input(
        "Processor frequency (GHz)",
        min_value=0.0011,
        max_value=4.7,
        step=0.1,
        value=2.5  
    )
    input_data["procesador_frecuencia"] = float(proc_freq)
    
    st.subheader("Graphics Card")
    
    graphics_brand_name = {
        0: "2 x AMD FirePro D700",
        1: "2 x nVidia GeForce GTX 980 Ti",
        2: "AMD Radeon",
        3: "AMD Radeon 610M",
        4: "AMD Radeon 660M",
        5: "AMD Radeon 680M",
        6: "AMD Radeon 740M",
        7: "AMD Radeon 760M",
        8: "AMD Radeon 780M",
        9: "AMD Radeon 860M",
        10: "AMD Radeon 880M",
        11: "AMD Radeon 890M",
        12: "AMD Radeon Graphics",
        13: "AMD Radeon R2E",
        14: "AMD Radeon R3",
        15: "AMD Radeon R4",
        16: "AMD Radeon R4 Graphics",
        17: "AMD Radeon R7",
        18: "AMD Radeon RX 480",
        19: "AMD Radeon RX 550",
        20: "AMD Radeon RX 6400",
        21: "AMD Radeon RX 6500 XT",
        22: "AMD Radeon RX 6500M",
        23: "AMD Radeon RX 6600",
        24: "AMD Radeon RX 6600M",
        25: "AMD Radeon RX 6700 XT",
        26: "AMD Radeon RX 6700S",
        27: "AMD Radeon RX 6750 XT",
        28: "AMD Radeon RX 7600",
        29: "AMD Radeon RX 7600S",
        30: "AMD Radeon RX 7700 XT",
        31: "AMD Radeon RX 7800 XT",
        32: "AMD Radeon RX 7900 GRE",
        33: "AMD Radeon RX 7900 XT",
        34: "AMD Radeon RX 7900 XTX",
        35: "AMD Radeon RX Vega",
        36: "AMD Radeon RX Vega 10",
        37: "AMD Radeon RX Vega 11",
        38: "AMD Radeon RX Vega 3",
        39: "AMD Radeon RX Vega 6",
        40: "AMD Radeon RX Vega 7",
        41: "AMD Radeon RX Vega 8",
        42: "AMD Radeon Vega 8 Graphics",
        43: "AMD Radeon Vega 9",
        44: "AMD Uma",
        45: "ARM Mali-G72 MP3",
        46: "Apple M2 GPU",
        47: "Apple M2 Graphics",
        48: "Apple M2 Max GPU",
        49: "Apple M2 Pro GPU",
        50: "Apple M2 Pro Graphics",
        51: "Apple M2 Ultra GPU",
        52: "Apple M3 Graphics",
        53: "Apple M3 Pro Graphics",
        54: "Apple M4 10-Core GPU",
        55: "Apple M4 Graphics",
        56: "Apple M4 Max Graphics",
        57: "Apple M4 Pro 16-Core GPU",
        58: "Apple M4 Pro Graphics",
        59: "Intel Arc A350M",
        60: "Intel Arc A370M",
        61: "Intel Arc A730M",
        62: "Intel Arc A770 Graphics",
        63: "Intel Arc Graphics",
        64: "Intel Arc Graphics 130V",
        65: "Intel Arc Graphics 140V",
        66: "Intel Arc Pro A30M",
        67: "Intel Graphics",
        68: "Intel HD Graphics",
        69: "Intel HD Graphics 400",
        70: "Intel HD Graphics 4000",
        71: "Intel HD Graphics 4400",
        72: "Intel HD Graphics 4600",
        73: "Intel HD Graphics 500",
        74: "Intel HD Graphics 520",
        75: "Intel HD Graphics 530",
        76: "Intel HD Graphics 540",
        77: "Intel HD Graphics 5500",
        78: "Intel HD Graphics 600",
        79: "Intel HD Graphics 605",
        80: "Intel HD Graphics 620",
        81: "Intel HD Graphics 630",
        82: "Intel Iris Graphics",
        83: "Intel Iris Graphics 6100",
        84: "Intel Iris Graphics 650",
        85: "Intel Iris Plus Graphics",
        86: "Intel Iris Plus Graphics 655",
        87: "Intel Iris Xe Graphics",
        88: "Intel UHD Graphics",
        89: "Intel UHD Graphics 1250",
        90: "Intel UHD Graphics 600",
        91: "Intel UHD Graphics 605",
        92: "Intel UHD Graphics 610",
        93: "Intel UHD Graphics 620",
        94: "Intel UHD Graphics 630",
        95: "Intel UHD Graphics 730",
        96: "Intel UHD Graphics 750",
        97: "Intel UHD Graphics 770",
        98: "Intel Xe Graphics",
        99: "Matrox G200",
        100: "Missing_value",
        101: "NVIDIA GeForce GT 1030",
        102: "NVIDIA GeForce GT 710",
        103: "NVIDIA GeForce GT 730",
        104: "NVIDIA GeForce GTX 1050",
        105: "NVIDIA GeForce GTX 1060",
        106: "NVIDIA GeForce GTX 1080",
        107: "NVIDIA GeForce GTX 1630",
        108: "NVIDIA GeForce GTX 1650",
        109: "NVIDIA GeForce GTX 1650 Super",
        110: "NVIDIA GeForce GTX 1660 Super",
        111: "NVIDIA GeForce GTX 1660 Ti",
        112: "NVIDIA GeForce GTX 950M",
        113: "NVIDIA GeForce GTX 970",
        114: "NVIDIA GeForce GTX 980",
        115: "NVIDIA GeForce MX250",
        116: "NVIDIA GeForce MX330",
        117: "NVIDIA GeForce MX350",
        118: "NVIDIA GeForce MX450",
        119: "NVIDIA GeForce MX550",
        120: "NVIDIA GeForce MX570",
        121: "NVIDIA GeForce RTX 2050",
        122: "NVIDIA GeForce RTX 2060",
        123: "NVIDIA GeForce RTX 2070 Super",
        124: "NVIDIA GeForce RTX 2080 Super",
        125: "NVIDIA GeForce RTX 3050",
        126: "NVIDIA GeForce RTX 3050 Ti",
        127: "NVIDIA GeForce RTX 3060",
        128: "NVIDIA GeForce RTX 3060 Ti",
        129: "NVIDIA GeForce RTX 3070",
        130: "NVIDIA GeForce RTX 3070 Ti",
        131: "NVIDIA GeForce RTX 3080",
        132: "NVIDIA GeForce RTX 3080 Ti",
        133: "NVIDIA GeForce RTX 3090",
        134: "NVIDIA GeForce RTX 4050",
        135: "NVIDIA GeForce RTX 4060",
        136: "NVIDIA GeForce RTX 4060 Ti",
        137: "NVIDIA GeForce RTX 4070",
        138: "NVIDIA GeForce RTX 4070 Super",
        139: "NVIDIA GeForce RTX 4070 Ti",
        140: "NVIDIA GeForce RTX 4070 Ti Super",
        141: "NVIDIA GeForce RTX 4080",
        142: "NVIDIA GeForce RTX 4080 Super",
        143: "NVIDIA GeForce RTX 4090",
        144: "NVIDIA Quadro 600",
        145: "NVIDIA Quadro M2000",
        146: "NVIDIA Quadro P1000",
        147: "NVIDIA Quadro P2200",
        148: "NVIDIA Quadro P520",
        149: "NVIDIA Quadro RTX 3000",
        150: "NVIDIA Quadro RTX 4000",
        151: "NVIDIA Quadro RTX 5000",
        152: "NVIDIA Quadro RTX 6000",
        153: "NVIDIA Quadro RTX A5000",
        154: "NVIDIA Quadro T1000",
        155: "NVIDIA Quadro T2000",
        156: "NVIDIA Quadro T400",
        157: "NVIDIA RTX 1000 Ada",
        158: "NVIDIA RTX 2000",
        159: "NVIDIA RTX 2000 Ada",
        160: "NVIDIA RTX 3000 Ada",
        161: "NVIDIA RTX 3500",
        162: "NVIDIA RTX 3500 Ada",
        163: "NVIDIA RTX 4000 Ada",
        164: "NVIDIA RTX 4500 Ada",
        165: "NVIDIA RTX 500 Ada",
        166: "NVIDIA RTX 5000 Ada",
        167: "NVIDIA RTX A1000",
        168: "NVIDIA RTX A2000",
        169: "NVIDIA RTX A3000",
        170: "NVIDIA RTX A400",
        171: "NVIDIA RTX A4000",
        172: "NVIDIA RTX A4500",
        173: "NVIDIA RTX A500",
        174: "NVIDIA RTX A5000",
        175: "NVIDIA RTX A5500",
        176: "NVIDIA RTX A6000",
        177: "NVIDIA T1000",
        178: "NVIDIA T1200",
        179: "NVIDIA T400",
        180: "NVIDIA T550",
        181: "NVIDIA T600",
        182: "PowerVR SGX6110",
        183: "Qualcomm Adreno",
        184: "Qualcomm Adreno 540 GPU",
        185: "Qualcomm Adreno 618",
        186: "Qualcomm Adreno 680",
        187: "Qualcomm Adreno 690",
        188: "Qualcomm Adreno X Elite",
        189: "Qualcomm Adreno X Plus",
        190: "VIA Chrome9",
        191: "nVidia NextGen Ion",
        192: "nVidia Quadro M4000",
        193: "sin tarjeta gráfica"
    }
    
    grafics_to_label = {v: k for k, v in graphics_brand_name.items()}
    selected_grafics_card = st.selectbox("Select Graphics Card", options=sorted(grafics_to_label.keys()))
    input_data["gráfica_tarjeta_gráfica_label"] = grafics_to_label[selected_grafics_card]
    
    st.subheader("Video Graphics Output")
    
    grafica_features = [
    'gráfica_salida_vídeo_DVI', 'gráfica_salida_vídeo_DisplayPort', 'gráfica_salida_vídeo_HDMI', 'gráfica_salida_vídeo_HDMI 1.4', 'gráfica_salida_vídeo_HDMI 2.0', 'gráfica_salida_vídeo_HDMI 2.1', 'gráfica_salida_vídeo_Micro HDMI', 'gráfica_salida_vídeo_Mini DisplayPort', 'gráfica_salida_vídeo_Mini HDMI', 'gráfica_salida_vídeo_Thunderbolt 3', 'gráfica_salida_vídeo_Thunderbolt 4', 'gráfica_salida_vídeo_USB-C', 'gráfica_salida_vídeo_VGA', 'gráfica_salida_vídeo_nan'
    ]

    grafica_names = [f.replace("gráfica_salida_vídeo_", "") for f in grafica_features]
    selected_grafica = st.selectbox("Select type of graphics output", options=grafica_names)
    for g in grafica_features:
        input_data[g] = 0
    input_data[f"gráfica_salida_vídeo_{selected_grafica}"] = 1
    
    st.subheader("Battery")
    battery_capacity = st.number_input(
        "Battery Capacity (Wh)", 
        min_value=10.0, 
        max_value=150.0, 
        value=50.0, 
        step=1.0, 
        format="%.1f"
    )
    input_data["alimentación_vatios_hora_Wh"] = float(battery_capacity)

    battery_life = st.number_input(
        "Battery Life (hours)", 
        min_value=1.0, 
        max_value=24.0, 
        value=8.0, 
        step=0.5, 
        format="%.1f"
    )
    input_data["alimentación_autonomía_batería_h"] = float(battery_life)

    st.subheader("Dimensions")
    
    height = st.number_input(
        "Height (mm)",
        min_value=0,
        max_value=560,
        value=101,
        step=1
    )
    input_data["altura_mm"] = int(height)

    depth = st.number_input(
        "Depth (cm)",
        min_value=3.0,
        max_value=55.0,
        value=24.7,
        step=0.1,
        format="%.1f"
    )
    input_data["medidas_profundidad_cm"] = float(depth)

    weight = st.number_input(
        "Weight (kg)",
        min_value=0.2,
        max_value=24.0,
        value=2.45,
        step=0.1,
        format="%.2f"
    )
    input_data["medidas_peso_kg"] = float(weight)

    width = st.number_input(
        "Width (cm)",
        min_value=3.0,
        max_value=92.0,
        value=27.3,
        step=0.1,
        format="%.1f"
    )
    input_data["medidas_ancho_cm"] = float(width)

    fecha_lanzamiento = st.number_input(
        "Release Year",
        min_value=2013,
        max_value=2025,
        value=2023,
        step=1,
        format="%d"
    )
    input_data["otras_características_fecha_lanzamiento"] = int(fecha_lanzamiento)
    
    
    input_data["ofertas_count"] = 3
    
    st.subheader("Equipment Features")
    equip_features = [
    'equip_Force Touch Trackpad', 'equip_ScreenPad', 'equip_Touch Bar', 'equip_Touch ID',
    'equip_Touchpad multitáctil', 'equip_TrackPoint / TouchStick / Pointing Stick', 'equip_USB-C',
    'equip_altavoces estéreo', 'equip_altavoces estéreo JBL', 'equip_altavoz integrado',
    'equip_con iluminación', 'equip_conector de seguridad Kensington', 'equip_disco duro SSD',
    'equip_lector de tarjetas', 'equip_lector de tarjetas inteligentes', 'equip_micrófono integrado',
    'equip_refrigeración líquida', 'equip_webcam'
    ]

    equip_feature_display_names = {
        'equip_Force Touch Trackpad': 'Force Touch Trackpad',
        'equip_ScreenPad': 'ScreenPad',
        'equip_Touch Bar': 'Touch Bar',
        'equip_Touch ID': 'Touch ID',
        'equip_Touchpad multitáctil': 'Multitouch Touchpad',
        'equip_TrackPoint / TouchStick / Pointing Stick': 'TrackPoint / TouchStick / Pointing Stick',
        'equip_USB-C': 'USB-C',
        'equip_altavoces estéreo': 'Stereo Speakers',
        'equip_altavoces estéreo JBL': 'JBL Stereo Speakers',
        'equip_altavoz integrado': 'Built-in Speaker',
        'equip_con iluminación': 'Backlit Keyboard',
        'equip_conector de seguridad Kensington': 'Kensington Security Slot',
        'equip_disco duro SSD': 'SSD Hard Drive',
        'equip_lector de tarjetas': 'Card Reader',
        'equip_lector de tarjetas inteligentes': 'Smart Card Reader',
        'equip_micrófono integrado': 'Built-in Microphone',
        'equip_refrigeración líquida': 'Liquid Cooling',
        'equip_webcam': 'Webcam'
        }

    cols = st.columns(3)

    for i, feature in enumerate(equip_features):
        label = equip_feature_display_names.get(feature, feature)
        input_data[feature] = int(cols[i % 3].checkbox(label))

    num_altavoces = st.selectbox(
        "Número de altavoces",
        options=[2, 4, 6, 8],
        index=0
    )
    input_data["sonido_número_altavoces"] = int(num_altavoces)

    st.subheader("Connectivity Flags")
    
    connectivity_flags = [
        'Bluetooth', 'Ethernet', 'LAN', 'NFC', 'infrarrojos', 'wifi', 'wifi Direct'
    ]

    connectivity_display_names = {
        'Bluetooth': 'Bluetooth',
        'Ethernet': 'Ethernet',
        'LAN': 'LAN (Wired Network)',
        'NFC': 'NFC (Near Field Communication)',
        'infrarrojos': 'Infrared',
        'wifi': 'WiFi',
        'wifi Direct': 'WiFi Direct'
    }

    conn_cols = st.columns(2)

    for i, conn in enumerate(connectivity_flags):
        label = connectivity_display_names.get(conn, conn)
        input_data[conn] = int(conn_cols[i % 2].checkbox(label))
    
    st.subheader("Optical Reader")
    almacenamiento_features = [
    'almacenamiento_lector_óptico_grabadora DVD', 'almacenamiento_lector_óptico_lector DVD', 'almacenamiento_lector_óptico_ninguno', 'almacenamiento_lector_óptico_nan'
    ]  
    
    almacenamiento_names = [f.replace("almacenamiento_lector_óptico_", "") for f in almacenamiento_features]
    selected_almacenamiento = st.selectbox("Select type of optical reader", options=almacenamiento_names)
    for a in almacenamiento_features:
        input_data[a] = 0
    input_data[f"almacenamiento_lector_óptico_{selected_almacenamiento}"] = 1

if st.button("Predict 💰"):
    model_input = pd.DataFrame([input_data])[feature_names]
    
    missing_features = set(feature_names) - set(model_input.columns)
    if missing_features:
        st.error(f"Missing features in input data: {missing_features}")
    else:
        log_prediction = model.predict(model_input)
        prediction = np.expm1(log_prediction)
        
        st.session_state.last_prediction = prediction[0]
        
        st.success(f"Estimated Price: €{prediction[0]:,.2f}")
        
        st.session_state.show_feedback = True
        st.session_state.feedback_submitted = False

def handle_submit():
    if st.session_state.accuracy_select == "Select an option":
        st.session_state.feedback_error = True
    else:
        save_feedback(
            st.session_state.last_prediction,
            st.session_state.accuracy_select,
            st.session_state.comment_text
        )
        st.session_state.feedback_error = False
        st.session_state.feedback_submitted = True

if st.session_state.show_feedback:
    st.write("---")
    st.header("Feedback")
    
    with st.container():
        st.text("Please help us improve by providing feedback on this prediction:")
        
        if st.session_state.feedback_submitted:
            st.success("Thank you for your feedback! It will help us improve our predictions.")
        else:

            st.write("How accurate was this price prediction?")
            accuracy_options = ["Select an option", "Very Inaccurate", "Somewhat Inaccurate", "Neutral", "Somewhat Accurate", "Very Accurate"]
            

            st.selectbox(
                "Accuracy", 
                options=accuracy_options,
                key="accuracy_select",
                label_visibility="collapsed"
            )
            
            if st.session_state.get('feedback_error', False):
                st.error("Please select an accuracy rating.")
            
            st.write("Additional comments (optional):")
            st.text_area(
                "Comments",
                key="comment_text",
                label_visibility="collapsed", 
                height=150
            )
            
            st.button("Submit Feedback", on_click=handle_submit)
