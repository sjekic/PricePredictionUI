import streamlit as st
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import warnings
import datetime
import json
import os
warnings.filterwarnings('ignore')

def save_feedback(recommendations, accuracy_rating, feedback_text=None):
    """
    Save user feedback to a flat file on disk.
    
    Parameters:
    - recommendations: The laptop recommendations that were shown
    - accuracy_rating: Rating of recommendation accuracy
    - feedback_text: Optional text feedback
    """
    timestamp = datetime.datetime.now().isoformat()
    
    # Capture input data for potential model retraining
    feedback_data = {
        "timestamp": timestamp,
        "recommendations": recommendations,
        "accuracy_rating": accuracy_rating,
        "comments": feedback_text,
        "session_id": st.session_state.get("session_id", "unknown"),
        "input_data": {k: (float(v) if isinstance(v, (int, float, np.number)) else v) 
                     for k, v in st.session_state.last_input_data.items()}
    }
    
    os.makedirs("feedback", exist_ok=True)
    

    feedback_file = "feedback/knn_recommendation_feedback.jsonl"
    with open(feedback_file, "a") as f:
        f.write(json.dumps(feedback_data) + "\n")
    
    return True

if 'session_id' not in st.session_state:
    st.session_state.session_id = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
if 'recommendations' not in st.session_state:
    st.session_state.recommendations = None
if 'last_input_data' not in st.session_state:
    st.session_state.last_input_data = {}
if 'feedback_submitted' not in st.session_state:
    st.session_state.feedback_submitted = False

st.set_page_config(
    page_title="Laptop Price Predictor",
    page_icon="💻",
    layout="wide"
)


st.markdown("""
<style>
    .stSelectbox, .stSlider, .stMultiselect {
        padding-bottom: 20px;
    }
    .section-header {
        font-size: 26px;
        font-weight: bold;
        margin-top: 30px;
        margin-bottom: 20px;
    }
    .section-subheader {
        font-size: 20px;
        font-weight: bold;
        margin-top: 20px;
        margin-bottom: 10px;
    }
    .section-description {
        margin-bottom: 20px;
        color: #4e4e4e;
    }
    .prediction-price {
        font-size: 40px;
        font-weight: bold;
        color: #0066cc;
        text-align: center;
        padding: 20px;
        margin: 20px 0;
        background-color: #f0f7ff;
        border-radius: 10px;
    }
    .recommendation-title {
        font-weight: bold;
        font-size: 18px;
    }
    .recommendation-price {
        font-weight: bold;
        color: #0066cc;
    }
</style>
""", unsafe_allow_html=True)

st.title("Laptop Price Predictor & Recommendation System")
st.write("""
### How to Use the Laptop Price Predictor

1. **Select Specifications**: Use the dropdown menus to set your desired laptop specifications.
    - Set device type and brand
    - Select RAM, processor, and storage options
    - Choose screen size and graphics options
    - Pick operating system and color

2. **Get Predictions**: Click the "Predict Price & Find Similar Laptops" button to see:
    - Estimated price for your configuration
    - Similar laptops from our database

3. **Refine Your Search**: Adjust specifications and click the button again to see updated results.

### How It Works

This app uses a K-Nearest Neighbors algorithm to find laptops with similar specifications to what you've chosen.
The price prediction is based on these similar laptops, weighted by their similarity scores.
""")


@st.cache_data
def load_data():
    try:
        X_train = pd.read_csv("X_train_final.csv")
        y_train = pd.read_csv("y_train.csv")
        
        if len(y_train.columns) > 0 and (y_train.columns[0] == '' or y_train.columns[0] == 'Unnamed: 0'):
            y_train = y_train.drop(y_train.columns[0], axis=1)
            
        titulos = pd.read_csv("titulos.csv")
        
        return X_train, y_train, titulos
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None

X_train, y_train, titulos = load_data()

if X_train is not None and y_train is not None and titulos is not None:

    @st.cache_resource
    def build_recommendation_model():
        k = 5
        
        X_knn = X_train.copy()
        
        # Add price column
        if isinstance(y_train, pd.DataFrame):
            if 'price_avg' in y_train.columns:
                X_knn['precio'] = y_train['price_avg'].values
            else:
                X_knn['precio'] = y_train.iloc[:, 0].values
        else:
            X_knn['precio'] = y_train
        
        columns_to_drop = [
            'medidas_profundidad_cm', 'medidas_peso_kg', 'procesador_tdp_W',
            'medidas_ancho_cm', 'ofertas_count', 'pantalla_diagonal_pantalla_cm',
            'altura_mm'
        ]
        X_knn = X_knn.drop(columns=[col for col in columns_to_drop if col in X_knn.columns], errors='ignore')
        
        numeric_features = []
        categorical_features = []
        
        for col in X_knn.columns:
            if col == 'precio':
                continue
            if pd.api.types.is_numeric_dtype(X_knn[col]):
                numeric_features.append(col)
            else:
                categorical_features.append(col)

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', Pipeline([
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ]), numeric_features),
                ('cat', Pipeline([
                    ('imputer', SimpleImputer(strategy='most_frequent')), 
                    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
                ]), categorical_features)
            ],
            remainder='passthrough'
        )
        
        precio_column = X_knn['precio'].copy()
        X_knn_without_precio = X_knn.drop('precio', axis=1)
        
        X_knn_transformed = preprocessor.fit_transform(X_knn_without_precio)
        
        knn_model = NearestNeighbors(n_neighbors=k, metric='cosine')
        knn_model.fit(X_knn_transformed)
        
        return {
            'knn_model': knn_model,
            'preprocessor': preprocessor,
            'precio_column': precio_column,
            'X_knn_columns': X_knn_without_precio.columns,
            'numeric_features': numeric_features,
            'categorical_features': categorical_features
        }
    
    with st.spinner("Building recommendation model..."):
        model_data = build_recommendation_model()
    
    input_data = {}
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown("<div class='section-header'>Device Type</div>", unsafe_allow_html=True)
        
        st.write("Select type of device")
        form_factors = ["Desktop", "Laptop"]
        selected_form = st.selectbox("", options=form_factors, key="form_factor")
        
        for form in form_factors:
            col_name = f"tipo_{form}"
            input_data[col_name] = 1 if form == selected_form else 0
        

        st.write("Select product type")
        product_types = ["Barebone", "Chromebook", "Kit ampliación PC", "Mini PC", "Netbook", 
                        "PC completo", "PC de oficina", "PC gaming", "PC multimedia", 
                        "Portátil 3D", "Portátil convertible", "Portátil gaming", 
                        "Portátil multimedia", "Portátil profesional", "Thin Client", 
                        "Ultrabook", "Workstation"]
        
        selected_product = st.selectbox("", options=product_types, key="product_type")
        
        for product_type in product_types:
            col_name = f"tipo_producto_{product_type}"
            input_data[col_name] = 1 if product_type == selected_product else 0
        

        st.markdown("<div class='section-header'>Brand</div>", unsafe_allow_html=True)
        st.write("Select Brand")
        brands = ["Acer", "Apple", "ASUS", "Dell", "HP", "Lenovo", "MSI", "Samsung", "Toshiba", "Other"]
        selected_brand = st.selectbox("", options=brands, key="brand")
        input_data['company_name'] = selected_brand
        
        # ----- RAM -----
        st.markdown("<div class='section-header'>RAM</div>", unsafe_allow_html=True)
        
        # RAM Type
        st.write("Select type of RAM")
        ram_types = ["DDR3", "DDR3L", "DDR4", "DDR4L", "DDR5", "LPDDR3", "LPDDR4", "LPDDR4X", "LPDDR5", "LPDDR5X"]
        selected_ram_type = st.selectbox("", options=ram_types, key="ram_type")
        
        for ram_type in ram_types:
            col_name = f"ram_tipo_ram_{ram_type}"
            input_data[col_name] = 1 if ram_type == selected_ram_type else 0
        
        # RAM Capacity
        st.write("RAM capacity (GB)")
        ram_options = [2, 4, 8, 16, 32, 64, 128]
        ram_memory = st.selectbox("", options=ram_options, index=3, key="ram_capacity")
        input_data['ram_memoria_ram_GB'] = ram_memory
        
        # RAM Frequency
        st.write("RAM Frequency (MHz)")
        ram_freq_options = [1600, 2133, 2400, 2666, 3000, 3200, 3600, 4000, 4800, 5200]
        ram_frequency = st.selectbox("", options=ram_freq_options, index=5, key="ram_frequency")
        input_data['ram_frecuencia_memoria_MHz'] = ram_frequency
        
        # ----- PROCESSOR -----
        st.markdown("<div class='section-header'>Processor</div>", unsafe_allow_html=True)
        
        # Processor Brand
        st.write("Select Processor Brand")
        processor_brands = ["Intel", "AMD", "Apple", "Qualcomm", "MediaTek", "Other"]
        selected_processor_brand = st.selectbox("", options=processor_brands, key="processor_brand")
        
        # Processor cores
        st.write("Processor Cores")
        processor_cores_options = [2, 4, 6, 8, 10, 12, 16, 24, 32]
        processor_cores = st.selectbox("", options=processor_cores_options, index=2, key="processor_cores")
        input_data['procesador_número_núcleos_procesador_cores'] = processor_cores
        
        # Processor threads
        st.write("Processor Threads")
        processor_threads_options = [2, 4, 8, 12, 16, 24, 32, 64]
        processor_threads = st.selectbox("", options=processor_threads_options, index=2, key="processor_threads")
        input_data['procesador_número_hilos_ejecución'] = processor_threads
        
        # Processor frequency
        st.write("Base Frequency (GHz)")
        processor_frequency_options = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
        processor_frequency = st.selectbox("", options=processor_frequency_options, index=3, key="processor_frequency")
        input_data['procesador_frecuencia_reloj'] = processor_frequency
        
        # Processor turbo frequency
        st.write("Turbo Frequency (GHz)")
        processor_turbo_options = [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0]
        processor_turbo = st.selectbox("", options=processor_turbo_options, index=3, key="processor_turbo")
        input_data['procesador_frecuencia_turbo_máx__GHz'] = processor_turbo
        
        # ----- STORAGE -----
        st.markdown("<div class='section-header'>Storage</div>", unsafe_allow_html=True)
        
        # Storage type
        st.write("Select Storage Type")
        storage_types = ["PCIe SSD", "SATA", "SSD", "disco duro HDD", "disco duro M.2 SSD", 
                        "disco duro SSD", "disco híbrido (HHD)", "memoria flash", "sin disco duro"]
        selected_storage = st.selectbox("", options=storage_types, key="storage_type")
        
        # Set the one-hot encoded storage type
        for storage_type in storage_types:
            col_name = f"disco_duro_tipo_disco_duro_{storage_type}"
            input_data[col_name] = 1 if storage_type == selected_storage else 0
        
        # Storage capacity
        st.write("Storage Capacity (GB)")
        storage_capacity_options = [128, 256, 512, 1024, 2048, 4096]
        storage_capacity = st.selectbox("", options=storage_capacity_options, index=2, key="storage_capacity")
        input_data['disco_duro_capacidad_memoria_ssd_GB'] = storage_capacity
        
        # Number of disks
        st.write("Number of Disks")
        num_disks = st.selectbox("", options=[1, 2, 3, 4], key="num_disks")
        input_data['disco_duro_número_discos_duros_instalados'] = num_disks
    
    with col2:
        # ----- DISPLAY -----
        st.markdown("<div class='section-header'>Display</div>", unsafe_allow_html=True)
        
        # Screen size
        st.write("Screen Size (inches)")
        screen_size_options = [10.1, 11.6, 12.5, 13.3, 14.0, 15.6, 16.0, 17.3, 18.4]
        screen_size = st.selectbox("", options=screen_size_options, index=5, key="screen_size")
        input_data['pantalla_tamaño_pantalla_pulgadas'] = screen_size
        
        # Screen brightness
        st.write("Screen Brightness (cd/m²)")
        screen_brightness_options = [200, 250, 300, 350, 400, 450, 500, 600, 800]
        screen_brightness = st.selectbox("", options=screen_brightness_options, index=2, key="screen_brightness")
        input_data['pantalla_luminosidad_cd_m2'] = screen_brightness
        
        # ----- GRAPHICS -----
        st.markdown("<div class='section-header'>Graphics</div>", unsafe_allow_html=True)
        
        # Graphics Brand
        st.write("Select Graphics Brand")
        graphics_brands = ["NVIDIA", "AMD", "Intel", "Apple", "Integrated", "Other"]
        selected_graphics_brand = st.selectbox("", options=graphics_brands, key="graphics_brand")
        
        # Graphics memory
        st.write("Graphics Memory (GB)")
        graphics_memory_options = [0, 1, 2, 4, 6, 8, 12, 16, 24]
        graphics_memory = st.selectbox("", options=graphics_memory_options, index=3, key="graphics_memory")
        input_data['gráfica_memoria_gráfica'] = graphics_memory
        
        # ----- OPERATING SYSTEM -----
        st.markdown("<div class='section-header'>Operating System</div>", unsafe_allow_html=True)
        
        # OS selection
        st.write("Select Operating System")
        os_types = ["DOS", "No OS", "Other OS", "Windows", "macOS"]
        selected_os = st.selectbox("", options=os_types, index=3, key="os_type")
        
        # Set the one-hot encoded OS type
        for os_type in os_types:
            col_name = f"os_{os_type}"
            input_data[col_name] = 1 if os_type == selected_os else 0
        
        # ----- COLOR -----
        st.markdown("<div class='section-header'>Color</div>", unsafe_allow_html=True)
        st.write("Select Color")
        colors = ["azul", "blanco", "bronce", "dorado", "gris", "negro", "plateado", "rojo", "rosa", "verde"]
        selected_color = st.selectbox("", options=colors, index=5, key="color")
        
        # Set the one-hot encoded color
        for color in colors:
            col_name = f"color_{color}"
            input_data[col_name] = 1 if color == selected_color else 0
        
        # ----- BATTERY -----
        st.markdown("<div class='section-header'>Battery</div>", unsafe_allow_html=True)
        
        # Battery life
        st.write("Battery Life (hours)")
        battery_life_options = [2, 4, 6, 8, 10, 12, 15, 18, 24]
        battery_life = st.selectbox("", options=battery_life_options, index=3, key="battery_life")
        input_data['alimentación_autonomía_batería_h'] = battery_life
        
        # Battery capacity
        st.write("Battery Capacity (Wh)")
        battery_capacity_options = [30, 40, 50, 60, 70, 80, 90, 100]
        battery_capacity = st.selectbox("", options=battery_capacity_options, index=2, key="battery_capacity")
        input_data['alimentación_vatios_hora_Wh'] = battery_capacity
        
        # ----- CONNECTIVITY -----
        st.markdown("<div class='section-header'>Connectivity</div>", unsafe_allow_html=True)
        
        # Connectivity options
        st.write("Select Connectivity Options")
        connectivity_options = ["Bluetooth", "Ethernet", "LAN", "NFC", "infrarrojos", "wifi", "wifi Direct"]
        selected_connectivity = st.multiselect(
            "",
            options=connectivity_options,
            default=["Bluetooth", "wifi"],
            key="connectivity"
        )
        
        # Set the connectivity options
        for option in connectivity_options:
            input_data[option] = 1 if option in selected_connectivity else 0
    
    # Set defaults for other required fields if not already set
    
    # Add processor cache
    if 'procesador_caché_MB' not in input_data:
        input_data['procesador_caché_MB'] = 8
    
    # Add missing equipment features
    equipment_features = [
        "Force Touch Trackpad", "ScreenPad", "Touch Bar", "Touch ID", 
        "Touchpad multitáctil", "TrackPoint / TouchStick / Pointing Stick", 
        "USB-C", "altavoces estéreo", "altavoces estéreo JBL", 
        "altavoz integrado", "con iluminación", "conector de seguridad Kensington", 
        "disco duro SSD", "lector de tarjetas", "lector de tarjetas inteligentes", 
        "micrófono integrado", "refrigeración líquida", "webcam"
    ]
    
    # Set default equipment features
    for feature in equipment_features:
        col_name = f"equip_{feature}"
        if col_name not in input_data:
            input_data[col_name] = 1 if feature in ["USB-C", "webcam", "altavoces estéreo", "micrófono integrado"] else 0
    
    # Add graphics outputs
    graphics_outputs = ["DVI", "DisplayPort", "HDMI", "HDMI 1.4", "HDMI 2.0", "HDMI 2.1", 
                        "Micro HDMI", "Mini DisplayPort", "Mini HDMI", "Thunderbolt 3",
                        "Thunderbolt 4", "USB-C", "VGA"]
    
    for output in graphics_outputs:
        col_name = f"gráfica_salida_vídeo_{output}"
        if col_name not in input_data:
            input_data[col_name] = 1 if output == "HDMI" else 0
    
def predict_and_recommend(user_input):
        try:
            # Create DataFrame from user input
            user_df = pd.DataFrame([user_input])
            
            # Make sure all columns match the training data
            for col in model_data['X_knn_columns']:
                if col not in user_df.columns:
                    user_df[col] = np.nan
            
            # Keep only the columns used in training
            user_df = user_df[model_data['X_knn_columns']]
            
            # Apply preprocessing
            user_transformed = model_data['preprocessor'].transform(user_df)
            
            # Use KNN to find similar laptops
            distances, indices = model_data['knn_model'].kneighbors(user_transformed, n_neighbors=5)
            
            # Calculate mean distance for scaling
            mean_dist = np.mean(distances)
            
            # Prepare results
            results = []
            for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
                title = titulos.iloc[idx]['título'] if 'título' in titulos.columns else f"Laptop {idx}"
                price = model_data['precio_column'].iloc[idx]
                
                # Better similarity calculation that won't approach zero too quickly
                similarity = np.exp(-distance/max(mean_dist, 1.0))
                
                results.append({
                    'title': title,
                    'price': float(price),
                    'similarity': similarity
                })
            
            return {
                'recommendations': results
            }
        except Exception as e:
            st.error(f"Error in prediction: {e}")
            return None
        
    # Define callback function for the feedback form submission
def handle_submit():
        if st.session_state.accuracy_select == "Select an option":
            st.session_state.feedback_error = True
        else:
            # Save the feedback
            save_feedback(
                st.session_state.recommendations,
                st.session_state.accuracy_select,
                st.session_state.comment_text
            )
            # Update state to show success message
            st.session_state.feedback_error = False
            st.session_state.feedback_submitted = True
        
    # Button to predict
predict_button = st.button("Find Similar Laptops", type="primary", use_container_width=True)
    
    # When button is clicked
if predict_button:
        # Store the current input data
        st.session_state.last_input_data = input_data.copy()
        
        with st.spinner("Finding similar laptops..."):
            results = predict_and_recommend(input_data)
            
            if results:
                # Store recommendations in session state for feedback
                st.session_state.recommendations = results['recommendations']
                st.session_state.feedback_submitted = False
                
                # Display similar laptops
                st.markdown("<div class='section-header'>Similar Laptops</div>", unsafe_allow_html=True)
                
                # Create three columns for recommendations
                cols = st.columns(3)
                
                # Display each recommendation in a column
                for i, rec in enumerate(results['recommendations']):
                    col_idx = i % 3
                    with cols[col_idx]:
                        st.markdown(f"<div class='recommendation-title'>{rec['title']}</div>", unsafe_allow_html=True)
                        st.markdown(f"<div class='recommendation-price'>€{rec['price']:.2f}</div>", unsafe_allow_html=True)
                        st.write(f"Similarity: {rec['similarity']:.2f}")
                        st.divider()
    
    # Always show feedback if we have recommendations
if st.session_state.recommendations:
        # Add feedback section
        st.markdown("<div class='section-header'>Feedback</div>", unsafe_allow_html=True)
        
        # Create a yellow background container
        with st.container():
            st.markdown(
                """
                <div class="feedback-container">
                <p>Please help us improve by providing feedback on these recommendations:</p>
                </div>
                """, 
                unsafe_allow_html=True
            )
            
            # Show success message if feedback was submitted
            if st.session_state.get('feedback_submitted', False):
                st.success("Thank you for your feedback! It will help us improve our recommendations.")
            else:
                # Accuracy rating - simplified approach using session state properly
                st.write("How accurate were these laptop recommendations?")
                accuracy_options = ["Select an option", "Very Inaccurate", "Somewhat Inaccurate", "Neutral", "Somewhat Accurate", "Very Accurate"]
                
                # Use a key for the widget that's also in session state
                st.selectbox(
                    "Accuracy", 
                    options=accuracy_options,
                    key="accuracy_select",
                    label_visibility="collapsed"
                )
                
                # Show error if they tried to submit without selecting
                if st.session_state.get('feedback_error', False):
                    st.error("Please select an accuracy rating.")
                
                # Text area for additional comments
                st.write("Additional comments (optional):")
                st.text_area(
                    "Comments",
                    key="comment_text",
                    label_visibility="collapsed", 
                    height=150
                )
                
                # Submit button using the callback
                st.button("Submit Feedback", on_click=handle_submit)