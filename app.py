import json
import time
import streamlit as st
import cv2
import numpy as np
import pandas as pd

from PIL import Image
from aiConsult import gemini_pro_vision_api

hide_streamlit_style = """
<style>
.stAppHeader {
    visibility: hidden;
    height: 0px
}
[data-testid="stElementToolbar"] {
                    display: none;
}
div.block-container {
    padding-top:0rem;
}
table {
    width: 100%;
}
.row_heading {
    width: 30%;
}
.data {
    width: 70%;
}
</style>
"""
st.write(hide_streamlit_style, unsafe_allow_html=True)
st.subheader("Escaner de placas vehiculares", divider="blue")

# Inicializar el estado de la sesión para la imagen
if 'image' not in st.session_state:
    st.session_state.image = None
if 'placa' not in st.session_state:
    st.session_state.placa = None

# Función para capturar una imagen desde la cámara
def capture_image():
    try:
        # Usar st.camera_input para obtener la imagen desde la cámara del usuario
        picture = st.camera_input(label="Capturar imagen", key="captura")
        

        if picture:
            # Convertir la imagen a un formato utilizable por OpenCV
            bytes_data = picture.getvalue()
            cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

            # Guardar la imagen en el estado de la sesión
            st.session_state.image = cv2_img
            return cv2_img
        else:
            return None
    except Exception as e:
        st.error(f"Error al capturar la imagen: {e}")
        return None

# Contenedor para mostrar las placas detectadas
datos = st.empty()
response = None

# Bucle principal para la captura continua
while True:
    captured_image = capture_image()

    if captured_image is not None:
        # PIL image
        pil_image = Image.fromarray(captured_image)
        # Convertir la imagen de BGR a RGB para mostrarla correctamente en Streamlit
        captured_image = cv2.cvtColor(captured_image, cv2.COLOR_BGR2RGB)

        prompt ="""## Instrucciones
1. **Recibir imagen:** El modelo recibirá una imagen de un vehículo.
2. **Analizar imagen:** El modelo analizará la imagen para identificar la placa del vehículo, la marca y el modelo.
3. **Generar JSON:** El modelo generará un objeto JSON con la siguiente estructura:
```json
{
"placa": "XXXXXXX",
"marca": "XXXXXXX",
"modelo": "XXXXXXX"
}
Valores por defecto:
Si no se encuentra una placa en la imagen, el valor de "placa" debe ser "-".
Si no se puede identificar la marca del vehículo, el valor de "marca" debe ser "-".
Si no se puede identificar el modelo del vehículo, el valor de "modelo" debe ser "-".
"""
        # verificar si la imagen ya ha sido procesada
        if not st.session_state.placa:
            datos.write("Procesando imagen...")

            # Call the Gemini Pro Vision API
            gemini_response = gemini_pro_vision_api(pil_image, prompt)
            st.session_state.placa = json.loads(gemini_response)

        # verificar si es una lista
        placa = st.session_state.placa
        if isinstance(placa, list):
            datos.write("Verifica que solo hay un vehiculo en la imagen")
        elif isinstance(placa, dict):
            if placa.get("placa") == "-" \
                and placa.get("marca") == "-" \
                and placa.get("modelo") == "-":
                datos.write("No se ha encontrado placas de vehículos en la imagen")
            else:
                # datos.table(placa)

                df = pd.DataFrame.from_dict(placa, orient='index')
                datos.markdown(df.style.set_properties(**{'width': '100%'}).hide(axis=1).to_html(), unsafe_allow_html = True)

                # datos.dataframe(placa,use_container_width=True,column_config={
                #     "_index": "Vehículo",
                #     "value": "Datos"
                # })

        time.sleep(5) # Esperar 5 segundos antes de la siguiente captura
        st.rerun() #Forzar la actualización para la siguiente captura

    else:
        # Mostrar un mensaje mientras no se captura una imagen.
        datos.write("Esperando captura de imagen...")
        st.session_state.placa = None

        time.sleep(5) #Para no saturar el CPU
        st.rerun() #Forzar la actualización para la siguiente captura