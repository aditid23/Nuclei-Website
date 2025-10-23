import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.models import load_model
from io import BytesIO
import zipfile
import time

# ===============================================================
# Page config & header
# ===============================================================
st.set_page_config(page_title="Cell Nuclei Segmentation", layout="wide")
st.markdown("<h1 style='text-align:center; font-size:32px; font-weight:bold;'> Cell Nuclei Segmentation using TransUNet</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; font-size:18px;'>Upload one or more microscopy images to visualize the predicted nuclei masks.</p>", unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

# ===============================================================
# 0) Re-declare custom layers / blocks used by your saved model
#    (Make sure these match the classes used when the model was created/saved)
# ===============================================================
class PatchEmbedding(layers.Layer):
    def __init__(self, embed_dim=768, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.proj = layers.Conv2D(embed_dim, kernel_size=1, strides=1, padding="valid")
        self.flatten = layers.Reshape((-1, embed_dim))

    def call(self, x):
        x = self.proj(x)
        x = self.flatten(x)
        return x

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"embed_dim": self.embed_dim})
        return cfg

class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim=768, num_heads=12, mlp_dim=3072, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.dropout = dropout

        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.attn = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim, dropout=dropout)
        self.drop1 = layers.Dropout(dropout)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)

        self.mlp = models.Sequential([
            layers.Dense(mlp_dim, activation=tf.nn.gelu),
            layers.Dropout(dropout),
            layers.Dense(embed_dim),
            layers.Dropout(dropout),
        ])

    def call(self, x):
        attn_output = self.attn(x, x)
        x = x + self.drop1(attn_output)
        mlp_output = self.mlp(self.norm2(x))
        return x + mlp_output

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"embed_dim": self.embed_dim, "num_heads": self.num_heads, "mlp_dim": self.mlp_dim, "dropout": self.dropout})
        return cfg

def build_transformer_encoder(x, num_layers=4, embed_dim=768, num_heads=12, mlp_dim=3072):
    # This helper is used only during building a model programmatically; we keep it for completeness.
    for _ in range(num_layers):
        x = TransformerBlock(embed_dim=embed_dim, num_heads=num_heads, mlp_dim=mlp_dim)(x)
    return x

# ===============================================================
# 1) Model-loading helper: pass custom objects mapping
# ===============================================================
MODEL_PATH = "transunet_dsb.h5"  # update to your model filename if different

@st.cache_resource
def load_model_with_custom_objects(path=MODEL_PATH):
    # Provide a mapping of custom class names to the classes defined above
    custom_objects = {
        "PatchEmbedding": PatchEmbedding,
        "TransformerBlock": TransformerBlock,
        # If your model referenced build_transformer_encoder or other callables by name, include them too:
        "build_transformer_encoder": build_transformer_encoder,
    }
    # Use load_model with custom_objects. compile=False to speed up if not needed.
    model = load_model(path, custom_objects=custom_objects, compile=False)
    return model

# Attempt to load model and show status
try:
    model = load_model_with_custom_objects()
    st.success("✅ Model loaded successfully!")
except Exception as e:
    st.error(f"⚠️ Error loading model: {e}")
    st.stop()  # stop app if model can't be loaded

# ===============================================================
# 2) Prediction utilities
# ===============================================================
def preprocess_for_model(img_bgr, target_size=(224, 224)):
    # model expects RGB input normalized; adjust size to whatever model uses
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, target_size)
    img_norm = img_resized / 255.0
    return img_norm

def predict_mask(model, img_bgr, target_size=(224,224)):
    img_norm = preprocess_for_model(img_bgr, target_size)
    inp = np.expand_dims(img_norm, axis=0)
    pred = model.predict(inp, verbose=0)
    # if output is (H,W,1) with probabilities:
    pred_mask = (pred[0, :, :, 0] > 0.5).astype(np.uint8) * 255
    # resize back to original image size
    mask_resized = cv2.resize(pred_mask, (img_bgr.shape[1], img_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
    return mask_resized

def mask_to_bytes(mask):
    _, buffer = cv2.imencode(".png", mask)
    return BytesIO(buffer).getvalue()

# ===============================================================
# 3) Sidebar controls
# ===============================================================
st.sidebar.header("⚙️ Controls")
alpha = st.sidebar.slider("Overlay Transparency", 0.0, 1.0, 0.7, 0.05)
colormap = st.sidebar.selectbox("Mask Colormap", ["JET", "VIRIDIS", "PLASMA", "HOT", "COOL"])
device = "GPU" if tf.config.list_physical_devices('GPU') else "CPU"
st.sidebar.info(f"💻 Running on: **{device}**")

# ===============================================================
# 4) Upload & run
# ===============================================================
uploaded_files = st.file_uploader("📁 Upload microscopy images (PNG/JPG)", type=["png","jpg","jpeg"], accept_multiple_files=True)

if uploaded_files:
    st.markdown(f"### 📂 {len(uploaded_files)} file(s) uploaded")
    # Prepare zip container
    masks_zip_io = BytesIO()
    zipf = zipfile.ZipFile(masks_zip_io, 'w', zipfile.ZIP_DEFLATED)

    # Per-file processing & UI
    # We'll offer both individual (commented UI) and 'Segment All' behavior.
    if st.button("🔍 Segment All Images", key="segment_all"):
        start_time = time.time()
        for uploaded_file in uploaded_files:
            uploaded_file.seek(0)  # important to reset pointer
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            if img is None:
                st.warning(f"⚠️ Could not read {uploaded_file.name}, skipping.")
                continue

            # Predict
            mask = predict_mask(model, img, target_size=(224,224))
            # Color map and overlay
            cmap = getattr(cv2, f"COLORMAP_{colormap}")
            mask_color = cv2.applyColorMap(mask.astype(np.uint8), cmap)
            overlay = cv2.addWeighted(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), alpha, mask_color, 1 - alpha, 0)

            # Add to zip
            zipf.writestr(f"{uploaded_file.name.split('.')[0]}_mask.png", mask_to_bytes(mask))
            # Save overlay too
            overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
            _, overlay_buf = cv2.imencode(".png", overlay_bgr)
            zipf.writestr(f"{uploaded_file.name.split('.')[0]}_overlay.png", overlay_buf.tobytes())

            # Display in collapsible expander
            with st.expander(f"🔍 {uploaded_file.name}", expanded=False):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), width=350, caption="Original")
                with col2:
                    st.image(mask, width=350, caption="Predicted Mask")
                with col3:
                    st.image(overlay, width=350, caption="Overlay")

        zipf.close()
        masks_zip_io.seek(0)
        elapsed = time.time() - start_time
        st.success(f"Batch segmentation finished in {elapsed:.1f}s")

        # Download ZIP of masks + overlays
        st.download_button(
            "💾 Download All Results (ZIP)",
            data=masks_zip_io,
            file_name="segmentation_results.zip",
            mime="application/zip"
        )

    # Also allow individual segmentation buttons and download (optional UI variant)
    # st.markdown("---")
    # st.markdown("### 🔎 Individual image controls")
    # for uploaded_file in uploaded_files:
    #     uploaded_file.seek(0)
    #     file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    #     img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    #     if img is None:
    #         st.warning(f"⚠️ Could not read {uploaded_file.name}.")
    #         continue

    #     st.markdown(f"**{uploaded_file.name}**")
    #     c1, c2 = st.columns([1, 3])
    #     with c1:
    #         if st.button(f"Segment {uploaded_file.name}", key=f"seg_{uploaded_file.name}"):
    #             mask = predict_mask(model, img, target_size=(224,224))
    #             cmap = getattr(cv2, f"COLORMAP_{colormap}")
    #             mask_color = cv2.applyColorMap(mask.astype(np.uint8), cmap)
    #             overlay = cv2.addWeighted(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), alpha, mask_color, 1 - alpha, 0)

    #             col1, col2, col3 = st.columns(3)
    #             with col1: st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), width=350, caption="Original")
    #             with col2: st.image(mask, width=350, caption="Predicted Mask")
    #             with col3: st.image(overlay, width=350, caption="Overlay")

    #             # add to zip (so batch zip accumulates these too)
    #             zipf = zipfile.ZipFile(masks_zip_io, 'a', zipfile.ZIP_DEFLATED)
    #             zipf.writestr(f"{uploaded_file.name.split('.')[0]}_mask.png", mask_to_bytes(mask))
    #             overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    #             _, overlay_buf = cv2.imencode(".png", overlay_bgr)
    #             zipf.writestr(f"{uploaded_file.name.split('.')[0]}_overlay.png", overlay_buf.tobytes())
    #             zipf.close()
    #             masks_zip_io.seek(0)

    #     with c2:
    #         # small info and download
    #         st.caption("Preview and download")
    #         # tiny preview
    #         st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), width=150)
    #         if st.button(f"Download mask ({uploaded_file.name})", key=f"dl_{uploaded_file.name}"):
    #             # ensure mask exists: run prediction again quickly (cheap)
    #             mask = predict_mask(model, img, target_size=(224,224))
    #             st.download_button(
    #                 label=f"💾 Download {uploaded_file.name.split('.')[0]}_mask.png",
    #                 data=mask_to_bytes(mask),
    #                 file_name=f"{uploaded_file.name.split('.')[0]}_mask.png",
    #                 mime="image/png",
    #                 key=f"dlbtn_{uploaded_file.name}"
    #             )

# Footer
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:gray;'>Cell Nuclei Segmentation</p>", unsafe_allow_html=True)
