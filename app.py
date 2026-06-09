"""
app.py
------
Streamlit app for insect wing vein junction detection.

Features:
    - Upload one or many wing images
    - Use trained deep learning heatmap detector
    - Optional fallback classical Sato pipeline
    - Heatmap overlay
    - Per-image CSV download
    - Combined CSV download
    - ZIP download of all results
"""


from __future__ import annotations

import io
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from scipy import ndimage as ndi
from skimage import exposure, filters, morphology, measure, util

from inference import load_detector, detect_junctions, create_overlay


# VERY FIRST Streamlit call
st.set_page_config(
    page_title="Wing Vein Landmarking Tool",
    page_icon="🪽",
    layout="wide",
)


# --------------------------------------------------------------------------
# Session state
# --------------------------------------------------------------------------
if "best_params" not in st.session_state:
    st.session_state.best_params = {}


# --------------------------------------------------------------------------
# Helper functions
# --------------------------------------------------------------------------
def uploaded_file_to_bgr(uploaded_file) -> np.ndarray:
    """
    Convert Streamlit uploaded image to OpenCV BGR image.
    """

    file_bytes = np.asarray(
        bytearray(uploaded_file.read()),
        dtype=np.uint8,
    )

    image_bgr = cv2.imdecode(
        file_bytes,
        cv2.IMREAD_COLOR,
    )

    if image_bgr is None:
        raise ValueError(f"Could not read uploaded image: {uploaded_file.name}")

    return image_bgr


def bgr_to_rgb(image_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def normalize_to_uint8(image: np.ndarray) -> np.ndarray:
    """
    Convert any numeric image to uint8 range 0-255 for display.
    """

    image = image.astype(np.float32)

    min_val = float(np.nanmin(image))
    max_val = float(np.nanmax(image))

    if max_val - min_val < 1e-8:
        return np.zeros_like(image, dtype=np.uint8)

    out = (image - min_val) / (max_val - min_val)
    out = (out * 255).clip(0, 255).astype(np.uint8)

    return out


def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def create_zip_bytes(outputs: Dict[str, Dict]) -> bytes:
    """
    Create ZIP containing:
        - per-image CSV
        - per-image overlay PNG
        - combined CSV
    """

    zip_buffer = io.BytesIO()

    combined_rows = []

    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for image_name, item in outputs.items():
            csv_df = item["df"]
            overlay_bgr = item["overlay_bgr"]

            csv_name = f"{Path(image_name).stem}_junctions.csv"
            png_name = f"{Path(image_name).stem}_overlay.png"

            zf.writestr(
                csv_name,
                csv_df.to_csv(index=False),
            )

            ok, encoded_png = cv2.imencode(".png", overlay_bgr)

            if ok:
                zf.writestr(
                    png_name,
                    encoded_png.tobytes(),
                )

            for _, row in csv_df.iterrows():
                combined_rows.append(
                    {
                        "image": image_name,
                        "type": row.get("type", "junction"),
                        "x": row["x"],
                        "y": row["y"],
                    }
                )

        if combined_rows:
            combined_df = pd.DataFrame(combined_rows)
        else:
            combined_df = pd.DataFrame(columns=["image", "type", "x", "y"])

        zf.writestr(
            "combined_landmarks.csv",
            combined_df.to_csv(index=False),
        )

    zip_buffer.seek(0)
    return zip_buffer.getvalue()


# --------------------------------------------------------------------------
# Fallback classical Sato pipeline
# --------------------------------------------------------------------------
def detect_skeleton_junctions(
    skeleton: np.ndarray,
    min_cluster_size: int = 5,
) -> List[Tuple[int, int]]:
    """
    Detect junctions from skeleton.

    A skeleton pixel is treated as junction if it has 3 or more neighbours.
    Connected junction pixels are clustered and converted to centroids.
    """

    skel = skeleton.astype(bool)

    kernel = np.array(
        [
            [1, 1, 1],
            [1, 0, 1],
            [1, 1, 1],
        ],
        dtype=np.uint8,
    )

    neighbour_count = ndi.convolve(
        skel.astype(np.uint8),
        kernel,
        mode="constant",
        cval=0,
    )

    junction_mask = skel & (neighbour_count >= 3)

    junction_mask = morphology.remove_small_objects(
        junction_mask,
        min_size=max(1, int(min_cluster_size)),
    )

    labelled = measure.label(junction_mask)
    props = measure.regionprops(labelled)

    points = []

    for prop in props:
        cy, cx = prop.centroid
        points.append((int(round(cx)), int(round(cy))))

    return points


def detect_skeleton_endpoints(
    skeleton: np.ndarray,
    min_cluster_size: int = 2,
) -> List[Tuple[int, int]]:
    """
    Detect endpoints from skeleton.

    A skeleton pixel is endpoint if it has exactly one neighbour.
    """

    skel = skeleton.astype(bool)

    kernel = np.array(
        [
            [1, 1, 1],
            [1, 0, 1],
            [1, 1, 1],
        ],
        dtype=np.uint8,
    )

    neighbour_count = ndi.convolve(
        skel.astype(np.uint8),
        kernel,
        mode="constant",
        cval=0,
    )

    endpoint_mask = skel & (neighbour_count == 1)

    endpoint_mask = morphology.remove_small_objects(
        endpoint_mask,
        min_size=max(1, int(min_cluster_size)),
    )

    labelled = measure.label(endpoint_mask)
    props = measure.regionprops(labelled)

    points = []

    for prop in props:
        cy, cx = prop.centroid
        points.append((int(round(cx)), int(round(cy))))

    return points


def process_image_fallback(
    image_bgr: np.ndarray,
    clahe_clip: float = 2.0,
    blur_kernel: int = 3,
    sato_sigma: float = 2.0,
    threshold_percentile: float = 75.0,
    min_object_size: int = 80,
    hole_area: int = 80,
    closing_disk_size: int = 1,
    min_junction_cluster: int = 5,
    detect_endpoints: bool = False,
    min_endpoint_cluster: int = 2,
) -> Dict:
    """
    Classical fallback pipeline:
        image -> grayscale -> CLAHE -> blur -> Sato vesselness
        -> threshold -> clean mask -> skeleton -> junctions
    """

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    gray_float = util.img_as_float(gray)

    # Contrast enhancement
    clahe = exposure.equalize_adapthist(
        gray_float,
        clip_limit=float(clahe_clip) / 10.0,
    )

    # Blur must be odd
    blur_kernel = int(blur_kernel)
    if blur_kernel % 2 == 0:
        blur_kernel += 1

    if blur_kernel > 1:
        blurred = cv2.GaussianBlur(
            clahe.astype(np.float32),
            (blur_kernel, blur_kernel),
            0,
        )
    else:
        blurred = clahe.astype(np.float32)

    # Sato vesselness
    sigmas = [max(1.0, float(sato_sigma))]

    vesselness = filters.sato(
        blurred,
        sigmas=sigmas,
        black_ridges=False,
    )

    vesselness = np.nan_to_num(vesselness)

    threshold_value = np.percentile(
        vesselness,
        float(threshold_percentile),
    )

    mask = vesselness > threshold_value

    mask = morphology.remove_small_objects(
        mask,
        min_size=max(1, int(min_object_size)),
    )

    mask = morphology.remove_small_holes(
        mask,
        area_threshold=max(1, int(hole_area)),
    )

    closing_disk_size = int(closing_disk_size)

    if closing_disk_size > 0:
        mask = morphology.binary_closing(
            mask,
            morphology.disk(closing_disk_size),
        )

    skeleton = morphology.skeletonize(mask)

    junctions = detect_skeleton_junctions(
        skeleton,
        min_cluster_size=min_junction_cluster,
    )

    if detect_endpoints:
        endpoints = detect_skeleton_endpoints(
            skeleton,
            min_cluster_size=min_endpoint_cluster,
        )
    else:
        endpoints = []

    overlay = image_bgr.copy()

    # Draw junctions
    for x, y in junctions:
        cv2.circle(overlay, (x, y), 8, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.circle(overlay, (x, y), 6, (0, 90, 0), 2, cv2.LINE_AA)
        cv2.circle(overlay, (x, y), 4, (0, 255, 0), -1, cv2.LINE_AA)

    # Draw endpoints in blue
    for x, y in endpoints:
        cv2.circle(overlay, (x, y), 6, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.circle(overlay, (x, y), 4, (255, 0, 0), -1, cv2.LINE_AA)

    return {
        "junctions": junctions,
        "endpoints": endpoints,
        "overlay": overlay,
        "vesselness": vesselness,
        "mask": mask,
        "skeleton": skeleton,
    }


# --------------------------------------------------------------------------
# Model loader
# --------------------------------------------------------------------------
@st.cache_resource
def cached_load_model(weights_path: str):
    return load_detector(weights_path)


# --------------------------------------------------------------------------
# Main title and uploader
# --------------------------------------------------------------------------
st.title("🪽 Wing Vein Landmarking Tool")
st.caption(
    "Upload clean insect wing images. The app detects vein junctions and exports coordinates."
)

uploaded_files = st.file_uploader(
    "Upload wing images",
    type=["png", "jpg", "jpeg", "tif", "tiff", "bmp"],
    accept_multiple_files=True,
)


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
with st.sidebar:
    st.header("Controls")

    with st.expander("Help: What do these controls mean?", expanded=False):
        st.markdown(
            """
            **Use trained model**  
            Uses the deep learning heatmap model trained from your image/CSV pairs.

            **Detection threshold**  
            Lower value detects more points. Higher value detects fewer points.  
            Since your current `val_F1=0.257`, try `0.25–0.30`.

            **Min junction spacing**  
            Minimum distance between detected junctions. Increase it if duplicate dots appear.

            **Show heatmap overlay**  
            Shows the model confidence map on top of the wing.

            **Fallback Sato pipeline**  
            Classical image-processing method. Useful before the deep model is good.

            **CLAHE clip**  
            Controls contrast enhancement.

            **Blur kernel**  
            Smooths the image before vessel detection.

            **Sato sigma**  
            Controls vein thickness scale.

            **Threshold percentile**  
            Higher value keeps only stronger vein responses.

            **Min object size / hole area / closing disk**  
            Clean the binary vein mask.

            **Min junction cluster**  
            Removes tiny noisy junction groups.

            **Detect endpoints**  
            Also detects vein endpoints, shown in blue.
            """
        )

    st.subheader("Deep Learning Detector")

    default_model_path = "best_junction_detector.pth"

    model_path = st.text_input(
        "Model weights path",
        value=default_model_path,
    )

    use_dl_model = st.checkbox(
        "Use trained model",
        value=True,
    )

    threshold = st.slider(
        "Detection threshold",
        min_value=0.10,
        max_value=0.90,
        value=float(st.session_state.best_params.get("threshold", 0.65)),
        step=0.05,
    )

    min_distance = st.slider(
        "Min junction spacing",
        min_value=5,
        max_value=30,
        value=int(st.session_state.best_params.get("min_distance", 22)),
        step=1,
    )

    show_heatmap = st.checkbox(
        "Show heatmap overlay",
        value=True,
    )

    st.divider()

    if not use_dl_model:
        st.subheader("Fallback Sato Pipeline")

        clahe_clip = st.slider(
            "CLAHE clip",
            min_value=0.5,
            max_value=5.0,
            value=float(st.session_state.best_params.get("clahe_clip", 2.0)),
            step=0.1,
        )

        blur_kernel = st.slider(
            "Blur kernel",
            min_value=1,
            max_value=15,
            value=int(st.session_state.best_params.get("blur_kernel", 3)),
            step=2,
        )

        sato_sigma = st.slider(
            "Sato sigma",
            min_value=1.0,
            max_value=8.0,
            value=float(st.session_state.best_params.get("sato_sigma", 2.0)),
            step=0.5,
        )

        threshold_percentile = st.slider(
            "Threshold percentile",
            min_value=50.0,
            max_value=99.0,
            value=float(st.session_state.best_params.get("threshold_percentile", 75.0)),
            step=1.0,
        )

        min_object_size = st.slider(
            "Min object size",
            min_value=10,
            max_value=1000,
            value=int(st.session_state.best_params.get("min_object_size", 80)),
            step=10,
        )

        hole_area = st.slider(
            "Hole area",
            min_value=10,
            max_value=1000,
            value=int(st.session_state.best_params.get("hole_area", 80)),
            step=10,
        )

        closing_disk_size = st.slider(
            "Closing disk size",
            min_value=0,
            max_value=10,
            value=int(st.session_state.best_params.get("closing_disk_size", 1)),
            step=1,
        )

        st.subheader("Landmark Detection Controls")

        min_junction_cluster = st.slider(
            "Min junction cluster",
            min_value=1,
            max_value=50,
            value=int(st.session_state.best_params.get("min_junction_cluster", 5)),
            step=1,
        )

        detect_endpoints = st.checkbox(
            "Detect endpoints",
            value=False,
        )

        min_endpoint_cluster = st.slider(
            "Min endpoint cluster",
            min_value=1,
            max_value=30,
            value=int(st.session_state.best_params.get("min_endpoint_cluster", 2)),
            step=1,
        )

        show_intermediate = st.checkbox(
            "Show intermediate images",
            value=False,
        )

    else:
        show_intermediate = False
        clahe_clip = 2.0
        blur_kernel = 3
        sato_sigma = 2.0
        threshold_percentile = 75.0
        min_object_size = 80
        hole_area = 80
        closing_disk_size = 1
        min_junction_cluster = 5
        detect_endpoints = False
        min_endpoint_cluster = 2

    st.divider()

    st.subheader("Auto-tune")

    n_trials = st.slider(
        "Number of trials",
        min_value=5,
        max_value=50,
        value=10,
        step=5,
    )

    auto_tune_clicked = st.button(
        "Auto-tune",
        disabled=not uploaded_files,
    )

    if auto_tune_clicked:
        st.info(
            "Auto-tune placeholder: for the DL model, tune threshold manually first. "
            "For your current val_F1=0.257, start with threshold 0.25–0.30."
        )


# --------------------------------------------------------------------------
# Load model if needed
# --------------------------------------------------------------------------
model = None

if use_dl_model:
    try:
        model = cached_load_model(model_path)
        st.success(f"Loaded model: {model_path}")
    except Exception as e:
        st.error(
            f"Could not load model from `{model_path}`. "
            f"Turn off 'Use trained model' to use fallback pipeline.\n\nError: {e}"
        )
        model = None


# --------------------------------------------------------------------------
# Main processing
# --------------------------------------------------------------------------
all_rows = []
outputs = {}

if not uploaded_files:
    st.info("Upload one or more wing images to begin.")

else:
    for uploaded_file in uploaded_files:
        st.divider()
        st.subheader(uploaded_file.name)

        try:
            image_bgr = uploaded_file_to_bgr(uploaded_file)

            if use_dl_model and model is not None:
                junctions, heatmap = detect_junctions(
                    image_bgr=image_bgr,
                    model=model,
                    threshold=threshold,
                    min_distance=min_distance,
                )

                overlay = create_overlay(
                    image_bgr=image_bgr,
                    junctions=junctions,
                    heatmap=heatmap if show_heatmap else None,
                    alpha=0.35,
                )

                endpoints = []

                result = {
                    "junctions": junctions,
                    "endpoints": endpoints,
                    "overlay": overlay,
                    "heatmap": heatmap,
                }

            else:
                result = process_image_fallback(
                    image_bgr=image_bgr,
                    clahe_clip=clahe_clip,
                    blur_kernel=blur_kernel,
                    sato_sigma=sato_sigma,
                    threshold_percentile=threshold_percentile,
                    min_object_size=min_object_size,
                    hole_area=hole_area,
                    closing_disk_size=closing_disk_size,
                    min_junction_cluster=min_junction_cluster,
                    detect_endpoints=detect_endpoints,
                    min_endpoint_cluster=min_endpoint_cluster,
                )

                junctions = result["junctions"]
                endpoints = result["endpoints"]
                overlay = result["overlay"]
                heatmap = None

            # Display original and overlay
            col1, col2 = st.columns(2)

            with col1:
                st.image(
                    bgr_to_rgb(image_bgr),
                    caption="Original image",
                    use_container_width=True,
                )

            with col2:
                st.image(
                    bgr_to_rgb(overlay),
                    caption=f"Detected junctions: {len(junctions)}",
                    use_container_width=True,
                )

            if use_dl_model and show_heatmap and "heatmap" in result:
                st.image(
                    normalize_to_uint8(result["heatmap"]),
                    caption="Predicted heatmap",
                    use_container_width=True,
                )

            if show_intermediate and not use_dl_model:
                st.markdown("#### Intermediate outputs")

                c1, c2, c3 = st.columns(3)

                with c1:
                    st.image(
                        normalize_to_uint8(result["vesselness"]),
                        caption="Sato vesselness",
                        use_container_width=True,
                    )

                with c2:
                    st.image(
                        result["mask"].astype(np.uint8) * 255,
                        caption="Binary mask",
                        use_container_width=True,
                    )

                with c3:
                    st.image(
                        result["skeleton"].astype(np.uint8) * 255,
                        caption="Skeleton",
                        use_container_width=True,
                    )

            # Build per-image dataframe
            rows = []

            for x, y in junctions:
                rows.append(
                    {
                        "image": uploaded_file.name,
                        "type": "junction",
                        "x": int(x),
                        "y": int(y),
                    }
                )

            for x, y in endpoints:
                rows.append(
                    {
                        "image": uploaded_file.name,
                        "type": "endpoint",
                        "x": int(x),
                        "y": int(y),
                    }
                )

            df = pd.DataFrame(
                rows,
                columns=["image", "type", "x", "y"],
            )

            st.markdown("#### Landmark coordinates")
            st.dataframe(df, use_container_width=True)

            csv_bytes = dataframe_to_csv_bytes(df)

            st.download_button(
                label=f"Download CSV for {uploaded_file.name}",
                data=csv_bytes,
                file_name=f"{Path(uploaded_file.name).stem}_landmarks.csv",
                mime="text/csv",
            )

            all_rows.extend(rows)

            outputs[uploaded_file.name] = {
                "df": df,
                "overlay_bgr": overlay,
            }

        except Exception as e:
            st.error(f"Error processing {uploaded_file.name}: {e}")


# --------------------------------------------------------------------------
# Combined outputs
# --------------------------------------------------------------------------
if all_rows:
    st.divider()
    st.header("Combined outputs")

    combined_df = pd.DataFrame(
        all_rows,
        columns=["image", "type", "x", "y"],
    )

    st.dataframe(
        combined_df,
        use_container_width=True,
    )

    st.download_button(
        label="Download combined CSV",
        data=dataframe_to_csv_bytes(combined_df),
        file_name="combined_landmarks.csv",
        mime="text/csv",
    )

    zip_bytes = create_zip_bytes(outputs)

    st.download_button(
        label="Download ZIP of all outputs",
        data=zip_bytes,
        file_name="wing_junction_outputs.zip",
        mime="application/zip",
    )