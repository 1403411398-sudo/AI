import streamlit as st
import numpy as np
import os
from PIL import Image

from dinov2_numpy import Dinov2Numpy
from preprocess_image import resize_short_side

WEIGHTS_PATH = "vit-dinov2-base.npz"
INDEX_FILE = "gallery_features.npy"
GALLERY_DIR = "gallery_images"
TOP_K = 10  


st.set_page_config(page_title="ViT以图搜图系统", layout="wide")
st.title("图像检索系统(Vision Transformer)")

@st.cache_resource
def load_model():
    if not os.path.exists(WEIGHTS_PATH):
        st.error(f"找不到权重文件: {WEIGHTS_PATH}")
        return None
    with st.spinner("正在加载模型权重 (这可能需要几秒钟)..."):
        weights = np.load(WEIGHTS_PATH)
        model = Dinov2Numpy(weights)
    return model

@st.cache_data
def load_index():
    if not os.path.exists(INDEX_FILE):
        st.error(f"找不到特征索引: {INDEX_FILE}，请先运行 retrieval.py")
        return None, None
    
    data = np.load(INDEX_FILE, allow_pickle=True).item()
    return data["feats"], data["paths"]

def run_search(model, gallery_feats, gallery_paths, uploaded_file):
    temp_path = "temp_query.jpg"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    try:
        pixel_values = resize_short_side(temp_path)
        query_feat = model(pixel_values) 
        
        norm = np.linalg.norm(query_feat, axis=1, keepdims=True)
        query_feat = query_feat / (norm + 1e-6)
        
        scores = np.dot(query_feat, gallery_feats.T).flatten()
        
        top_indices = np.argsort(scores)[-TOP_K:][::-1]
        
        results = []
        for idx in top_indices:
            results.append((scores[idx], gallery_paths[idx]))
            
        return results
        
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

st.sidebar.header("系统状态")
model = load_model()
gallery_feats, gallery_paths = load_index()

if model is not None and gallery_feats is not None:
    st.sidebar.success("模型已加载")
    st.sidebar.success(f"索引库已加载 ({len(gallery_paths)} 张图片)")
else:
    st.sidebar.error("系统未就绪")


uploaded_file = st.file_uploader("请上传一张图片", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    col1, col2 = st.columns([1, 3])
    with col1:
        st.image(uploaded_file, caption="查询图片", use_container_width=True)
    
    with col2:
        st.write("### 检索结果")
        
        if st.button("开始搜索", type="primary"):
            if model and gallery_feats is not None:
                start_msg = st.empty()
                start_msg.info("正在计算特征并比对...")
                
                results = run_search(model, gallery_feats, gallery_paths, uploaded_file)
                
                start_msg.empty() 
                
                cols = st.columns(5)
                for i, (score, path) in enumerate(results):
                    col = cols[i % 5]
                    
                    with col:
                        if os.path.exists(path):
                            st.image(path, caption=f"Rank {i+1}\nSim: {score:.4f}", use_container_width=True)
                        else:
                            st.error(f"找不到文件: {path}")
                            
                st.success(f"搜索完成！找到了 {len(results)} 个相似结果。")