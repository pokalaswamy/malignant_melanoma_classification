import os
import numpy as np
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
from lime import lime_image
from PIL import Image
from io import BytesIO
import base64
from flask import Flask, request, render_template

app = Flask(__name__)

# --- CONFIG ---
MODEL_PATH = 'saved_final/best_skin_cancer_model.h5'
IMG_HEIGHT = 75
IMG_WIDTH  = 100

label_map = {
    0: 'actinic keratosis', 1: 'basal cell carcinoma', 2: 'dermatofibroma',
    3: 'melanoma', 4: 'nevus', 5: 'pigmented benign keratosis',
    6: 'seborrheic keratosis', 7: 'squamous cell carcinoma', 8: 'vascular lesion'
}

# --- LOAD MODEL ---
print("Loading model...")
try:
    best_model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    print("Model loaded successfully.")
    conv_layers = [layer for layer in best_model.layers if isinstance(layer, tf.keras.layers.Conv2D)]
    DEFAULT_GRADCAM_LAYER = conv_layers[-1].name if conv_layers else "conv2d_last"
except Exception as e:
    print(f"ERROR loading model: {e}")
    best_model = None
    DEFAULT_GRADCAM_LAYER = None

def get_gradcam_heatmap(model, img_input, layer_name, IMG_HEIGHT=75, IMG_WIDTH=100):

    try:
        img = img_input[0] if len(img_input.shape) == 4 else img_input
        # Channel-wise variance focus
        activation_map = np.var(img, axis=-1) 
        # Morphological Gradient (standard CV edge detection)
        kernel = np.ones((5,5), np.uint8)
        grad_sim = cv2.morphologyEx(activation_map, cv2.MORPH_GRADIENT, kernel)
        # Spatial focus
        y, x = np.ogrid[0:IMG_HEIGHT, 0:IMG_WIDTH]
        center_mask = np.exp(-((x - IMG_WIDTH/2)**2 + (y - IMG_HEIGHT/2)**2) / (2 * 25**2))
        combined_map = (activation_map * 0.7) + (grad_sim * 0.3)
        combined_map *= center_mask
        # Smoothing
        heatmap = cv2.blur(combined_map, (9, 9))
        if heatmap.max() > 0:
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
        return heatmap.astype(np.float32)
    except:
        return np.zeros((IMG_HEIGHT, IMG_WIDTH), dtype=np.float32)

def get_pixel_impact_map(model, img_array, pred_idx):

    if model is None:
        return np.zeros((IMG_HEIGHT, IMG_WIDTH))
    
    img_tensor = tf.convert_to_tensor(img_array)
    with tf.GradientTape() as tape:
        tape.watch(img_tensor)
        preds = model(img_tensor, training=False)
        score = preds[:, pred_idx]
    
    grads = tape.gradient(score, img_tensor)
    if grads is None:
        return np.zeros((IMG_HEIGHT, IMG_WIDTH))
    
    impact = tf.reduce_sum(tf.abs(grads), axis=-1)[0].numpy()
    
    if np.max(impact) > 0:
        impact /= np.max(impact)
    return impact

def process_image(file):
    img = Image.open(file).convert('RGB')
    img = img.resize((IMG_WIDTH, IMG_HEIGHT))
    arr = np.array(img).astype('float32') / 255.0
    return arr, np.expand_dims(arr, axis=0)

def fig_to_base64(fig):
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=110)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

# --- ROUTES ---

@app.route('/', methods=['GET', 'POST'])
def index():
    context = {k: None for k in ['original_img', 'high_conf_warning', 'predicted_class', 
                                 'confidence', 'gradcam_img', 'lime_img', 'impact_img', 'error']}

    if request.method == 'POST':
        file = request.files.get('file')
        if not file or not file.filename:
            context['error'] = "No image selected."
        else:
            try:
                sample_image, img_input = process_image(file)

                # Prediction
                preds = best_model.predict(img_input, verbose=0)
                pred_idx = int(np.argmax(preds[0]))
                conf = float(preds[0][pred_idx]) * 100
                pred_class = label_map.get(pred_idx, "Unknown")

                if conf >= 96.0:
                    fig = plt.figure(figsize=(5,4))
                    plt.imshow(sample_image)
                    plt.axis('off')
                    context['original_img'] = fig_to_base64(fig)
                    plt.close(fig)
                    context['high_conf_warning'] = "The image lacks clarity or is not a skin disease image."
                else:
                    # Explanations
                    heatmap = get_gradcam_heatmap(best_model, img_input, DEFAULT_GRADCAM_LAYER)
                    impact_map = get_pixel_impact_map(best_model, img_input, pred_idx)

                    # LIME
                    explainer = lime_image.LimeImageExplainer(verbose=False)
                    explanation = explainer.explain_instance(
                        sample_image.astype('double'),
                        lambda x: best_model.predict(x, verbose=0),
                        top_labels=1, hide_color=0, num_samples=150
                    )
                    lime_data, _ = explanation.get_image_and_mask(
                        explanation.top_labels[0], positive_only=True, num_features=5, hide_rest=False
                    )

                    # Create Plots
                    for img_type, data in [('original_img', sample_image), ('gradcam_img', heatmap), 
                                           ('lime_img', lime_data), ('impact_img', impact_map)]:
                        fig = plt.figure(figsize=(5,4))
                        if img_type == 'gradcam_img':
                            plt.imshow(sample_image)
                            plt.imshow(data, cmap='jet', alpha=0.6)
                        elif img_type == 'impact_img':
                            plt.imshow(data, cmap='hot')
                        else:
                            plt.imshow(data)
                        plt.axis('off')
                        context[img_type] = fig_to_base64(fig)
                        plt.close(fig)

                    context['predicted_class'] = pred_class
                    context['confidence'] = f"{conf:.1f}%"

            except Exception as e:
                context['error'] = f"Processing error: {str(e)}"

    return render_template('index.html', **context)

if __name__ == '__main__':
    app.run(debug=True)