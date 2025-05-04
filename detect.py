import cv2
import numpy as np
import tensorflow as tf
from typing import Tuple, Optional
import logging
from scipy.stats import entropy
import os
import uuid
import math

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LeukemiaDetector:
    def __init__(self, model_path: str):
        """Initialize the detector with the combined VGG16 U-Net model."""
        try:
            logger.debug(f"Loading model from: {model_path}")
            self.model = tf.keras.models.load_model(model_path)
            self.input_size = (256, 256)
            self.labels = ["Benign", "Malignant - Early", "Malignant - Pre", "Malignant - Pro"]
            sample_input = np.zeros((1, 256, 256, 3))
            sample_output = self.model.predict(sample_input, verbose=0)
            if len(sample_output) != 2 or sample_output[0].shape[1] != len(self.labels) or sample_output[1].shape[1:3] != (256, 256):
                raise ValueError(f"Unexpected model output shapes: {sample_output[0].shape}, {sample_output[1].shape}")
            logger.info(f"Model loaded successfully with input shape: (None, 256, 256, 3)")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self.model = None
            raise

    def macenko_normalization(self, image: np.ndarray) -> np.ndarray:
        """Apply Macenko stain normalization with mild contrast."""
        try:
            image = image.astype(np.float32)
            image = np.maximum(image, 1e-6)
            od = -np.log(image / 255.0)
            cov = np.cov(od.reshape(-1, 3).T)
            eigvals, eigvecs = np.linalg.eigh(cov)
            stain_matrix = eigvecs[:, np.argsort(eigvals)[::-1]][:, :2]
            stains = np.dot(od.reshape(-1, 3), stain_matrix).reshape(od.shape[0], od.shape[1], 2)
            stains = (stains - np.mean(stains, axis=(0, 1))) / (np.std(stains, axis=(0, 1)) + 1e-8) * 2.0
            normalized_od = np.dot(stains, stain_matrix.T)
            normalized_image = 255.0 * np.exp(-normalized_od)
            normalized_image = np.clip(normalized_image, 0, 255).astype(np.uint8)
            logger.debug("Applied Macenko stain normalization with mild contrast")
            return normalized_image
        except Exception as e:
            logger.warning(f"Macenko normalization failed: {e}")
            return image

    def enhance_image(self, image: np.ndarray) -> np.ndarray:
        """Enhance image brightness, contrast, and reduce blur."""
        try:
            debug_dir = "static/debug"
            os.makedirs(debug_dir, exist_ok=True)
            raw_filename = f"raw_{uuid.uuid4().hex}.png"
            cv2.imwrite(os.path.join(debug_dir, raw_filename), image)
            logger.debug(f"Saved raw image to {os.path.join(debug_dir, raw_filename)}")
            
            image = self.macenko_normalization(image)
            image = cv2.bilateralFilter(image, d=9, sigmaColor=50, sigmaSpace=50)
            clahe = cv2.createCLAHE(clipLimit=15.0, tileGridSize=(8, 8))
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            l = clahe.apply(l)
            lab = cv2.merge((l, a, b))
            enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            mean_intensity = np.mean(enhanced)
            gamma = 1.0 if mean_intensity > 128 else 0.4
            enhanced = np.power(enhanced / 255.0, gamma) * 255.0
            enhanced = enhanced.astype(np.uint8)
            enhanced = cv2.ximgproc.anisotropicDiffusion(enhanced, alpha=0.1, K=10, niters=25)
            
            preprocessed_filename = f"preprocessed_{uuid.uuid4().hex}.png"
            cv2.imwrite(os.path.join(debug_dir, preprocessed_filename), enhanced)
            logger.debug(f"Saved preprocessed image to {os.path.join(debug_dir, preprocessed_filename)}")
            logger.debug(f"Image stats - Mean: {np.mean(enhanced):.2f}, Std: {np.std(enhanced):.2f}")
            logger.debug("Image enhanced with Macenko normalization, bilateral filter, CLAHE, gamma correction, and anisotropic diffusion")
            return enhanced
        except Exception as e:
            logger.error(f"Image enhancement failed: {e}")
            return image

    def validate_blood_smear(self, image: np.ndarray) -> bool:
        """Validate if the image is a properly stained blood smear (color only)."""
        try:
            logger.debug(f"Validating image with shape: {image.shape}")
            enhanced_image = self.enhance_image(image)
            hsv = cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv)
            mean_hue = np.mean(h)
            mean_sat = np.mean(s)
            mean_val = np.mean(v)
            logger.debug(f"HSV Means - Hue: {mean_hue:.2f}, Sat: {mean_sat:.2f}, Val: {mean_val:.2f}")
            is_valid_color = 0 <= mean_hue <= 255 and mean_sat >= 0.000001 and mean_val >= 0.01
            if not is_valid_color:
                logger.warning(f"Color validation failed: Hue={mean_hue:.2f}, Sat={mean_sat:.2f}, Val={mean_val:.2f}")
                return False
            logger.info("Image validation passed (color check)")
            return True
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            return False

    def preprocess_image(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Preprocess image for model input."""
        try:
            logger.debug(f"Preprocessing image with shape: {image.shape}")
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
            resized_image = cv2.resize(image, self.input_size)
            processed_image = resized_image.astype(np.float32) / 255.0
            processed_image = np.expand_dims(processed_image, axis=0)
            logger.debug(f"Processed image shape: {processed_image.shape}")
            return processed_image
        except Exception as e:
            logger.error(f"Preprocessing failed: {e}")
            return None

    def predict(self, processed_image: np.ndarray) -> Tuple[str, float, np.ndarray, np.ndarray]:
        try:
            logger.debug("Running model prediction")
            cls_pred, seg_pred = self.model.predict(processed_image, verbose=0)
            cls_pred = cls_pred / np.sum(cls_pred, axis=1, keepdims=True)
            probs = cls_pred[0]
            sorted_idx = np.argsort(probs)[::-1]
            top_class = sorted_idx[0]
            second_class = sorted_idx[1]
            top_prob = probs[top_class]
            second_prob = probs[second_class]
            
            if top_prob < 0.7 or (top_prob - second_prob < 0.15):
                predicted_label = "Uncertain"
                confidence = 0.0
            else:
                predicted_label = self.labels[top_class]
                confidence = top_prob
            
            pred_entropy = entropy(probs)
            logger.debug(f"Prediction entropy: {pred_entropy:.4f}")
            if pred_entropy > 1.2:
                logger.warning(f"High uncertainty in prediction (entropy: {pred_entropy:.4f}). Manual review recommended.")
            if confidence < 0.9 and predicted_label != "Uncertain":
                logger.warning(f"Confidence below 90% ({confidence*100:.2f}%). Manual review recommended.")
            
            logger.info(f"Classification probabilities - Benign: {probs[0]:.4f}, Malignant-Early: {probs[1]:.4f}, Malignant-Pre: {probs[2]:.4f}, Malignant-Pro: {probs[3]:.4f}")
            if predicted_label == "Benign" and probs[1] > 0.3:
                logger.warning(f"Benign image has high probability for Malignant - Early: {probs[1]:.4f}")
            
            logger.info(f"Prediction: {predicted_label}, Confidence: {confidence:.4f}")
            return predicted_label, confidence, cls_pred, seg_pred
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return "Error", 0.0, np.array([]), np.array([])

    def localize_cells(self, image: np.ndarray, prediction: str, confidence: float, seg_prediction: np.ndarray) -> Tuple[np.ndarray, int]:
        try:
            localized_image = image.copy()
            if prediction == "Uncertain" or prediction == "Error" or prediction == "Benign":
                logger.warning(f"Skipping localization for {prediction} prediction")
                return localized_image, 0

            mask = seg_prediction[0]
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
            mask = (mask > 0.2).astype(np.uint8) * 255

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            color = (0, 0, 255)  # Red for malignant
            for contour in contours:
                cv2.drawContours(localized_image, [contour], -1, color, 2)

            cell_count = len(contours)
            logger.debug(f"Mask stats - Min: {mask.min():.4f}, Max: {mask.max():.4f}, Mean: {mask.mean():.4f}")
            logger.info(f"Detected {cell_count} cancerous cells")
            return localized_image, cell_count
        except Exception as e:
            logger.error(f"Localization failed: {e}")
            return image, 0

    def detect_cancer(self, image: np.ndarray, perform_segmentation: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray], str, float, int]:
        try:
            original_image = image.copy()
            if not self.validate_blood_smear(image):
                logger.warning("Invalid blood smear image")
                return original_image, None, "Invalid image", 0.0, 0

            if self.model is None:
                logger.error("Model not initialized")
                return original_image, None, "Error", 0.0, 0

            processed_image = self.preprocess_image(image)
            if processed_image is None:
                logger.error("Image preprocessing failed")
                return original_image, None, "Error", 0.0, 0

            prediction, confidence, cls_pred, seg_pred = self.predict(processed_image)
            if prediction == "Error":
                logger.error("Prediction failed")
                return original_image, None, "Error", 0.0, 0

            localized_image, cell_count = None, 0
            if perform_segmentation and prediction not in ["Uncertain", "Benign"]:
                localized_image, cell_count = self.localize_cells(original_image, prediction, confidence, seg_pred)

            logger.info(f"Final result: {prediction}, Confidence: {confidence:.4f}, Cell count: {cell_count}")
            return original_image, localized_image, prediction, confidence, cell_count
        except Exception as e:
            logger.error(f"Detection failed: {e}")
            return original_image, None, "Error", 0.0, 0

if __name__ == "__main__":
    detector = LeukemiaDetector("path_to_model.h5")
    image = cv2.imread("path_to_image.jpg")
    original, localized, pred, conf, count = detector.detect_cancer(image)
    if localized is not None:
        cv2.imwrite("output_marked.jpg", localized)
    print(f"Prediction: {pred}, Confidence: {conf}, Cells Detected: {count}")