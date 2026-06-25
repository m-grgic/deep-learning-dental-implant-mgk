"""
Utility functions for the dental X-ray pose estimation thesis project.

Provides helpers for IoU calculation, keypoint geometry, OKS scoring,
bounding-box / skeleton visualisation, and ground-truth data loading.
"""

from __future__ import annotations

import os
import glob
import time
from typing import Tuple

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from shapely.geometry import MultiPoint
from sklearn.metrics import accuracy_score, confusion_matrix

from config import (
    CONF_THRESHOLD,
    IOU_THRESHOLD,
    IMG_SIZE,
    OKS_LOW,
    OKS_HIGH,
    BASE_PATH,
)

# ---------------------------------------------------------------------------
# Google Colab compatibility
# ---------------------------------------------------------------------------
try:
    from google.colab.patches import cv2_imshow as _cv2_imshow
    from google.colab import drive as _drive
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

    def _cv2_imshow(img: np.ndarray) -> None:  # type: ignore[misc]
        """Fallback for cv2_imshow outside Colab – opens a window."""
        cv2.imshow("image", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# TypedDict definitions
# ---------------------------------------------------------------------------
from typing import TypedDict


class Tocka(TypedDict):
    x: float
    y: float


class Vektor(TypedDict):
    i: float
    j: float


class Keypoints(TypedDict):
    apex: Tocka
    desni_kost: Tocka
    desni_vrh: Tocka
    vrh: Tocka
    lijevi_vrh: Tocka
    lijevi_kost: Tocka


class True_vs_Pred(TypedDict):
    true: Keypoints
    pred: Keypoints
    true_box_wh: Tuple[float, float]


# ---------------------------------------------------------------------------
# Bounding-box helpers
# ---------------------------------------------------------------------------

def IoU(
    pravokutnik_true: np.ndarray,
    pravokutnik_pred: np.ndarray,
) -> Tuple[float, list, list]:
    """Calculate Intersection over Union for two bounding boxes.

    Args:
        pravokutnik_true: Array [x_center, y_center, w, h] for the ground-truth box.
        pravokutnik_pred: Array [x_center, y_center, w, h] for the predicted box.

    Returns:
        Tuple of (iou_score, true_corner_points, pred_corner_points).
    """
    # PRAVOKUTNIK UVIJEK IMA FORMAT [x_srednji, y_srednji, w, h]
    def _corners(r: np.ndarray) -> list:
        return [
            (r[0] - r[2] / 2, r[1] - r[3] / 2),
            (r[0] - r[2] / 2, r[1] + r[3] / 2),
            (r[0] + r[2] / 2, r[1] + r[3] / 2),
            (r[0] + r[2] / 2, r[1] - r[3] / 2),
        ]

    true_points = _corners(pravokutnik_true)
    pred_points = _corners(pravokutnik_pred)

    true_poly = MultiPoint(true_points).convex_hull
    pred_poly = MultiPoint(pred_points).convex_hull

    intersection_area = true_poly.intersection(pred_poly).area
    union_area = true_poly.union(pred_poly).area

    return intersection_area / union_area, true_points, pred_points


def choose_the_best_match(
    pravokutnik_true: np.ndarray,
    svi_pred_pravokutnici: np.ndarray,
) -> np.ndarray | None:
    """Return the predicted bounding box with the highest IoU against the ground truth.

    Args:
        pravokutnik_true: Ground-truth box as [x_c, y_c, w, h].
        svi_pred_pravokutnici: Array of predicted boxes, each [x_c, y_c, w, h].

    Returns:
        The best-matching predicted box, or None if the list is empty.
    """
    max_iou = 0.0
    returning_pravokutnik = None
    for pravokutnik in svi_pred_pravokutnici:
        iou_score, _, _ = IoU(pravokutnik_true, pravokutnik)  # compute once
        if iou_score > max_iou:
            returning_pravokutnik = pravokutnik
            max_iou = iou_score
    return returning_pravokutnik


# ---------------------------------------------------------------------------
# Ground-truth loading helpers
# ---------------------------------------------------------------------------

def getTrueData(img_path: str, lbl_dir: str) -> np.ndarray:
    """Load YOLO-format annotation file for *img_path* from *lbl_dir*.

    Args:
        img_path: Full path to the image file.
        lbl_dir:  Directory that contains the label ``.txt`` file.

    Returns:
        2-D numpy array where each row is one annotated instance.

    Raises:
        ValueError: If the label file does not exist.
    """
    fname = os.path.splitext(os.path.basename(img_path))[0]
    lbl_path = os.path.join(lbl_dir, f"{fname}.txt")

    if not os.path.isfile(lbl_path):
        print(lbl_path)
        raise ValueError(f"Label file not found: {lbl_path}")

    true_data = np.loadtxt(lbl_path)
    if len(true_data.shape) != 2:
        true_data = np.array([true_data])
    return true_data


def getTrueNumberOfImages(img_path: str, lbl_dir: str) -> int:
    """Return the number of annotated instances in the label file.

    Args:
        img_path: Full path to the image file.
        lbl_dir:  Directory that contains the label ``.txt`` file.

    Returns:
        Count of annotated instances.
    """
    return len(getTrueData(img_path, lbl_dir))


# ---------------------------------------------------------------------------
# Keypoint geometry helpers
# ---------------------------------------------------------------------------

def getApexVrh(
    key_point_vectors: np.ndarray,
    width: int,
    height: int,
) -> Tuple[Tocka, Tocka, Vektor]:
    """Extract apex and crown keypoints and their connecting vector.

    Args:
        key_point_vectors: Flat array of normalised keypoint coordinates.
        width:  Image width in pixels.
        height: Image height in pixels.

    Returns:
        Tuple (APEX Tocka, VRH Tocka, vector from VRH to APEX).
    """
    x_px_apex = key_point_vectors[0] * width
    y_px_apex = key_point_vectors[1] * height
    x_px_vrh = key_point_vectors[9] * width
    y_px_vrh = key_point_vectors[10] * height

    return (
        Tocka(x=x_px_apex, y=y_px_apex),
        Tocka(x=x_px_vrh, y=y_px_vrh),
        Vektor(i=x_px_apex - x_px_vrh, j=y_px_apex - y_px_vrh),
    )


def getRightSide(
    key_point_vectors: np.ndarray,
    width: int,
    height: int,
) -> Tuple[Tocka, Tocka, Vektor]:
    """Extract the right-side bone-loss keypoints and their vector.

    Args:
        key_point_vectors: Flat array of normalised keypoint coordinates.
        width:  Image width in pixels.
        height: Image height in pixels.

    Returns:
        Tuple (desni_donji Tocka, desni_gornji Tocka, vector).
    """
    x_px_DD = key_point_vectors[3] * width
    y_px_DD = key_point_vectors[4] * height
    x_px_DG = key_point_vectors[6] * width
    y_px_DG = key_point_vectors[7] * height

    return (
        Tocka(x=x_px_DD, y=y_px_DD),
        Tocka(x=x_px_DG, y=y_px_DG),
        Vektor(i=x_px_DD - x_px_DG, j=y_px_DD - y_px_DG),
    )


def getLeftSide(
    key_point_vectors: np.ndarray,
    width: int,
    height: int,
) -> Tuple[Tocka, Tocka, Vektor]:
    """Extract the left-side bone-loss keypoints and their vector.

    Args:
        key_point_vectors: Flat array of normalised keypoint coordinates.
        width:  Image width in pixels.
        height: Image height in pixels.

    Returns:
        Tuple (lijevi_gornji Tocka, lijevi_donji Tocka, vector).
    """
    x_px_LG = key_point_vectors[12] * width
    y_px_LG = key_point_vectors[13] * height
    x_px_LD = key_point_vectors[15] * width
    y_px_LD = key_point_vectors[16] * height

    return (
        Tocka(x=x_px_LG, y=y_px_LG),
        Tocka(x=x_px_LD, y=y_px_LD),
        Vektor(i=x_px_LD - x_px_LG, j=y_px_LD - y_px_LG),
    )


def pointProjection(tocka_P: Tocka, tocka_V: Tocka, vector_V: Vektor) -> Tocka:
    """Project point P orthogonally onto the line through V with direction vector_V.

    Args:
        tocka_P:  The point to project.
        tocka_V:  A reference point on the line.
        vector_V: The direction vector of the line.

    Returns:
        The projected Tocka on the line.
    """
    np_V = np.array([tocka_V["x"], tocka_V["y"]])
    np_P = np.array([tocka_P["x"], tocka_P["y"]])
    np_vec = np.array([vector_V["i"], vector_V["j"]])

    projekcija = np_V + (np.dot(np_P - np_V, np_vec) / np.linalg.norm(np_vec) ** 2) * np_vec

    vektor_projekcije = Vektor(
        i=projekcija[0] - tocka_P["x"],
        j=projekcija[1] - tocka_P["y"],
    )
    assert (
        abs(vektor_projekcije["i"] * vector_V["i"] + vektor_projekcije["j"] * vector_V["j"]) < 5e-2
    ), f"{vektor_projekcije['i'] * vector_V['i'] + vektor_projekcije['j'] * vector_V['j']}"

    return Tocka(x=projekcija[0], y=projekcija[1])


def vectorProjection(
    tocka_hvatiste: Tocka,
    tocka_vrh: Tocka,
    tocka_V: Tocka,
    vektor_V: Vektor,
) -> Tuple[Tocka, Tocka, float]:
    """Project two points onto a reference line and return the projected distance.

    Args:
        tocka_hvatiste: Start point of the segment.
        tocka_vrh:      End point of the segment.
        tocka_V:        A reference point on the line.
        vektor_V:       Direction vector of the line.

    Returns:
        Tuple (projected hvatiste, projected vrh, signed distance).
    """
    projekcija_hvatiste = pointProjection(tocka_hvatiste, tocka_V, vektor_V)
    projekcija_vrh = pointProjection(tocka_vrh, tocka_V, vektor_V)

    vektor_HV = Vektor(
        i=projekcija_vrh["x"] - projekcija_hvatiste["x"],
        j=projekcija_vrh["y"] - projekcija_hvatiste["y"],
    )

    distance = 0.0
    if vektor_HV["i"] * vektor_V["i"] + vektor_HV["j"] * vektor_V["j"] > 0:
        distance = np.sqrt(vektor_HV["i"] ** 2 + vektor_HV["j"] ** 2)

    return projekcija_hvatiste, projekcija_vrh, distance


def getTwoSides(
    desni_donji: Tocka,
    desni_gornji: Tocka,
    lijevi_donji: Tocka,
    lijevi_gornji: Tocka,
    tocka_V: Tocka,
    vektor_V: Vektor,
) -> dict:
    """Calculate the medial and bilateral bone-side lengths projected onto the median axis.

    Args:
        desni_donji:  Right inferior keypoint.
        desni_gornji: Right superior keypoint.
        lijevi_donji: Left inferior keypoint.
        lijevi_gornji: Left superior keypoint.
        tocka_V:  Median axis reference point (VRH).
        vektor_V: Median axis direction vector.

    Returns:
        Dict with keys ``srednji``, ``desni``, ``lijevi`` (projected distances).
    """
    _, _, distance_desni = vectorProjection(desni_gornji, desni_donji, tocka_V, vektor_V)
    _, _, distance_lijevi = vectorProjection(lijevi_gornji, lijevi_donji, tocka_V, vektor_V)
    return {
        "srednji": np.sqrt(vektor_V["i"] ** 2 + vektor_V["j"] ** 2),
        "desni": distance_desni,
        "lijevi": distance_lijevi,
    }


def get_distances_for_true_pred(true_pred: True_vs_Pred) -> dict:
    """Compute bilateral side distances for a matched true/predicted pair.

    Args:
        true_pred: Dictionary with keys ``true``, ``pred``, ``true_box_wh``.

    Returns:
        Dict ``{"true": {...}, "pred": {...}}`` each containing ``srednji``,
        ``desni``, and ``lijevi`` projected distances.
    """
    medijalni_vektor_true = Vektor(
        i=true_pred["true"]["apex"]["x"] - true_pred["true"]["vrh"]["x"],
        j=true_pred["true"]["apex"]["y"] - true_pred["true"]["vrh"]["y"],
    )
    medijalni_vektor_pred = Vektor(
        i=true_pred["pred"]["apex"]["x"] - true_pred["pred"]["vrh"]["x"],
        j=true_pred["pred"]["apex"]["y"] - true_pred["pred"]["vrh"]["y"],
    )

    return {
        "true": getTwoSides(
            desni_donji=true_pred["true"]["desni_kost"],
            desni_gornji=true_pred["true"]["desni_vrh"],
            lijevi_donji=true_pred["true"]["lijevi_kost"],
            lijevi_gornji=true_pred["true"]["lijevi_vrh"],
            tocka_V=true_pred["true"]["vrh"],
            vektor_V=medijalni_vektor_true,
        ),
        "pred": getTwoSides(
            desni_donji=true_pred["pred"]["desni_kost"],
            desni_gornji=true_pred["pred"]["desni_vrh"],
            lijevi_donji=true_pred["pred"]["lijevi_kost"],
            lijevi_gornji=true_pred["pred"]["lijevi_vrh"],
            tocka_V=true_pred["pred"]["vrh"],
            vektor_V=medijalni_vektor_pred,
        ),
    }


# ---------------------------------------------------------------------------
# Prediction helpers
# ---------------------------------------------------------------------------

def get_pred_skeletons_from_image(
    model,
    image_path: str,
    conf: float = CONF_THRESHOLD,
    iou_conf: float = IOU_THRESHOLD,
) -> np.ndarray:
    """Run the pose model on a single image and return raw keypoint data.

    Args:
        model:      Ultralytics YOLO pose model instance.
        image_path: Path to the input image.
        conf:       Confidence threshold.
        iou_conf:   IoU threshold for NMS.

    Returns:
        Numpy array of shape (N, 6, 3) – N detected skeletons, 6 keypoints each
        with (x, y, confidence).
    """
    prediction = model(image_path, verbose=False)
    predicted_points = np.array(prediction[0].keypoints.data.cpu())
    return predicted_points


def find_corresponding_image_distance(
    model,
    image_path: str,
    conf: float,
    true_keypoints: Keypoints,
    true_box_wh: Tuple[float, float],
    iou_conf: float,
) -> True_vs_Pred | None:
    """Find the predicted skeleton that best matches *true_keypoints*.

    Selects the prediction with the minimum mean scaled Euclidean distance.

    Args:
        model:          Ultralytics YOLO pose model.
        image_path:     Path to the input image.
        conf:           Confidence threshold for inference.
        true_keypoints: Ground-truth keypoints in pixel coordinates.
        true_box_wh:    Ground-truth bounding-box (w, h) for distance scaling.
        iou_conf:       IoU threshold for NMS during inference.

    Returns:
        Dict ``{"true": ..., "pred": ..., "true_box_wh": ...}`` or None if no
        valid prediction is found.
    """
    tocke = get_pred_skeletons_from_image(model, image_path, conf, iou_conf)

    if tocke.size == 0:
        return None

    selection_keypoints = Keypoints(
        apex=None, desni_kost=None, desni_vrh=None,
        vrh=None, lijevi_vrh=None, lijevi_kost=None,
    )
    min_dist = np.inf

    for kostur in tocke:
        if kostur.size >= 6 * 3:
            pred_keypoints = Keypoints(
                apex=Tocka(x=kostur[0][0], y=kostur[0][1]),
                desni_kost=Tocka(x=kostur[1][0], y=kostur[1][1]),
                desni_vrh=Tocka(x=kostur[2][0], y=kostur[2][1]),
                vrh=Tocka(x=kostur[3][0], y=kostur[3][1]),
                lijevi_vrh=Tocka(x=kostur[4][0], y=kostur[4][1]),
                lijevi_kost=Tocka(x=kostur[5][0], y=kostur[5][1]),
            )
            current_mean = PointDistance(
                {"true": true_keypoints, "pred": pred_keypoints, "true_box_wh": true_box_wh},
                scale=True,
                scaling_type="dist",
            )["mean_error"]

            if not np.isnan(current_mean) and current_mean < min_dist:
                min_dist = current_mean
                selection_keypoints = pred_keypoints

    if min_dist == np.inf:
        return None

    return {"true": true_keypoints, "pred": selection_keypoints, "true_box_wh": true_box_wh}


def find_corresponding_image_distance_from_results(
    prediction_results,
    true_keypoints: Keypoints,
    true_box_wh: Tuple[float, float],
) -> True_vs_Pred | None:
    """Like ``find_corresponding_image_distance`` but accepts pre-computed results.

    Args:
        prediction_results: Ultralytics result object (e.g. ``predikcija[0]``).
        true_keypoints:     Ground-truth keypoints in pixel coordinates.
        true_box_wh:        Ground-truth bounding-box (w, h) for distance scaling.

    Returns:
        Dict ``{"true": ..., "pred": ..., "true_box_wh": ...}`` or None.
    """
    predicted_kpts_all = (
        prediction_results.keypoints.data.cpu().numpy()
        if prediction_results.keypoints is not None
        else np.array([])
    )

    if len(predicted_kpts_all) == 0:
        return None

    min_mean_dist = np.inf
    best_pred_kpts = None

    for predicted_kpts_set_data in predicted_kpts_all:
        if predicted_kpts_set_data.size >= 6 * 3:
            predicted_kpts_set = Keypoints(
                apex=Tocka(x=predicted_kpts_set_data[0][0], y=predicted_kpts_set_data[0][1]),
                desni_kost=Tocka(x=predicted_kpts_set_data[1][0], y=predicted_kpts_set_data[1][1]),
                desni_vrh=Tocka(x=predicted_kpts_set_data[2][0], y=predicted_kpts_set_data[2][1]),
                vrh=Tocka(x=predicted_kpts_set_data[3][0], y=predicted_kpts_set_data[3][1]),
                lijevi_vrh=Tocka(x=predicted_kpts_set_data[4][0], y=predicted_kpts_set_data[4][1]),
                lijevi_kost=Tocka(x=predicted_kpts_set_data[5][0], y=predicted_kpts_set_data[5][1]),
            )
            try:
                dist_dict = PointDistance(
                    {"true": true_keypoints, "pred": predicted_kpts_set, "true_box_wh": true_box_wh},
                    scale=True,
                    scaling_type="dist",
                )
                if dist_dict["mean_error"] < min_mean_dist:
                    min_mean_dist = dist_dict["mean_error"]
                    best_pred_kpts = predicted_kpts_set
            except Exception:
                pass

    if best_pred_kpts is not None:
        return {"true": true_keypoints, "pred": best_pred_kpts, "true_box_wh": true_box_wh}
    return None


# ---------------------------------------------------------------------------
# Distance / error metrics
# ---------------------------------------------------------------------------

def PointDistance(
    true_pred: True_vs_Pred,
    scale: bool = True,
    scaling_type: str = "dist",
) -> dict[str, float]:
    """Calculate per-keypoint scaled Euclidean distance between true and predicted.

    Args:
        true_pred:    Dict with ``true``, ``pred``, and ``true_box_wh``.
        scale:        Whether to normalise distances by the bounding-box diagonal.
        scaling_type: ``"dist"`` (diagonal), ``"root"`` (4th root of area), or
                      ``"none"`` (raw pixel distance).

    Returns:
        Dict mapping keypoint names to their (possibly scaled) Euclidean distance,
        plus ``"mean_error"`` across all valid (non-NaN) keypoints.
    """
    assert scaling_type in ("dist", "root", "none")

    if not scale:
        scaling_factor = 1.0
    elif scaling_type == "root":
        scaling_factor = np.sqrt(
            np.sqrt(true_pred["true_box_wh"][0] * true_pred["true_box_wh"][-1])
        )
    else:
        scaling_factor = np.sqrt(
            true_pred["true_box_wh"][0] ** 2 + true_pred["true_box_wh"][-1] ** 2
        )

    dict_vrijednosti: dict[str, float] = {}
    for key in ("apex", "desni_kost", "desni_vrh", "vrh", "lijevi_vrh", "lijevi_kost"):
        if true_pred["pred"][key] is None:
            dict_vrijednosti[key] = np.nan
        else:
            dict_vrijednosti[key] = (
                np.sqrt(
                    (true_pred["true"][key]["x"] - true_pred["pred"][key]["x"]) ** 2
                    + (true_pred["true"][key]["y"] - true_pred["pred"][key]["y"]) ** 2
                )
                / scaling_factor
            )

    valid = [d for d in dict_vrijednosti.values() if not np.isnan(d)]
    dict_vrijednosti["mean_error"] = np.mean(valid) if valid else np.nan
    return dict_vrijednosti


def PointDistanceForSigma(true_pred: True_vs_Pred) -> dict[str, float]:
    """Calculate area-normalised squared distances (used for OKS sigma estimation).

    Args:
        true_pred: Dict with ``true``, ``pred``, and ``true_box_wh``.

    Returns:
        Dict of per-keypoint values and ``"mean_error"``.
    """
    dict_vrijednosti: dict[str, float] = {}
    area = 0.85 * true_pred["true_box_wh"][0] * true_pred["true_box_wh"][-1]

    for key in ("apex", "desni_kost", "desni_vrh", "vrh", "lijevi_vrh", "lijevi_kost"):
        if true_pred["pred"][key] is None:
            dict_vrijednosti[key] = np.nan
        else:
            dict_vrijednosti[key] = (
                (true_pred["true"][key]["x"] - true_pred["pred"][key]["x"]) ** 2
                + (true_pred["true"][key]["y"] - true_pred["pred"][key]["y"]) ** 2
            ) / area

    valid = [d for d in dict_vrijednosti.values() if not np.isnan(d)]
    dict_vrijednosti["mean_error"] = np.mean(valid) if valid else np.nan
    return dict_vrijednosti


# ---------------------------------------------------------------------------
# OKS helpers
# ---------------------------------------------------------------------------

def _edit_keypoints(kpts: list) -> Tuple[np.ndarray, np.ndarray]:
    """Reshape flat keypoint list into (N,2) coords and visibility flags.

    Args:
        kpts: Flat list [x1, y1, v1, x2, y2, v2, ...].

    Returns:
        Tuple (coords array of shape (N,2), visibility array of shape (N,)).
    """
    kpts_arr = np.array(kpts).reshape(-1, 3)
    vi = kpts_arr[:, 2]
    coords = kpts_arr[:, :2]
    return coords, vi


def OKS(
    kpts1: list,
    kpts2: list,
    sigma: np.ndarray,
    area: float,
) -> float:
    """Compute the Object Keypoint Similarity score.

    Args:
        kpts1:  Ground-truth keypoints as flat list [x, y, v, ...].
        kpts2:  Predicted keypoints as flat list [x, y, v, ...].
        sigma:  Per-keypoint spread (standard deviation) array.
        area:   Object area used for normalisation.

    Returns:
        OKS score in [0, 1].

    Raises:
        ValueError: If keypoint arrays differ in shape, or all visibilities are 0.
    """
    coords1, vi1 = _edit_keypoints(kpts1)
    coords2, vi2 = _edit_keypoints(kpts2)

    if coords1.shape != coords2.shape:
        raise ValueError("Keypoint arrays are not the same size.")

    k = 2 * sigma
    d = np.linalg.norm(coords1 - coords2, ord=2, axis=1)
    v = np.ones(len(d))

    for part in range(len(d)):
        if vi1[part] == 0 or vi2[part] == 0:
            d[part] = 0
            v[part] = 0

    if np.sum(v) == 0:
        raise ValueError("All keypoint visibilities are zero.")

    oks_score = np.sum(
        [np.exp((-d[i] ** 2) / (2 * area * (k[i] ** 2))) * v[i] for i in range(len(d))]
    ) / np.sum(v)

    return oks_score


def CalculateOKS(true_pred: True_vs_Pred) -> float:
    """Compute OKS for a matched true/predicted keypoint pair.

    Args:
        true_pred: Dict with ``true``, ``pred``, and ``true_box_wh``.

    Returns:
        OKS score in [0, 1].
    """
    kpts_true: list = []
    kpts_false: list = []
    for key in ("apex", "desni_kost", "desni_vrh", "vrh", "lijevi_vrh", "lijevi_kost"):
        kpts_true += [true_pred["true"][key]["x"], true_pred["true"][key]["y"], 2]
        kpts_false += [true_pred["pred"][key]["x"], true_pred["pred"][key]["y"], 2]

    sigma = np.array([0.075, 0.075, 0.075, 0.075, 0.075, 0.075])
    area = 0.85 * true_pred["true_box_wh"][0] * true_pred["true_box_wh"][-1]

    return OKS(kpts_true, kpts_false, sigma, area)


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------

def draw_iou_visualization(
    img_path: str,
    lbl_path: str,
    pose_model,
    put_text: bool = True,
) -> np.ndarray:
    """Draw ground-truth (green) and predicted (red) bounding boxes on an image.

    Args:
        img_path:   Path to the input image.
        lbl_path:   Path to the YOLO label file.
        pose_model: Ultralytics YOLO pose model.
        put_text:   Whether to overlay the IoU score.

    Returns:
        Annotated image as a BGR numpy array.
    """
    img = cv2.imread(img_path)
    predikcija = pose_model([img_path], conf=CONF_THRESHOLD, iou=IOU_THRESHOLD, verbose=False)

    for pravokutnik in getTrueData(img_path, os.path.dirname(lbl_path)):
        pravokutnik_true = pravokutnik[1:5] * IMG_SIZE
        pravokutnik_pred = choose_the_best_match(
            pravokutnik_true,
            predikcija[0].boxes.xywh.cpu().numpy(),
        )

        if pravokutnik_pred is None:
            continue

        iou, true_points, pred_points = IoU(pravokutnik_true, pravokutnik_pred)

        for i in range(4):
            cv2.line(img, np.int32(true_points[i]), np.int32(true_points[(i + 1) % 4]), (0, 255, 0), 2)
        for i in range(4):
            cv2.line(img, np.int32(pred_points[i]), np.int32(pred_points[(i + 1) % 4]), (0, 0, 255), 2)

        if put_text:
            corner = np.int32(true_points[3])
            cv2.rectangle(img, corner + [-50, 30], corner, (0, 0, 0), -1)
            cv2.putText(
                img, str(np.round(iou, 2)),
                corner + [-40, 20],
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
            )
    return img


def draw_oks_visualization(
    img_path: str,
    lbl_path: str,
    pose_model,
    put_text: bool = True,
) -> np.ndarray:
    """Draw ground-truth (green) and predicted (red) skeleton lines on an image.

    Args:
        img_path:   Path to the input image.
        lbl_path:   Path to the YOLO label file.
        pose_model: Ultralytics YOLO pose model.
        put_text:   Whether to overlay the OKS score.

    Returns:
        Annotated image as a BGR numpy array.
    """
    img = cv2.imread(img_path)
    predikcija = pose_model([img_path], conf=CONF_THRESHOLD, iou=IOU_THRESHOLD, verbose=False)

    for keypoints_row in getTrueData(img_path, os.path.dirname(lbl_path)):
        w, h = IMG_SIZE, IMG_SIZE
        true_box_wh = keypoints_row[3:5] * IMG_SIZE
        true_kps = keypoints_row[5:]

        APEX, VRH, _ = getApexVrh(true_kps, w, h)
        DESNI_DONJI, DESNI_GORNJI, _ = getRightSide(true_kps, w, h)
        LIJEVI_GORNJI, LIJEVI_DONJI, _ = getLeftSide(true_kps, w, h)

        true_keypoints = Keypoints(
            apex=APEX, desni_kost=DESNI_DONJI, desni_vrh=DESNI_GORNJI,
            vrh=VRH, lijevi_vrh=LIJEVI_GORNJI, lijevi_kost=LIJEVI_DONJI,
        )
        true_pred = find_corresponding_image_distance(
            pose_model, img_path, 0.1, true_keypoints, true_box_wh, 0.1
        )

        kp_keys = list(true_keypoints.keys())

        def _draw_skeleton(tp_dict: dict, color: tuple) -> None:
            for i in range(len(kp_keys)):
                k1, k2 = kp_keys[i], kp_keys[(i + 1) % 6]
                p1 = np.array([tp_dict[k1]["x"], tp_dict[k1]["y"]]).astype(int)
                p2 = np.array([tp_dict[k2]["x"], tp_dict[k2]["y"]]).astype(int)
                cv2.line(img, p1, p2, color, 2)
            # Extra connecting lines
            for a, b in [
                ("lijevi_kost", "desni_vrh"),
                ("desni_kost", "lijevi_vrh"),
                ("desni_kost", "lijevi_kost"),
            ]:
                pa = np.array([tp_dict[a]["x"], tp_dict[a]["y"]]).astype(int)
                pb = np.array([tp_dict[b]["x"], tp_dict[b]["y"]]).astype(int)
                cv2.line(img, pa, pb, color, 2)

        _draw_skeleton(true_pred["true"], (0, 255, 0))
        _draw_skeleton(true_pred["pred"], (0, 0, 255))

        if put_text:
            corner = np.int32([true_pred["true"]["apex"]["x"] + 5, true_pred["true"]["apex"]["y"] + 5])
            cv2.rectangle(img, corner + [50, 30], corner, (0, 0, 0), -1)
            cv2.putText(
                img, str(np.round(CalculateOKS(true_pred), 2)),
                corner + [10, 20],
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
            )
    return img


def make_image_grid(
    image_list: list,
    rows: int = 3,
    cols: int = 3,
    size: Tuple[int, int] = (IMG_SIZE, IMG_SIZE),
) -> np.ndarray:
    """Arrange images in a regular grid canvas.

    Args:
        image_list: List of BGR numpy arrays (may contain None placeholders).
        rows:       Number of grid rows.
        cols:       Number of grid columns.
        size:       (width, height) of each cell in pixels.

    Returns:
        Grid canvas as a BGR numpy array.
    """
    canvas = np.zeros((rows * size[1], cols * size[0], 3), dtype=np.uint8)
    for idx, img in enumerate(image_list):
        if img is None:
            continue
        resized = cv2.resize(img, size)
        row = idx // cols
        col = idx % cols
        canvas[row * size[1]:(row + 1) * size[1], col * size[0]:(col + 1) * size[0]] = resized
    return canvas


def generate_iou_grid(base: str = BASE_PATH, pose_model=None) -> np.ndarray:
    """Generate a 3x3 grid of bounding-box visualisations categorised by IoU.

    Categories (rows): low (< OKS_LOW), medium (< OKS_HIGH), high (>= OKS_HIGH).

    Args:
        base:       Base path to the dataset split directories.
        pose_model: Ultralytics YOLO pose model.

    Returns:
        3x3 grid image as a BGR numpy array.
    """
    lower, middle, upper = [], [], []

    img_dir = os.path.join(base, "test", "images")
    lbl_dir = os.path.join(base, "test", "labels")
    img_paths = sorted(glob.glob(os.path.join(img_dir, "*.*")))

    for img_path in img_paths:
        predikcija = pose_model([img_path], conf=CONF_THRESHOLD, iou=IOU_THRESHOLD, verbose=False)
        true_data_instances = getTrueData(img_path, lbl_dir)

        for pravokutnik in true_data_instances:
            pravokutnik_true = pravokutnik[1:5] * IMG_SIZE
            pred_boxes = (
                predikcija[0].boxes.xywh.cpu().numpy()
                if predikcija[0].boxes is not None
                else np.array([])
            )
            pravokutnik_pred = choose_the_best_match(pravokutnik_true, pred_boxes)

            if pravokutnik_pred is None:
                continue

            iou, _, _ = IoU(pravokutnik_true, pravokutnik_pred)
            lbl_path = os.path.join(lbl_dir, f"{os.path.splitext(os.path.basename(img_path))[0]}.txt")

            if len(true_data_instances) != 1:
                continue

            if iou < OKS_LOW:
                lower.append([img_path, lbl_path])
            elif iou < OKS_HIGH:
                middle.append([img_path, lbl_path])
            else:
                upper.append([img_path, lbl_path])

    np.random.seed(102)
    selected: list = []
    for bucket in (lower, middle, upper):
        n = min(len(bucket), 3)
        idxs = np.random.choice(len(bucket), n, replace=False) if n > 0 else []
        selected.extend([bucket[i] for i in idxs])
    while len(selected) < 9:
        selected.append([None, None])

    drawn = [
        draw_iou_visualization(img, lbl, pose_model) if img is not None else None
        for img, lbl in selected
    ]
    return make_image_grid(drawn)


def generate_oks_grid(base: str = BASE_PATH, pose_model=None) -> Tuple[list, list, list]:
    """Collect image paths categorised by OKS value.

    Args:
        base:       Base path to the dataset.
        pose_model: Ultralytics YOLO pose model.

    Returns:
        Tuple (lower, middle, upper) lists of [img_path, lbl_path] pairs.
    """
    lower, middle, upper = [], [], []

    img_dir = os.path.join(base, "test", "images")
    lbl_dir = os.path.join(base, "test", "labels")
    img_paths = sorted(glob.glob(os.path.join(img_dir, "*.*")))

    for img_path in img_paths:
        predikcija = pose_model([img_path], conf=CONF_THRESHOLD, iou=IOU_THRESHOLD, verbose=False)

        for keypoints_row in getTrueData(img_path, lbl_dir):
            w, h = IMG_SIZE, IMG_SIZE
            true_box_wh = keypoints_row[3:5] * IMG_SIZE
            true_kps = keypoints_row[5:]

            APEX, VRH, _ = getApexVrh(true_kps, w, h)
            DESNI_DONJI, DESNI_GORNJI, _ = getRightSide(true_kps, w, h)
            LIJEVI_GORNJI, LIJEVI_DONJI, _ = getLeftSide(true_kps, w, h)

            true_keypoints = Keypoints(
                apex=APEX, desni_kost=DESNI_DONJI, desni_vrh=DESNI_GORNJI,
                vrh=VRH, lijevi_vrh=LIJEVI_GORNJI, lijevi_kost=LIJEVI_DONJI,
            )
            true_pred = find_corresponding_image_distance(
                pose_model, img_path, 0.1, true_keypoints, true_box_wh, 0.1
            )
            oks = CalculateOKS(true_pred)
            lbl_path = os.path.join(lbl_dir, f"{os.path.splitext(os.path.basename(img_path))[0]}.txt")

            if getTrueNumberOfImages(img_path, lbl_dir) != 1:
                continue

            if oks < OKS_LOW:
                lower.append([img_path, lbl_path])
            elif oks < OKS_HIGH:
                middle.append([img_path, lbl_path])
            else:
                upper.append([img_path, lbl_path])

    return lower, middle, upper


def show_grid(lower: list, middle: list, upper: list, pose_model) -> np.ndarray:
    """Build and return a 3x3 OKS skeleton visualisation grid.

    Args:
        lower:      Image paths with low OKS.
        middle:     Image paths with medium OKS.
        upper:      Image paths with high OKS.
        pose_model: Ultralytics YOLO pose model.

    Returns:
        3x3 grid as a BGR numpy array.
    """
    np.random.seed(102)
    selected: list = []
    for bucket in (lower, middle, upper):
        n = min(len(bucket), 3)
        idxs = np.random.choice(len(bucket), n, replace=False) if n > 0 else []
        selected.extend([bucket[i] for i in idxs])
    while len(selected) < 9:
        selected.append([None, None])

    drawn = [
        draw_oks_visualization(img, lbl, pose_model) if img is not None else None
        for img, lbl in selected
    ]
    return make_image_grid(drawn)


# ---------------------------------------------------------------------------
# Classification helpers
# ---------------------------------------------------------------------------

def determine_bigger_side(row: pd.Series) -> str:
    """Label which side has greater bone loss for a result row.

    Args:
        row: DataFrame row with ``true_desno_prop`` and ``true_lijevo_prop``.

    Returns:
        ``"desno"``, ``"lijevo"``, or ``"bez gubitka"``.
    """
    if row["true_desno_prop"] > row["true_lijevo_prop"]:
        return "desno"
    elif row["true_desno_prop"] < row["true_lijevo_prop"]:
        return "lijevo"
    elif row["true_desno_prop"] == 0 and row["true_lijevo_prop"] == 0:
        return "bez gubitka"
    else:
        return "desno"


def assign_mbl_category(row: pd.Series) -> str:
    """Map proportional bone loss to a clinical category.

    Args:
        row: DataFrame row with ``bigger``, ``true_desno_prop``, and
             ``true_lijevo_prop`` columns.

    Returns:
        One of ``"no_loss"``, ``"initial"``, ``"mild"``, ``"moderate"``,
        or ``"severe"``.
    """
    if row["bigger"] == "desno":
        proportion = row["true_desno_prop"]
    elif row["bigger"] == "lijevo":
        proportion = row["true_lijevo_prop"]
    else:
        proportion = row["true_desno_prop"]

    if proportion == 0:
        return "no_loss"
    elif 0 < proportion < 0.10:
        return "initial"
    elif 0.10 <= proportion < 0.25:
        return "mild"
    elif 0.25 <= proportion < 0.50:
        return "moderate"
    else:
        return "severe"


# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------

def run_hypothesis_test(
    data: list,
    threshold: float,
    alternative: str = 'less',
    alpha: float = 0.05,
) -> dict:
    """Shapiro-Wilk normality check, then t-test or Wilcoxon signed-rank test.

    Returns a dict with keys: test, stat, p_value, normality_p, significant, mean, std.
    """
    from scipy import stats
    import numpy as np
    data = [x for x in data if x is not None and not np.isnan(x)]
    shapiro_stat, shapiro_p = stats.shapiro(data)
    if shapiro_p >= alpha:
        stat, p = stats.ttest_1samp(data, threshold, alternative=alternative)
        test_used = 't-test'
    else:
        # Wilcoxon signed-rank: shift data by threshold, then test if median differs
        shifted = [x - threshold for x in data]
        stat, p = stats.wilcoxon(shifted, alternative=alternative)
        test_used = 'wilcoxon'
    return {
        'test': test_used,
        'stat': float(stat),
        'p_value': float(p),
        'normality_p': float(shapiro_p),
        'significant': bool(p < alpha),
        'mean': float(np.mean(data)),
        'std': float(np.std(data)),
        'n': len(data),
    }


def compute_regression_metrics(y_true: list, y_pred: list) -> dict:
    """Compute MAE, RMSE, and Pearson correlation between two lists."""
    from scipy import stats
    import numpy as np
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true, y_pred = y_true[mask], y_pred[mask]
    abs_diff = np.abs(y_true - y_pred)
    r, p = stats.pearsonr(y_true, y_pred)
    return {
        'mae': float(np.mean(abs_diff)),
        'rmse': float(np.sqrt(np.mean((y_true - y_pred) ** 2))),
        'pearson_r': float(r),
        'pearson_p': float(p),
        'n': int(len(y_true)),
        'abs_diffs': abs_diff.tolist(),
    }


def bootstrap_ci(data, stat_fn=None, n: int = 1000, alpha: float = 0.05):
    """Bootstrap confidence interval for a statistic.

    stat_fn: callable(array) -> scalar. Defaults to np.mean.
    Returns (lower, upper) for the (1-alpha)*100% CI.
    """
    import numpy as np
    data = np.array([x for x in data if x is not None and not np.isnan(x)], dtype=float)
    if stat_fn is None:
        stat_fn = np.mean
    rng = np.random.default_rng(42)
    boot_stats = [stat_fn(rng.choice(data, size=len(data), replace=True)) for _ in range(n)]
    lower = float(np.percentile(boot_stats, 100 * alpha / 2))
    upper = float(np.percentile(boot_stats, 100 * (1 - alpha / 2)))
    return lower, upper


def plot_keypoint_error_boxplot(df, value_col: str, group_col: str, title: str) -> None:
    """Box plot of keypoint errors grouped by keypoint name."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    keypoint_label_map = {
        'apex': 'Apex',
        'desni_kost': 'Distal bone',
        'desni_vrh': 'Distal tip',
        'vrh': 'Crown tip',
        'lijevi_vrh': 'Mesial tip',
        'lijevi_kost': 'Mesial bone',
    }
    keypoint_order = ['apex', 'desni_kost', 'desni_vrh', 'vrh', 'lijevi_vrh', 'lijevi_kost']
    order = [k for k in keypoint_order if k in df[group_col].unique()]
    display_labels = [keypoint_label_map.get(k, k) for k in order]
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x=group_col, y=value_col, order=order)
    plt.title(title)
    plt.xlabel('Keypoint')
    plt.ylabel('Scaled Euclidean error')
    plt.xticks(ticks=range(len(order)), labels=display_labels, rotation=30)
    plt.tight_layout()
    plt.show()


def plot_metric_histogram(values: list, xlabel: str, title: str, bins: int = 20) -> None:
    """Simple histogram with KDE overlay for a metric distribution."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    plt.figure(figsize=(8, 5))
    sns.histplot(values, bins=bins, kde=True)
    plt.xlabel(xlabel)
    plt.ylabel('Frekvencija')
    plt.title(title)
    plt.tight_layout()
    plt.show()
