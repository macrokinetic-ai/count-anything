package com.locateanything.ml

/**
 * A geometry proposal produced by the on-device YOLOv8n TFLite model.
 * Coordinates are normalized 0.0–1.0 relative to the input image dimensions.
 * The [score] is the maximum class confidence across all 80 COCO classes —
 * used purely as a geometric saliency measure, not a semantic label.
 */
data class EdgeProposal(
    val x1: Float,
    val y1: Float,
    val x2: Float,
    val y2: Float,
    val score: Float,
)
