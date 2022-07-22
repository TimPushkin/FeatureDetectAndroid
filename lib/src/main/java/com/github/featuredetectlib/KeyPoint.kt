package com.github.featuredetectlib

/**
 * Keypoint which represents an image feature.
 */
data class KeyPoint(
    /**
     * x coordinate of the keypoint center.
     */
    val x: Float,
    /**
     * y coordinate of the keypoint center.
     */
    val y: Float,
    /**
     * Keypoint strength.
     *
     * The higher the strength the "better" the keypoint.
     */
    val strength: Float,
    /**
     * Diameter of the meaningful keypoint neighborhood.
     */
    val size: Float? = null,
    /**
     * Orientation of the feature represented by the keypoint.
     *
     * It is in [0, 360) degrees clockwise relative to image coordinates.
     */
    val angle: Float? = null
)
