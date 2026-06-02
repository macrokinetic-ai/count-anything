package com.locateanything.data.model

import com.google.gson.annotations.SerializedName

data class DetectedBox(
    val id: Int,
    val x1: Float,
    val y1: Float,
    val x2: Float,
    val y2: Float,
    val score: Float,
)

data class DetectionResponse(
    val count: Int,
    @SerializedName("image_width")  val imageWidth: Int,
    @SerializedName("image_height") val imageHeight: Int,
    val model: String,
    @SerializedName("version_id")   val versionId: String,
    @SerializedName("inference_ms") val inferenceMs: Long,
    val boxes: List<DetectedBox>,
)
