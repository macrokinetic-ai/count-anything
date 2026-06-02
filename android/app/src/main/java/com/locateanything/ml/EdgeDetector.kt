package com.locateanything.ml

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.OrtSession.SessionOptions
import android.content.Context
import android.graphics.Bitmap
import android.os.Build
import android.util.Log
import java.io.Closeable
import java.nio.FloatBuffer

/**
 * On-device geometry proposal engine using ONNX Runtime + NNAPI.
 *
 * Loads yolov8n.onnx from assets and runs class-agnostic object detection
 * on the device NPU (via NNAPI on API >= 28) or CPU. Returns normalized
 * bounding box proposals for the server to verify semantically.
 *
 * Model: android/app/src/main/assets/edge_detector.onnx  (12 MB, YOLOv8n)
 * Output tensor shape: [1, 84, 8400]  (4 bbox coords + 80 class scores × 8400 anchors)
 */
class EdgeDetector(context: Context) : Closeable {

    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()
    private val session: OrtSession

    init {
        val modelBytes = context.assets.open(MODEL_FILENAME).use { it.readBytes() }
        val opts = SessionOptions().apply {
            setIntraOpNumThreads(2)
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
                try {
                    addNnapi()   // NNAPI EP → Exynos NPU on API 28+
                    Log.i(TAG, "NNAPI execution provider active")
                } catch (e: Exception) {
                    Log.w(TAG, "NNAPI unavailable, using CPU: ${e.message}")
                }
            }
        }
        session = env.createSession(modelBytes, opts)
        Log.i(TAG, "EdgeDetector ready — model: $MODEL_FILENAME  inputs=${session.inputNames}  outputs=${session.outputNames}")
    }

    /**
     * Run the edge model on [bitmap] (any resolution — scaled to 640×640 internally).
     * Returns geometry proposals sorted by score descending, coords normalized 0–1.
     */
    fun run(bitmap: Bitmap): List<EdgeProposal> {
        val scaled = Bitmap.createScaledBitmap(bitmap, INPUT_SIZE, INPUT_SIZE, true)
        val floatBuffer = bitmapToFloatBuffer(scaled)

        val inputTensor = OnnxTensor.createTensor(
            env,
            floatBuffer,
            longArrayOf(1, 3, INPUT_SIZE.toLong(), INPUT_SIZE.toLong()),
        )

        inputTensor.use {
            val results = session.run(mapOf(session.inputNames.first() to inputTensor))
            results.use {
                @Suppress("UNCHECKED_CAST")
                val output = (results.first().value as Array<Array<FloatArray>>)[0]
                return decodeAndNms(output)
            }
        }
    }

    override fun close() {
        session.close()
        env.close()
    }

    // ─── Private ─────────────────────────────────────────────────────────────

    /**
     * Convert [bitmap] (640×640) to a normalized NCHW float buffer.
     * YOLOv8 expects pixel values in [0, 1], channel order RGB, layout [1,3,H,W].
     */
    private fun bitmapToFloatBuffer(bitmap: Bitmap): FloatBuffer {
        val pixels = IntArray(INPUT_SIZE * INPUT_SIZE)
        bitmap.getPixels(pixels, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE)

        val buf = FloatBuffer.allocate(3 * INPUT_SIZE * INPUT_SIZE)
        val rOffset = 0
        val gOffset = INPUT_SIZE * INPUT_SIZE
        val bOffset = 2 * INPUT_SIZE * INPUT_SIZE

        for (i in pixels.indices) {
            val px = pixels[i]
            buf.put(rOffset + i, ((px shr 16) and 0xFF) / 255f)
            buf.put(gOffset + i, ((px shr 8)  and 0xFF) / 255f)
            buf.put(bOffset + i, (px and 0xFF)           / 255f)
        }
        return buf
    }

    /**
     * Decode raw [84 × 8400] YOLOv8 output into [EdgeProposal] list.
     * Rows 0-3: cx, cy, w, h (in 640-px space).
     * Rows 4-83: 80 COCO class scores.
     * Score = max(class scores) — class-agnostic saliency for geometry proposals.
     */
    private fun decodeAndNms(output: Array<FloatArray>): List<EdgeProposal> {
        data class Raw(val x1: Float, val y1: Float, val x2: Float, val y2: Float, val score: Float)

        val numAnchors = output[0].size
        val scale = INPUT_SIZE.toFloat()
        val raw = ArrayList<Raw>(numAnchors / 20)

        for (i in 0 until numAnchors) {
            var maxScore = 0f
            for (c in 4 until OUTPUT_CHANNELS) {
                if (output[c][i] > maxScore) maxScore = output[c][i]
            }
            if (maxScore < SCORE_THRESHOLD) continue

            val cx = output[0][i]; val cy = output[1][i]
            val w  = output[2][i]; val h  = output[3][i]
            raw.add(Raw(
                x1 = ((cx - w / 2f) / scale).coerceIn(0f, 1f),
                y1 = ((cy - h / 2f) / scale).coerceIn(0f, 1f),
                x2 = ((cx + w / 2f) / scale).coerceIn(0f, 1f),
                y2 = ((cy + h / 2f) / scale).coerceIn(0f, 1f),
                score = maxScore,
            ))
        }

        raw.sortByDescending { it.score }

        val keep = BooleanArray(raw.size) { true }
        for (i in raw.indices) {
            if (!keep[i]) continue
            for (j in i + 1 until raw.size) {
                if (!keep[j]) continue
                val a = raw[i]; val b = raw[j]
                val ix1 = maxOf(a.x1, b.x1); val iy1 = maxOf(a.y1, b.y1)
                val ix2 = minOf(a.x2, b.x2); val iy2 = minOf(a.y2, b.y2)
                val inter = maxOf(0f, ix2 - ix1) * maxOf(0f, iy2 - iy1)
                if (inter > 0f) {
                    val union = (a.x2-a.x1)*(a.y2-a.y1) + (b.x2-b.x1)*(b.y2-b.y1) - inter
                    if (inter / union >= NMS_IOU_THRESHOLD) keep[j] = false
                }
            }
        }

        return raw.indices
            .filter { keep[it] }
            .take(MAX_PROPOSALS)
            .map { raw[it].let { r -> EdgeProposal(r.x1, r.y1, r.x2, r.y2, r.score) } }
    }

    companion object {
        private const val TAG = "EdgeDetector"
        private const val MODEL_FILENAME = "edge_detector.onnx"
        private const val INPUT_SIZE = 640
        private const val OUTPUT_CHANNELS = 84
        private const val SCORE_THRESHOLD = 0.05f
        private const val NMS_IOU_THRESHOLD = 0.45f
        private const val MAX_PROPOSALS = 100
    }
}
