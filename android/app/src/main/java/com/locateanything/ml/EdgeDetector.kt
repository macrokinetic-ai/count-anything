package com.locateanything.ml

import android.content.Context
import android.graphics.Bitmap
import android.os.Build
import android.util.Log
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.nnapi.NnApiDelegate
import java.io.Closeable
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * On-device geometry proposal engine.
 *
 * Loads a YOLOv8n INT8 TFLite model from app assets and runs inference on
 * the device NPU (NNAPI delegate, API >= 28) or CPU (fallback). Returns
 * class-agnostic bounding box proposals — the model detects "any salient
 * object" and the server then verifies each box semantically with YOLO-World.
 *
 * Model asset: app/src/main/assets/edge_detector.tflite
 * If the asset is absent, [run] will throw and the ViewModel falls back to
 * server_only mode automatically.
 */
class EdgeDetector(context: Context) : Closeable {

    private val interpreter: Interpreter
    private val nnApiDelegate: NnApiDelegate?

    init {
        val pair = buildInterpreter(context)
        interpreter = pair.first
        nnApiDelegate = pair.second

        val shape = interpreter.getOutputTensor(0).shape()
        check(shape.size == 3 && shape[1] == OUTPUT_CHANNELS) {
            "Unexpected TFLite output shape ${shape.toList()}; expected [1, 84, N]. " +
            "Ensure model is YOLOv8n exported with format=tflite int8=True."
        }
        Log.i(TAG, "EdgeDetector ready — anchors=${shape[2]}  nnapi=${nnApiDelegate != null}")
    }

    /**
     * Run inference on [bitmap] (any size — scaled to 640×640 internally).
     * Returns geometry proposals sorted by score descending, coords normalized 0–1.
     */
    fun run(bitmap: Bitmap): List<EdgeProposal> {
        val scaled = Bitmap.createScaledBitmap(bitmap, INPUT_SIZE, INPUT_SIZE, true)
        val inputBuffer = bitmapToByteBuffer(scaled)

        val numAnchors = interpreter.getOutputTensor(0).shape()[2]
        val rawOutput = Array(1) { Array(OUTPUT_CHANNELS) { FloatArray(numAnchors) } }
        interpreter.run(inputBuffer, rawOutput)

        return decodeAndNms(rawOutput[0], numAnchors)
    }

    override fun close() {
        interpreter.close()
        nnApiDelegate?.close()
    }

    // ─── Private ─────────────────────────────────────────────────────────────

    private fun buildInterpreter(context: Context): Pair<Interpreter, NnApiDelegate?> {
        val buffer = context.assets.openFd(MODEL_FILENAME).use { fd ->
            ByteBuffer.allocateDirect(fd.length.toInt()).also { buf ->
                buf.order(ByteOrder.nativeOrder())
                fd.createInputStream().use { it.read(buf.array()) }
                buf.rewind()
            }
        }
        val options = Interpreter.Options().apply { numThreads = 2 }

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
            try {
                val nnApiOptions = NnApiDelegate.Options().apply {
                    setAllowFp16(true)
                    setExecutionPreference(NnApiDelegate.Options.EXECUTION_PREFERENCE_SUSTAINED_SPEED)
                    // null acceleratorName → NNAPI auto-selects best (NPU on Exynos 2500)
                }
                val delegate = NnApiDelegate(nnApiOptions)
                options.addDelegate(delegate)
                Log.i(TAG, "NNAPI delegate active")
                return Pair(Interpreter(buffer, options), delegate)
            } catch (e: Exception) {
                Log.w(TAG, "NNAPI delegate unavailable, falling back to CPU: ${e.message}")
            }
        }

        Log.i(TAG, "CPU interpreter (API ${Build.VERSION.SDK_INT})")
        return Pair(Interpreter(buffer, options), null)
    }

    private fun bitmapToByteBuffer(bitmap: Bitmap): ByteBuffer {
        val buf = ByteBuffer
            .allocateDirect(1 * INPUT_SIZE * INPUT_SIZE * 3)
            .apply { order(ByteOrder.nativeOrder()) }
        val pixels = IntArray(INPUT_SIZE * INPUT_SIZE)
        bitmap.getPixels(pixels, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE)
        for (px in pixels) {
            buf.put(((px shr 16) and 0xFF).toByte())
            buf.put(((px shr 8)  and 0xFF).toByte())
            buf.put((px and 0xFF).toByte())
        }
        buf.rewind()
        return buf
    }

    /**
     * Decode raw [84 × numAnchors] YOLOv8 output into [EdgeProposal] list.
     * Channels 0-3: cx, cy, w, h (in 640-px space).
     * Channels 4-83: 80 COCO class probabilities.
     *
     * We use max(class_probs) as class-agnostic saliency — we don't label the
     * object here, just flag that "something is present" for the server to verify.
     */
    private fun decodeAndNms(output: Array<FloatArray>, numAnchors: Int): List<EdgeProposal> {
        data class Raw(val x1: Float, val y1: Float, val x2: Float, val y2: Float, val score: Float)

        val raw = ArrayList<Raw>(numAnchors / 10)
        val scale = INPUT_SIZE.toFloat()

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

        // Greedy NMS
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
                    val iou = inter / ((a.x2-a.x1)*(a.y2-a.y1) + (b.x2-b.x1)*(b.y2-b.y1) - inter)
                    if (iou >= NMS_IOU_THRESHOLD) keep[j] = false
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
        private const val MODEL_FILENAME = "edge_detector.tflite"
        private const val INPUT_SIZE = 640
        private const val OUTPUT_CHANNELS = 84
        private const val SCORE_THRESHOLD = 0.05f
        private const val NMS_IOU_THRESHOLD = 0.45f
        private const val MAX_PROPOSALS = 100
    }
}
