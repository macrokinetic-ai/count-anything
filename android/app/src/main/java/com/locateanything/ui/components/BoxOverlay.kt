package com.locateanything.ui.components

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.nativeCanvas
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.layout.onSizeChanged
import androidx.compose.ui.unit.IntSize
import com.locateanything.data.model.DetectedBox

private val BOX_COLOR = Color(0xFF00E5FF)  // cyan

@Composable
fun BoxOverlay(
    boxes: List<DetectedBox>,
    onRemoveBox: (Int) -> Unit,
    modifier: Modifier = Modifier,
) {
    var composableSize by remember { mutableStateOf(IntSize.Zero) }

    Box(
        modifier = modifier
            .onSizeChanged { composableSize = it }
            .pointerInput(boxes) {
                detectTapGestures { tapOffset ->
                    if (composableSize == IntSize.Zero) return@detectTapGestures
                    val xNorm = tapOffset.x / composableSize.width
                    val yNorm = tapOffset.y / composableSize.height
                    val hit = boxes.firstOrNull { b ->
                        xNorm in b.x1..b.x2 && yNorm in b.y1..b.y2
                    }
                    hit?.let { onRemoveBox(it.id) }
                }
            },
    ) {
        Canvas(modifier = Modifier.fillMaxSize()) {
            if (composableSize == IntSize.Zero) return@Canvas
            val w = composableSize.width.toFloat()
            val h = composableSize.height.toFloat()

            val labelPaint = android.graphics.Paint().apply {
                color = android.graphics.Color.CYAN
                textSize = 28f
                isFakeBoldText = true
            }

            boxes.forEach { box ->
                val left   = box.x1 * w
                val top    = box.y1 * h
                val right  = box.x2 * w
                val bottom = box.y2 * h

                drawRect(
                    color = BOX_COLOR,
                    topLeft = Offset(left, top),
                    size = Size(right - left, bottom - top),
                    style = Stroke(width = 3f),
                )

                // Score label in top-left corner of the box
                drawContext.canvas.nativeCanvas.drawText(
                    "%.0f%%".format(box.score * 100),
                    left + 4f,
                    top + 30f,
                    labelPaint,
                )
            }
        }
    }
}
