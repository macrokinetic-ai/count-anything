package com.locateanything.viewmodel

import android.app.Application
import android.graphics.BitmapFactory
import android.net.Uri
import android.util.Log
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.locateanything.data.DetectionRepository
import com.locateanything.data.model.DetectedBox
import com.locateanything.ml.EdgeDetector
import com.locateanything.ml.EdgeProposal
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

private const val TAG = "DetectionViewModel"

sealed class DetectionUiState {
    object Idle : DetectionUiState()
    object Loading : DetectionUiState()
    data class Result(
        val imageUri: Uri,
        val imageWidth: Int,
        val imageHeight: Int,
        val boxes: List<DetectedBox>,
        val model: String,
        val versionId: String,
        val inferenceMs: Long,
        val prompt: String,
        val threshold: Float,
        val mode: String = "server_only",
    ) : DetectionUiState()
    data class Error(val message: String) : DetectionUiState()
}

class DetectionViewModel(application: Application) : AndroidViewModel(application) {

    private val repository = DetectionRepository()

    // EdgeDetector is initialized lazily so a missing .tflite asset doesn't crash
    // the ViewModel on startup — it only throws when run() is first called.
    private val edgeDetector: EdgeDetector? by lazy {
        runCatching { EdgeDetector(getApplication()) }
            .onFailure { Log.w(TAG, "EdgeDetector unavailable: ${it.message}") }
            .getOrNull()
    }

    private val _uiState = MutableStateFlow<DetectionUiState>(DetectionUiState.Idle)
    val uiState: StateFlow<DetectionUiState> = _uiState.asStateFlow()

    private var lastImageUri: Uri? = null
    private var lastPrompt: String = "book spine"

    fun detect(imageUri: Uri, prompt: String = "book spine", threshold: Float = 0.30f) {
        lastImageUri = imageUri
        lastPrompt = prompt
        runDetection(imageUri, prompt, threshold)
    }

    fun redetectWithThreshold(threshold: Float) {
        val uri = lastImageUri ?: return
        runDetection(uri, lastPrompt, threshold)
    }

    fun removeBox(id: Int) {
        val current = _uiState.value as? DetectionUiState.Result ?: return
        _uiState.value = current.copy(boxes = current.boxes.filter { it.id != id })
    }

    fun reset() {
        _uiState.value = DetectionUiState.Idle
    }

    override fun onCleared() {
        super.onCleared()
        edgeDetector?.close()
    }

    private fun runDetection(imageUri: Uri, prompt: String, threshold: Float) {
        viewModelScope.launch {
            _uiState.value = DetectionUiState.Loading

            // Phase A: on-device geometry proposals (NPU/CPU, isolated failure)
            val proposals: List<EdgeProposal>? = withContext(Dispatchers.Default) {
                runCatching {
                    val detector = edgeDetector ?: return@runCatching null
                    val context = getApplication<Application>()
                    val bitmap = context.contentResolver.openInputStream(imageUri)?.use {
                        BitmapFactory.decodeStream(it)
                    } ?: return@runCatching null
                    val start = System.currentTimeMillis()
                    val result = detector.run(bitmap)
                    Log.i(TAG, "EdgeDetector: ${result.size} proposals in ${System.currentTimeMillis()-start}ms")
                    result.takeIf { it.isNotEmpty() }
                }.onFailure { e ->
                    Log.w(TAG, "EdgeDetector failed, server_only fallback: ${e.message}")
                }.getOrNull()
            }

            // Phase B: server call — with proposals (hybrid) or without (server_only)
            runCatching {
                repository.detect(
                    context = getApplication(),
                    imageUri = imageUri,
                    prompt = prompt,
                    threshold = threshold,
                    proposals = proposals,
                )
            }.onSuccess { response ->
                _uiState.value = DetectionUiState.Result(
                    imageUri = imageUri,
                    imageWidth = response.imageWidth,
                    imageHeight = response.imageHeight,
                    boxes = response.boxes,
                    model = response.model,
                    versionId = response.versionId,
                    inferenceMs = response.inferenceMs,
                    prompt = prompt,
                    threshold = threshold,
                    mode = response.mode,
                )
            }.onFailure { e ->
                _uiState.value = DetectionUiState.Error(
                    e.message ?: "Connection failed. Is the backend running?"
                )
            }
        }
    }
}
