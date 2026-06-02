package com.locateanything.viewmodel

import android.app.Application
import android.net.Uri
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.locateanything.data.DetectionRepository
import com.locateanything.data.model.DetectedBox
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch

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

    private fun runDetection(imageUri: Uri, prompt: String, threshold: Float) {
        viewModelScope.launch {
            _uiState.value = DetectionUiState.Loading
            runCatching {
                repository.detect(
                    context = getApplication(),
                    imageUri = imageUri,
                    prompt = prompt,
                    threshold = threshold,
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
