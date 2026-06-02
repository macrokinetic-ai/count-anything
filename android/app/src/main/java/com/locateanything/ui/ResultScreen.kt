package com.locateanything.ui

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Slider
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.lifecycle.viewmodel.compose.viewModel
import coil3.compose.AsyncImage
import com.locateanything.ui.components.BoxOverlay
import com.locateanything.viewmodel.DetectionUiState
import com.locateanything.viewmodel.DetectionViewModel

@Composable
fun ResultScreen(
    detectionViewModel: DetectionViewModel = viewModel(),
    onNewPhoto: () -> Unit,
    modifier: Modifier = Modifier,
) {
    val uiState by detectionViewModel.uiState.collectAsState()

    when (val state = uiState) {
        is DetectionUiState.Loading -> {
            Box(modifier = modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
                Column(horizontalAlignment = Alignment.CenterHorizontally) {
                    CircularProgressIndicator()
                    Spacer(modifier = Modifier.height(16.dp))
                    Text("Detecting books…")
                }
            }
        }

        is DetectionUiState.Result -> {
            val aspectRatio = if (state.imageWidth > 0 && state.imageHeight > 0)
                state.imageWidth.toFloat() / state.imageHeight.toFloat()
            else
                4f / 3f

            var sliderValue by remember(state.threshold) { mutableFloatStateOf(state.threshold) }

            Column(
                modifier = modifier
                    .fillMaxSize()
                    .verticalScroll(rememberScrollState()),
            ) {
                // Count header
                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(horizontal = 16.dp, vertical = 12.dp),
                    horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment = Alignment.CenterVertically,
                ) {
                    Text(
                        text = "${state.boxes.size} books found",
                        style = MaterialTheme.typography.headlineMedium,
                        fontWeight = FontWeight.Bold,
                    )
                    Text(
                        text = "${state.inferenceMs} ms",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                }

                // Image + box overlay
                Box(
                    modifier = Modifier
                        .fillMaxWidth()
                        .aspectRatio(aspectRatio),
                ) {
                    AsyncImage(
                        model = state.imageUri,
                        contentDescription = "Bookshelf photo",
                        contentScale = ContentScale.FillBounds,  // matches our aspect ratio container
                        modifier = Modifier.fillMaxSize(),
                    )
                    BoxOverlay(
                        boxes = state.boxes,
                        onRemoveBox = detectionViewModel::removeBox,
                        modifier = Modifier.fillMaxSize(),
                    )
                }

                // Hint
                Text(
                    text = "Tap a box to remove it · prompt: \"${state.prompt}\"",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    modifier = Modifier.padding(horizontal = 16.dp, vertical = 4.dp),
                )

                // Confidence threshold slider
                Column(modifier = Modifier.padding(horizontal = 16.dp)) {
                    Text(
                        text = "Confidence threshold: ${(sliderValue * 100).toInt()}%",
                        style = MaterialTheme.typography.bodyMedium,
                    )
                    Slider(
                        value = sliderValue,
                        onValueChange = { sliderValue = it },
                        onValueChangeFinished = {
                            detectionViewModel.redetectWithThreshold(sliderValue)
                        },
                        valueRange = 0.05f..0.90f,
                        steps = 16,
                    )
                }

                Spacer(modifier = Modifier.height(8.dp))

                // Model attribution
                Text(
                    text = "Model: ${state.model}",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    modifier = Modifier.padding(horizontal = 16.dp),
                )

                Spacer(modifier = Modifier.height(16.dp))

                Button(
                    onClick = {
                        detectionViewModel.reset()
                        onNewPhoto()
                    },
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(horizontal = 16.dp),
                ) {
                    Text("New Photo")
                }

                Spacer(modifier = Modifier.height(24.dp))
            }
        }

        is DetectionUiState.Error -> {
            Box(modifier = modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
                Column(
                    horizontalAlignment = Alignment.CenterHorizontally,
                    modifier = Modifier.padding(32.dp),
                ) {
                    Text(
                        text = "Detection failed",
                        style = MaterialTheme.typography.headlineSmall,
                        color = MaterialTheme.colorScheme.error,
                    )
                    Spacer(modifier = Modifier.height(8.dp))
                    Text(
                        text = state.message,
                        style = MaterialTheme.typography.bodyMedium,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                    Spacer(modifier = Modifier.height(24.dp))
                    OutlinedButton(onClick = onNewPhoto) {
                        Text("Try Again")
                    }
                }
            }
        }

        else -> {
            // Idle — should not be visible (navigator pops back to HomeScreen)
        }
    }
}
