package com.locateanything

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Scaffold
import androidx.compose.ui.Modifier
import androidx.lifecycle.viewmodel.compose.viewModel
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.rememberNavController
import com.locateanything.ui.HomeScreen
import com.locateanything.ui.ResultScreen
import com.locateanything.viewmodel.DetectionViewModel

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            val navController = rememberNavController()
            val detectionViewModel: DetectionViewModel = viewModel()

            Scaffold(modifier = Modifier.fillMaxSize()) { innerPadding ->
                NavHost(
                    navController = navController,
                    startDestination = "home",
                    modifier = Modifier.padding(innerPadding),
                ) {
                    composable("home") {
                        HomeScreen(
                            onImageSelected = { uri, prompt ->
                                detectionViewModel.detect(uri, prompt)
                                navController.navigate("result")
                            },
                        )
                    }
                    composable("result") {
                        ResultScreen(
                            detectionViewModel = detectionViewModel,
                            onNewPhoto = {
                                navController.popBackStack("home", inclusive = false)
                            },
                        )
                    }
                }
            }
        }
    }
}
