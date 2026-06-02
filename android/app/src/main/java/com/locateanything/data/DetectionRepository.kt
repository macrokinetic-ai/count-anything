package com.locateanything.data

import android.content.Context
import android.net.Uri
import com.google.gson.Gson
import com.locateanything.data.model.DetectionResponse
import com.locateanything.ml.EdgeProposal
import okhttp3.MediaType.Companion.toMediaTypeOrNull
import okhttp3.MultipartBody
import okhttp3.OkHttpClient
import okhttp3.RequestBody.Companion.toRequestBody
import retrofit2.Retrofit
import retrofit2.converter.gson.GsonConverterFactory
import java.util.concurrent.TimeUnit

class DetectionRepository {

    companion object {
        // Emulator → host machine localhost. For a physical device:
        //   1. Run: adb reverse tcp:8000 tcp:8000   (still uses 10.0.2.2 equivalent via reverse)
        //   OR
        //   2. Change to your machine's LAN IP, e.g. "http://192.168.1.42:8000/"
        const val BASE_URL = "http://10.80.1.209:8000/"
    }

    private val api: DetectionApi by lazy {
        val client = OkHttpClient.Builder()
            .connectTimeout(10, TimeUnit.SECONDS)
            .readTimeout(60, TimeUnit.SECONDS)   // inference can be slow on CPU
            .writeTimeout(30, TimeUnit.SECONDS)
            .build()

        Retrofit.Builder()
            .baseUrl(BASE_URL)
            .client(client)
            .addConverterFactory(GsonConverterFactory.create())
            .build()
            .create(DetectionApi::class.java)
    }

    suspend fun detect(
        context: Context,
        imageUri: Uri,
        prompt: String,
        threshold: Float,
        maxBoxes: Int = 60,
        proposals: List<EdgeProposal>? = null,
    ): DetectionResponse {
        val bytes = context.contentResolver.openInputStream(imageUri)!!.use { it.readBytes() }
        val imagePart = MultipartBody.Part.createFormData(
            "image",
            "photo.jpg",
            bytes.toRequestBody("image/jpeg".toMediaTypeOrNull()),
        )
        val promptBody   = prompt.toRequestBody("text/plain".toMediaTypeOrNull())
        val threshBody   = threshold.toString().toRequestBody("text/plain".toMediaTypeOrNull())
        val maxBoxesBody = maxBoxes.toString().toRequestBody("text/plain".toMediaTypeOrNull())

        // Serialize proposals to JSON only if non-empty; null = omit from multipart (server_only)
        val proposedBoxesBody = proposals
            ?.takeIf { it.isNotEmpty() }
            ?.let { Gson().toJson(it).toRequestBody("application/json".toMediaTypeOrNull()) }

        return api.detect(imagePart, promptBody, threshBody, maxBoxesBody, proposedBoxesBody)
    }
}
