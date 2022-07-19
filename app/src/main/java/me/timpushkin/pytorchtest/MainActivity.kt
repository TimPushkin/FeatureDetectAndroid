package me.timpushkin.pytorchtest

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.os.Bundle
import android.os.SystemClock
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.material.Button
import androidx.compose.material.MaterialTheme
import androidx.compose.material.Surface
import androidx.compose.material.Text
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.asImageBitmap
import me.timpushkin.pytorchtest.ui.theme.PyTorchTestTheme
import java.io.File

private const val MODEL_ASSET = "superpoint.ptl"
private const val DECODER_ASSET = "superpoint_decoder.ptl"
private const val IMAGE_ASSET = "image.png"

class MainActivity : ComponentActivity() {
    private lateinit var mSuperPointNet: SuperPoint

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        mSuperPointNet = SuperPoint(assetFilePath(MODEL_ASSET), assetFilePath(DECODER_ASSET))
        val image = BitmapFactory.decodeStream(assets.open(IMAGE_ASSET))

        setContent {
            PyTorchTestTheme {
                Surface(
                    modifier = Modifier.fillMaxSize(),
                    color = MaterialTheme.colors.background
                ) {
                    Column(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalAlignment = Alignment.CenterHorizontally
                    ) {
                        var result by rememberSaveable { mutableStateOf<Pair<String, Long>?>(null) }

                        Image(
                            bitmap = image.asImageBitmap(),
                            contentDescription = "Image"
                        )

                        result?.let { (name, timeNanos) ->
                            Column(horizontalAlignment = Alignment.CenterHorizontally) {
                                Text(text = name)
                                Text(text = "${timeNanos * 1e-9} sec")
                            }
                        }

                        Button(onClick = { result = runTest(image) }) {
                            Text(text = "Run")
                        }
                    }
                }
            }
        }
    }

    private fun assetFilePath(assetName: String): String {
        val file = File(filesDir, assetName)
        if (!file.exists()) file.outputStream().use { os -> assets.open(assetName).copyTo(os) }
        return file.absolutePath
    }

    private fun runTest(image: Bitmap): Pair<String, Long> {
        val startTimeNanos = SystemClock.elapsedRealtimeNanos()
        val (keyPoints, descriptors) = mSuperPointNet.forward(image)
        val calcTimeNanos = SystemClock.elapsedRealtimeNanos() - startTimeNanos
        return "Key points tensor shape: " +
                "${keyPoints.size}x${keyPoints[0].size}x${keyPoints[0][0].size}.\n" +
                "Descriptors tensor shape: " +
                "${descriptors.size}x${descriptors[0].size}x${descriptors[0][0].size}" to
                calcTimeNanos
    }
}
