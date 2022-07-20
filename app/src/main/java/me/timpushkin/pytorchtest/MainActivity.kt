package me.timpushkin.pytorchtest

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.os.Bundle
import android.os.SystemClock
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.size
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
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.PointMode
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.platform.LocalDensity
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
                        var keyPoints by rememberSaveable { mutableStateOf<List<Offset>?>(null) }
                        var calcTimeNanos by rememberSaveable { mutableStateOf<Long?>(null) }

                        Box {
                            Image(
                                bitmap = image.asImageBitmap(),
                                contentDescription = "Image"
                            )

                            val (width, height) = with(LocalDensity.current) {
                                image.width.toDp() to image.height.toDp()
                            }

                            keyPoints?.let { points ->
                                Canvas(modifier = Modifier.size(width, height)) {
                                    drawPoints(
                                        points = points,
                                        pointMode = PointMode.Points,
                                        color = Color.Green,
                                        strokeWidth = 10f
                                    )
                                }
                            }
                        }

                        calcTimeNanos?.let { timeNanos ->
                            Column(horizontalAlignment = Alignment.CenterHorizontally) {
                                Text(text = "${timeNanos * 1e-9} sec")
                            }
                        }

                        Button(
                            onClick = {
                                val (points, time) = calcKeyPoints(image)
                                keyPoints = points
                                calcTimeNanos = time
                            }
                        ) {
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

    private fun calcKeyPoints(image: Bitmap): Pair<List<Offset>, Long> {
        val startTimeNanos = SystemClock.elapsedRealtimeNanos()
        val (keyPoints, _) = mSuperPointNet.forward(image)
        val calcTimeNanos = SystemClock.elapsedRealtimeNanos() - startTimeNanos
        return keyPoints.map { (x, y, _) -> Offset(x, y) } to calcTimeNanos
    }
}
