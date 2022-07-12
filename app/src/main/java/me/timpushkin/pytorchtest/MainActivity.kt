package me.timpushkin.pytorchtest

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.material.MaterialTheme
import androidx.compose.material.Surface
import androidx.compose.material.Text
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.asImageBitmap
import me.timpushkin.pytorchtest.ui.theme.PyTorchTestTheme
import org.pytorch.IValue
import org.pytorch.LiteModuleLoader
import org.pytorch.torchvision.TensorImageUtils
import java.io.File

private const val MODEL_ASSET = "model.ptl"
private const val IMAGE_ASSET = "image.png"

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        val image = BitmapFactory.decodeStream(assets.open(IMAGE_ASSET))
        val scores = calcScores(image)
        val name = scoresToName(scores)

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
                        Image(
                            bitmap = image.asImageBitmap(),
                            contentDescription = "Image"
                        )
                        Text(text = name)
                    }
                }
            }
        }
    }

    private fun calcScores(image: Bitmap): List<Float> {
        val module = LiteModuleLoader.load(assetFilePath(MODEL_ASSET))

        val input = TensorImageUtils.bitmapToFloat32Tensor(
            image,
            TensorImageUtils.TORCHVISION_NORM_MEAN_RGB,
            TensorImageUtils.TORCHVISION_NORM_STD_RGB
        )
        val output = module.forward(IValue.from(input)).toTensor()

        val result = output.dataAsFloatArray
        return result.asList()
    }

    private fun scoresToName(scores: List<Float>): String {
        val maxScoreIndex = scores.indices.maxByOrNull { scores[it] }
        return maxScoreIndex?.let { IMAGENET_CLASSES[it] } ?: "None"
    }

    private fun assetFilePath(assetName: String): String {
        val file = File(filesDir, assetName)
        if (!file.exists()) file.outputStream().use { os -> assets.open(assetName).copyTo(os) }
        return file.absolutePath
    }
}
