package me.timpushkin.pytorchtest

import android.graphics.Bitmap
import org.pytorch.IValue
import org.pytorch.LiteModuleLoader
import org.pytorch.torchvision.TensorImageUtils

class ModelWrapper(modelPath: String) {
    private val module = LiteModuleLoader.load(modelPath)

    fun calcScores(image: Bitmap): List<Float> {
        val input = TensorImageUtils.bitmapToFloat32Tensor(
            image,
            TensorImageUtils.TORCHVISION_NORM_MEAN_RGB,
            TensorImageUtils.TORCHVISION_NORM_STD_RGB
        )
        val output = module.forward(IValue.from(input)).toTensor()

        val result = output.dataAsFloatArray
        return result.asList()
    }

    fun scoresToName(scores: List<Float>): String {
        val maxScoreIndex = scores.indices.maxByOrNull { scores[it] }
        return maxScoreIndex?.let { IMAGENET_CLASSES[it] } ?: "None"
    }
}
