package me.timpushkin.pytorchtest

import android.graphics.Bitmap
import android.graphics.Color
import android.os.SystemClock
import android.util.Log
import org.pytorch.IValue
import org.pytorch.LiteModuleLoader
import org.pytorch.Tensor
import kotlin.math.pow

private const val TAG = "SuperPointNet"

private const val OUTPUT_SIZE = 2

class SuperPointNet(modelPath: String) {
    private val module = LiteModuleLoader.load(modelPath)

    fun forward(image: Bitmap): Pair<List<List<List<Float>>>, List<List<List<Float>>>> {
        val height = image.height.toLong()
        val width = image.width.toLong()

        val input = Tensor.fromBlob(image.toGrayscaleArray(), longArrayOf(1, 1, height, width))

        val startTimeNanos = SystemClock.elapsedRealtimeNanos()
        val rawOutput = module.forward(IValue.from(input))
        val calcTimeNanos = SystemClock.elapsedRealtimeNanos() - startTimeNanos
        Log.i(TAG, "Model's calculation time: ${calcTimeNanos * 1e-9} sec")

        val output = rawOutput.toTuple()
        if (output.size != OUTPUT_SIZE) {
            throw IllegalStateException(
                "Expected $OUTPUT_SIZE tensors in output, but was ${output.size}"
            )
        }

        val semiKeyPointsTensor = output[0].toTensor()
        val coarseDescriptorsTensor = output[1].toTensor()

        val semiKeyPoints = outputTensorToList(semiKeyPointsTensor)
        val coarseDescriptors = outputTensorToList(coarseDescriptorsTensor)

        return semiKeyPoints to coarseDescriptors
    }

    private fun Bitmap.toGrayscaleArray(): FloatArray {
        val pixels = IntArray(width * height)
        getPixels(pixels, 0, width, 0, 0, width, height)

        val grayscale = FloatArray(width * height) { i ->
            val r = linearizeSrgbChannel(Color.red(pixels[i]) / 255f)
            val g = linearizeSrgbChannel(Color.green(pixels[i]) / 255f)
            val b = linearizeSrgbChannel(Color.blue(pixels[i]) / 255f)
            return@FloatArray 0.2126f * r + 0.7152f * g + 0.0722f * b
        }

        return grayscale
    }

    private fun linearizeSrgbChannel(value: Float): Float =
        if (value <= 0.04045) value / 12.92f else ((value + 0.055f) / 1.055f).pow(2.4f)

    /**
     * Transforms tensor of shape 1xDxHxW to list with dimensions WxHxD according to the following
     * rule:
     * ```
     * result[x][y][d] == tensor[0][d][y][x]
     * ```
     */
    private fun outputTensorToList(tensor: Tensor): List<List<List<Float>>> {
        val data = tensor.dataAsFloatArray
        val (len1, len2, len3) = tensor
            .shape()
            .asList()
            .subList(1, 4) // strip redundant first dimension
            .map { it.toInt() }
        val list =
            List(len3) { k ->
                List(len2) { j ->
                    List(len1) { i ->
                        data[i * (len2 * len3) + j * (len3) + k]
                    }
                }
            }
        return list
    }
}
