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

    fun forward(image: Bitmap): Pair<LongArray, LongArray> {
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

        val keypoints = output[0].toTensor()
        val descriptors = output[1].toTensor()

        // TODO: convert tensors to multi-dimensional float arrays

        return keypoints.shape() to descriptors.shape()
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
}
