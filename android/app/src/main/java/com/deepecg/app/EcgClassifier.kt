package com.deepecg.app

import android.content.Context
import java.io.Closeable
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel
import kotlin.math.sqrt

data class ClassificationResult(
    val labels: List<String>,
    val labelIndex: Int,
    val probabilities: FloatArray
) {
    val label: String get() = labels[labelIndex]
    val confidence: Float get() = probabilities[labelIndex]
}

/**
 * Classifieur embarqué : charge ``ecg_model.tflite`` depuis les assets et
 * prédit une des 4 classes PhysioNet/CinC 2017 (N, A, O, ~).
 *
 * La longueur d'entrée et le besoin de normalisation sont lus dans
 * ``model_info.txt`` (écrit par export_tflite.py) ; à défaut on retombe sur
 * 9000 échantillons sans normalisation (modèle CNN_ECG.py).
 */
class EcgClassifier(context: Context) : Closeable {

    private val interpreter: org.tensorflow.lite.Interpreter
    private val inputLength: Int
    private val normalize: Boolean
    private val labels: List<String> =
        listOf("Normal", "Fibrillation atriale", "Autre", "Bruité")

    init {
        val afd = context.assets.openFd(MODEL_ASSET)
        val inputStream = FileInputStream(afd.fileDescriptor)
        val buffer = inputStream.channel.map(
            FileChannel.MapMode.READ_ONLY, afd.startOffset, afd.declaredLength
        )
        val options = org.tensorflow.lite.Interpreter.Options().setNumThreads(4)
        interpreter = org.tensorflow.lite.Interpreter(buffer, options)

        val shape = interpreter.getInputTensor(0).shape()
        inputLength = shape.fold(1) { acc, d -> acc * d }

        val info = mutableMapOf<String, String>()
        try {
            context.assets.open(INFO_ASSET).bufferedReader().useLines { lines ->
                lines.forEach { line ->
                    val parts = line.split('=', limit = 2)
                    if (parts.size == 2) info[parts[0].trim()] = parts[1].trim()
                }
            }
        } catch (e: Exception) {
            // pas de model_info.txt : comportement par défaut CNN_ECG.py
        }
        normalize = info["normalize"] == "true"
    }

    fun classify(ecg: DigitizedEcg): ClassificationResult {
        val resampled = resample(ecg.samplesMv, inputLength)
        val input = if (normalize) zScore(resampled) else resampled

        val byteBuffer = ByteBuffer.allocateDirect(4 * inputLength).order(ByteOrder.nativeOrder())
        for (v in input) byteBuffer.putFloat(v)

        val output = Array(1) { FloatArray(labels.size) }
        interpreter.run(byteBuffer, output)
        val probs = output[0]
        val idx = probs.indices.maxByOrNull { probs[it] } ?: 0
        return ClassificationResult(labels, idx, probs)
    }

    /** Rééchantillonne le tracé sur [length] points (fenêtre de 30 s à 300 Hz). */
    private fun resample(src: FloatArray, length: Int): FloatArray {
        if (src.size == length) return src.copyOf()
        val out = FloatArray(length)
        val step = (src.size - 1).toDouble() / (length - 1)
        for (i in 0 until length) {
            val pos = i * step
            val i0 = pos.toInt().coerceIn(0, src.size - 1)
            val i1 = (i0 + 1).coerceIn(0, src.size - 1)
            val t = pos - i0
            out[i] = (src[i0] * (1.0f - t.toFloat()) + src[i1] * t.toFloat())
        }
        return out
    }

    /** Normalisation z-score par segment, identique à Conv1D_ECG.py. */
    private fun zScore(src: FloatArray): FloatArray {
        val mean = src.average()
        var variance = 0.0
        for (v in src) variance += (v - mean) * (v - mean)
        val std = sqrt(variance / src.size).toFloat().coerceAtLeast(1e-6f)
        return FloatArray(src.size) { ((src[it] - mean) / std).toFloat() }
    }

    override fun close() {
        interpreter.close()
    }

    companion object {
        const val MODEL_ASSET = "ecg_model.tflite"
        const val INFO_ASSET = "model_info.txt"
    }
}
