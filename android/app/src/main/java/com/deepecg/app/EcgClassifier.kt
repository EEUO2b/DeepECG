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
    val probabilities: FloatArray,
    val multilabel: Boolean,
    val thresholds: FloatArray?
) {
    val label: String get() = labels[labelIndex]
    val confidence: Float get() = probabilities[labelIndex]
}

/**
 * Classifieur embarqué : charge ``ecg_model.tflite`` depuis les assets.
 *
 * La configuration est lue dans ``model_info.txt`` (écrit par export_tflite.py) :
 * ``num_leads``, ``samples_per_lead``, ``sampling_hz``, ``normalize``,
 * ``multilabel``, ``labels``, ``labels_fr``, ``thresholds``.
 *
 * Deux dispositions d'entrée sont supportées (déduites de la forme du tenseur) :
 *  - [1, T, C] avec C > 1 : entrelacement temps-major (ex. PTB-XL [1, 1000, 12]) ;
 *  - [1, ...] plat : concaténation des dérivations (ex. [1, 9000, 1] mono-dérivation).
 *
 * Chaque dérivation est ramenée à ``samples_per_lead`` échantillons à
 * ``sampling_hz`` : sous-échantillonnage linéaire si la trace est plus longue,
 * répétition (tiling) si elle est plus courte (ex. cellules de 2,5 s d'une
 * impression 3×4 ramenées à 10 s).
 */
class EcgClassifier(
    context: Context,
    modelAsset: String = MODEL_ASSET,
    infoAsset: String = INFO_ASSET
) : Closeable {

    val numLeads: Int
    val codes: List<String>
    private val interpreter: org.tensorflow.lite.Interpreter
    private val inputLength: Int
    private val timeChannels: Boolean
    private val samplesPerLead: Int
    private val samplingHz: Double
    private val normalize: Boolean
    private val padShort: Boolean
    private val multilabel: Boolean
    private val thresholds: FloatArray?
    private val labels: List<String>

    init {
        val info = mutableMapOf<String, String>()
        try {
            context.assets.open(infoAsset).bufferedReader().useLines { lines ->
                lines.forEach { line ->
                    val parts = line.split('=', limit = 2)
                    if (parts.size == 2) info[parts[0].trim()] = parts[1].trim()
                }
            }
        } catch (e: Exception) {
            // pas de model_info.txt : comportement par défaut mono-dérivation N,A,O,~
        }
        numLeads = info["num_leads"]?.toIntOrNull()?.coerceAtLeast(1) ?: 1
        samplingHz = info["sampling_hz"]?.toDoubleOrNull() ?: 300.0
        normalize = info["normalize"] == "true"
        padShort = info["pad_short"] == "true"
        multilabel = info["multilabel"] == "true"
        thresholds = info["thresholds"]?.split(",")
            ?.map { it.trim().toFloatOrNull() }?.filterNotNull()?.toFloatArray()
        codes = info["labels"]?.split(",")?.map { it.trim() } ?: listOf("N", "A", "O", "~")
        val names = mapOf(
            "N" to "Normal",
            "A" to "Fibrillation atriale",
            "O" to "Autre",
            "~" to "Bruité"
        )
        val fr = info["labels_fr"]?.split(",")?.map { it.trim() }
        labels = if (fr != null && fr.size == codes.size) fr else codes.map { names[it] ?: it }

        val afd = context.assets.openFd(modelAsset)
        val inputStream = FileInputStream(afd.fileDescriptor)
        val buffer = inputStream.channel.map(
            FileChannel.MapMode.READ_ONLY, afd.startOffset, afd.declaredLength
        )
        val options = org.tensorflow.lite.Interpreter.Options().setNumThreads(4)
        interpreter = org.tensorflow.lite.Interpreter(buffer, options)

        val shape = interpreter.getInputTensor(0).shape()
        inputLength = shape.fold(1) { acc, d -> acc * d }
        timeChannels = shape.size == 3 && shape[2] > 1
        samplesPerLead = if (timeChannels) {
            shape[1]
        } else {
            info["samples_per_lead"]?.toIntOrNull() ?: (inputLength / numLeads)
        }
    }

    fun classify(ecg: DigitizedEcg): ClassificationResult {
        val leads = ecg.leads
        val fitted = ArrayList<FloatArray>(numLeads)
        for (i in 0 until numLeads) {
            val src = leads.getOrElse(i) { leads.first() }
            fitted.add(fitLead(src, ecg.samplesPerSec, samplesPerLead, padShort))
        }

        val flat = FloatArray(inputLength)
        if (timeChannels) {
            val c = (inputLength / samplesPerLead).coerceAtLeast(1)
            var k = 0
            for (t in 0 until samplesPerLead) {
                for (ch in 0 until c) {
                    if (k < inputLength) flat[k++] = fitted[ch][t]
                }
            }
        } else {
            var k = 0
            for (l in fitted) {
                for (v in l) {
                    if (k < inputLength) flat[k++] = v
                }
            }
        }
        val input = if (normalize) zScore(flat) else flat

        val byteBuffer = ByteBuffer.allocateDirect(4 * inputLength).order(ByteOrder.nativeOrder())
        for (v in input) byteBuffer.putFloat(v)

        val output = Array(1) { FloatArray(labels.size) }
        interpreter.run(byteBuffer, output)
        val probs = output[0]
        val idx = probs.indices.maxByOrNull { probs[it] } ?: 0
        return ClassificationResult(labels, idx, probs, multilabel, thresholds)
    }

    /**
     * Ramène une dérivation à [target] échantillons à [samplingHz] :
     * interpolation linéaire si la source est plus longue, répétition si plus courte.
     */
    private fun fitLead(src: FloatArray, samplesPerSec: Double, target: Int, padShort: Boolean): FloatArray {
        if (src.isEmpty()) return FloatArray(target)
        val out = FloatArray(target)
        if (padShort && src.size < target) {
            val padded = FloatArray(target)
            src.copyInto(padded)
            return padded
        }
        val step = samplesPerSec / samplingHz
        for (i in 0 until target) {
            val pos = i * step
            if (pos < src.size) {
                val i0 = pos.toInt().coerceIn(0, src.size - 1)
                val i1 = (i0 + 1).coerceIn(0, src.size - 1)
                val t = (pos - i0).toFloat()
                out[i] = src[i0] * (1f - t) + src[i1] * t
            } else {
                out[i] = src[(pos % src.size).toInt()]
            }
        }
        return out
    }

    /** Normalisation z-score globale, identique à Conv1D_ECG.py. */
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
        const val AF_MODEL_ASSET = "af_model.tflite"
        const val AF_INFO_ASSET = "af_model_info.txt"
    }
}
