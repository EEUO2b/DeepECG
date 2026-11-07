package com.deepecg.app

import android.graphics.Bitmap
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

/**
 * Signal extrait d'une photo d'impression ECG.
 * [samplesMv] : un échantillon par colonne de l'image, en mV.
 * [samplesPerSec] : fréquence d'échantillonnage du signal extrait.
 * [note] : informations de calibration (grille détectée ou non).
 */
data class DigitizedEcg(
    val samplesMv: FloatArray,
    val samplesPerSec: Double,
    val note: String
)

/**
 * Transforme la photo d'un tracé ECG imprimé sur papier millimétré en signal
 * numérique, sans dépendance externe (pas d'OpenCV) :
 *  1. conversion en niveaux de gris et seuillage pour isoler le tracé sombre ;
 *  2. pour chaque colonne, position verticale moyenne des pixels du tracé,
 *     puis médiane glissante et interpolation des trous ;
 *  3. détection de la grille rose pour calibrer le temps (25 mm/s, 300 Hz)
 *     et l'amplitude (10 mm/mV) ;
 *  4. conversion en mV.
 */
object EcgDigitizer {

    private const val MAX_WIDTH = 1600
    private const val SAMPLING_HZ = 300.0
    private const val PAPER_SPEED_MM_S = 25.0
    private const val MV_PER_MM = 0.1 // 10 mm/mV

    fun digitize(src: Bitmap): DigitizedEcg {
        val scale = min(1.0, MAX_WIDTH.toDouble() / src.width)
        val bmp = if (scale < 1.0) {
            Bitmap.createScaledBitmap(
                src,
                (src.width * scale).roundToInt().coerceAtLeast(1),
                (src.height * scale).roundToInt().coerceAtLeast(1),
                true
            )
        } else {
            src
        }
        val w = bmp.width
        val h = bmp.height
        val px = IntArray(w * h)
        bmp.getPixels(px, 0, w, 0, 0, w, h)
        if (bmp !== src) bmp.recycle()

        val lum = IntArray(w * h)
        for (k in 0 until w * h) {
            val c = px[k]
            lum[k] = (((c shr 16) and 0xFF) * 299 + ((c shr 8) and 0xFF) * 587 + (c and 0xFF) * 114) / 1000
        }
        val sorted = IntArray(w * h)
        var i = 0
        var n = 0
        while (i < w * h) {
            sorted[n++] = lum[i]
            i += 16
        }
        val sample = sorted.copyOf(n)
        sample.sort()
        val bg = sample[n / 2]

        val dark = BooleanArray(w * h)
        val grid = BooleanArray(w * h)
        for (k in 0 until w * h) {
            val c = px[k]
            val r = (c shr 16) and 0xFF
            val g = (c shr 8) and 0xFF
            val b = c and 0xFF
            val mx = max(r, max(g, b))
            val mn = min(r, min(g, b))
            val sat = if (mx == 0) 0 else (mx - mn) * 255 / mx
            dark[k] = (mx < bg - 50) || (sat > 90 && mx < bg - 10)
            grid[k] = r > g + 25 && r > b + 25 && lum[k] > bg - 90
        }

        val yTrace = DoubleArray(w) { Double.NaN }
        for (x in 0 until w) {
            var count = 0
            var sum = 0
            for (y in 0 until h) {
                if (dark[y * w + x]) {
                    count++
                    sum += y
                }
            }
            if (count >= 2) yTrace[x] = sum.toDouble() / count
        }

        val filtered = DoubleArray(w)
        for (x in 0 until w) {
            val vals = ArrayList<Double>(5)
            for (dx in -2..2) {
                val xx = x + dx
                if (xx in 0 until w && !yTrace[xx].isNaN()) vals.add(yTrace[xx])
            }
            filtered[x] = if (vals.isEmpty()) Double.NaN else vals.sorted()[vals.size / 2]
        }

        var first = 0
        while (first < w && filtered[first].isNaN()) first++
        var last = w - 1
        while (last >= 0 && filtered[last].isNaN()) last--
        if (last - first < 100) {
            throw IllegalArgumentException("Aucun tracé ECG détecté sur la photo")
        }

        val y = FloatArray(last - first + 1)
        var prevX = first
        var x = first
        while (x <= last) {
            if (!filtered[x].isNaN()) {
                y[x - first] = filtered[x].toFloat()
                prevX = x
                x++
            } else {
                var nextX = x
                while (nextX <= last && filtered[nextX].isNaN()) nextX++
                if (nextX > last) {
                    y[x - first] = y[prevX - first]
                } else {
                    val t = (x - prevX).toDouble() / (nextX - prevX)
                    y[x - first] = (filtered[prevX] * (1.0 - t) + filtered[nextX] * t).toFloat()
                }
                x++
            }
        }

        val colGrid = IntArray(w)
        for (xx in 0 until w) {
            var c = 0
            for (yy in 0 until h) if (grid[yy * w + xx]) c++
            colGrid[xx] = c
        }
        val rowGrid = IntArray(h)
        for (yy in 0 until h) {
            var c = 0
            for (xx in 0 until w) if (grid[yy * w + xx]) c++
            rowGrid[yy] = c
        }
        val pxPerMmX = estimateGridSpacing(colGrid)
        val pxPerMmY = estimateGridSpacing(rowGrid)

        // À 25 mm/s et 300 Hz, il y a 12 échantillons par mm de papier ;
        // une colonne de l'image vaut donc (12 / pxParMm) échantillons.
        val samplesPerSec: Double
        val timeNote: String
        if (pxPerMmX != null && pxPerMmX > 0) {
            samplesPerSec = SAMPLING_HZ / PAPER_SPEED_MM_S / pxPerMmX
            timeNote = "grille détectée (${"%.1f".format(pxPerMmX)} px/mm)"
        } else {
            samplesPerSec = (last - first + 1).toDouble() / (9000.0 / SAMPLING_HZ)
            timeNote = "grille non détectée, largeur supposée = 30 s"
        }

        val sortedY = y.clone().sorted()
        val y0 = sortedY[sortedY.size / 2]
        val p5 = sortedY[(sortedY.size * 0.05).toInt().coerceIn(0, sortedY.size - 1)]
        val p95 = sortedY[(sortedY.size * 0.95).toInt().coerceIn(0, sortedY.size - 1)]
        val mVperPx: Double
        val ampNote: String
        if (pxPerMmY != null && pxPerMmY > 0) {
            mVperPx = MV_PER_MM / pxPerMmY
            ampNote = "amplitude calibrée (${"%.1f".format(pxPerMmY)} px/mm, 10 mm/mV)"
        } else {
            mVperPx = 2.0 / max(1.0, (p95 - p5).toDouble())
            ampNote = "amplitude estimée (crête-crête supposée ≈ 2 mV)"
        }

        val samplesMv = FloatArray(y.size) { (y0 - y[it]) * mVperPx }
        return DigitizedEcg(
            samplesMv = samplesMv,
            samplesPerSec = samplesPerSec,
            note = "$timeNote ; $ampNote"
        )
    }

    /**
     * Estime l'espacement médian (en pixels) entre les pics d'un profil 1D,
     * c'est-à-dire le pas de la grille (1 mm). Retourne null si moins de
     * 3 pics fiables.
     */
    private fun estimateGridSpacing(profile: IntArray): Double? {
        val n = profile.size
        if (n < 30) return null
        var mean = 0.0
        var mx = 0
        for (v in profile) {
            mean += v
            if (v > mx) mx = v
        }
        mean /= n
        val threshold = mean + (mx - mean) * 0.4
        val peaks = ArrayList<Int>()
        var x = 1
        while (x < n - 1) {
            if (profile[x] >= threshold && profile[x] >= profile[x - 1] && profile[x] > profile[x + 1]) {
                if (peaks.isEmpty() || x - peaks[peaks.size - 1] > 3) peaks.add(x)
            }
            x++
        }
        if (peaks.size < 3) return null
        val diffs = ArrayList<Int>()
        for (k in 1 until peaks.size) diffs.add(peaks[k] - peaks[k - 1])
        diffs.sort()
        return diffs[diffs.size / 2].toDouble()
    }
}
