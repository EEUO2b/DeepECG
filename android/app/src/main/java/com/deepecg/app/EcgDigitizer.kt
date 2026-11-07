package com.deepecg.app

import android.graphics.Bitmap
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

/**
 * Signal extrait d'une photo d'impression ECG.
 * [leads] : une trace par dérivation (ordre standard I, II, III, aVR, aVL, aVF,
 * V1..V6 pour un ECG 12 dérivations ; une seule trace en mono-dérivation),
 * en mV, un échantillon par colonne de l'image.
 * [samplesPerSec] : fréquence d'échantillonnage des traces extraites.
 * [note] : informations de calibration / format détecté.
 */
data class DigitizedEcg(
    val leads: List<FloatArray>,
    val samplesPerSec: Double,
    val note: String
)

/**
 * Transforme la photo d'un tracé ECG imprimé sur papier millimétré en signal
 * numérique, sans dépendance externe (pas d'OpenCV) :
 *  1. conversion en niveaux de gris et seuillage pour isoler le tracé sombre ;
 *  2. détection des bandes horizontales (1 bande = mono-dérivation pleine
 *     largeur, 3 bandes = format 3×4, 12 bandes = format 12×1) ;
 *  3. pour chaque cellule, position verticale du tracé colonne par colonne,
 *     médiane glissante et interpolation des trous ;
 *  4. détection de la grille rose pour calibrer le temps (25 mm/s, 300 Hz)
 *     et l'amplitude (10 mm/mV) ;
 *  5. conversion en mV.
 */
object EcgDigitizer {

    private const val MAX_WIDTH = 1600
    private const val SAMPLING_HZ = 300.0
    private const val PAPER_SPEED_MM_S = 25.0
    private const val MV_PER_MM = 0.1 // 10 mm/mV

    /** Ordre standard des 12 dérivations d'une impression 3×4 (colonne par colonne). */
    val LEAD_ORDER_12 = listOf("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")

    private class Masks(
        val w: Int,
        val h: Int,
        val dark: BooleanArray,
        val grid: BooleanArray
    )

    /** Mono-dérivation : toute l'image est une seule trace (30 s supposées si pas de grille). */
    fun digitize(src: Bitmap): DigitizedEcg {
        val m = buildMasks(src)
        val pxX = gridSpacingX(m)
        val pxY = gridSpacingY(m)
        val samplesPerSec = if (pxX != null && pxX > 0) {
            SAMPLING_HZ / PAPER_SPEED_MM_S / pxX
        } else {
            m.w / 30.0
        }
        val timeNote = if (pxX != null && pxX > 0) {
            "grille détectée (${"%.1f".format(pxX)} px/mm)"
        } else {
            "grille non détectée, largeur supposée = 30 s"
        }
        val ampNote = if (pxY != null && pxY > 0) {
            "amplitude calibrée (${"%.1f".format(pxY)} px/mm, 10 mm/mV)"
        } else {
            "amplitude estimée (crête-crête ≈ 2 mV)"
        }
        val y = extractColumnTrace(m, 0, m.w, 0, m.h)
        val mv = toMv(y, if (pxY != null && pxY > 0) MV_PER_MM / pxY else null)
        return DigitizedEcg(listOf(mv), samplesPerSec, "$timeNote ; $ampNote")
    }

    /** 12 dérivations : détecte le format 3×4 (3 bandes × 4 colonnes de 2,5 s) ou 12×1. */
    fun digitize12(src: Bitmap): DigitizedEcg {
        val m = buildMasks(src)
        val bands = rowBands(m)
        val pxX = gridSpacingX(m)
        val pxY = gridSpacingY(m)
        val mVperPx = if (pxY != null && pxY > 0) MV_PER_MM / pxY else null

        val leads = ArrayList<FloatArray>(12)
        val note: String
        val cellSec: Double
        val refWidth: Int
        if (bands.size in 10..14) {
            cellSec = 10.0
            refWidth = m.w
            for (b in bands.take(12)) {
                leads.add(toMv(extractColumnTrace(m, 0, m.w, b.first, b.last), mVperPx))
            }
            note = "format 12×1 détecté (${leads.size} dérivations, pleine largeur)"
        } else if (bands.size in 2..4) {
            cellSec = 2.5
            val rows = bands.take(3)
            val cols = columnCells(m, rows)
            refWidth = cols[1] - cols[0]
            for (col in 0..3) {
                for (row in rows.indices) {
                    val b = rows[row]
                    leads.add(toMv(extractColumnTrace(m, cols[col], cols[col + 1], b.first, b.last), mVperPx))
                }
            }
            note = "format 3×4 détecté (2,5 s par dérivation, ordre colonne-major)"
        } else {
            throw IllegalArgumentException("Format 12 dérivations non reconnu (${bands.size} bandes détectées)")
        }

        val samplesPerSec = if (pxX != null && pxX > 0) {
            SAMPLING_HZ / PAPER_SPEED_MM_S / pxX
        } else {
            refWidth / cellSec
        }
        val calNote = if (pxX != null && pxX > 0) "grille détectée" else "grille non détectée"
        return DigitizedEcg(leads, samplesPerSec, "$note ; $calNote")
    }

    private fun buildMasks(src: Bitmap): Masks {
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
        val sample = IntArray((w * h) / 16 + 1)
        var i = 0
        var n = 0
        while (i < w * h) {
            sample[n++] = lum[i]
            i += 16
        }
        val sorted = sample.copyOf(n)
        sorted.sort()
        val bg = sorted[n / 2]

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
        return Masks(w, h, dark, grid)
    }

    /** Bandes horizontales contenant du tracé (après lissage). */
    private fun rowBands(m: Masks): List<IntRange> {
        val w = m.w
        val h = m.h
        val rowCount = IntArray(h)
        for (y in 0 until h) {
            var c = 0
            val base = y * w
            for (x in 0 until w) if (m.dark[base + x]) c++
            rowCount[y] = c
        }
        val sm = IntArray(h)
        for (y in 0 until h) {
            var s = 0
            for (d in -2..2) {
                val yy = y + d
                if (yy in 0 until h) s += rowCount[yy]
            }
            sm[y] = s
        }
        val thr = maxOf(8, w / 100) * 5
        val raw = ArrayList<IntRange>()
        var start = -1
        var y = 0
        while (y < h) {
            if (sm[y] >= thr) {
                if (start < 0) start = y
            } else if (start >= 0) {
                raw.add(start until y)
                start = -1
            }
            y++
        }
        if (start >= 0) raw.add(start until h)
        val merged = ArrayList<IntRange>()
        for (b in raw) {
            if (merged.isNotEmpty() && b.first - merged.last().last <= 6) {
                val prev = merged.removeAt(merged.size - 1)
                merged.add(prev.first until b.last)
            } else {
                merged.add(b)
            }
        }
        return merged.filter { it.last - it.first >= 30 }
    }

    /** Limites des 4 colonnes d'un format 3×4 (lignes verticales de la grille, sinon découpe égale). */
    private fun columnCells(m: Masks, rows: List<IntRange>): IntArray {
        val w = m.w
        val bandHeight = rows.sumOf { it.last - it.first + 1 }
        val colScore = IntArray(w)
        for (x in 0 until w) {
            var c = 0
            for (b in rows) {
                for (y in b.first..b.last) {
                    if (m.grid[y * w + x] || m.dark[y * w + x]) c++
                }
            }
            colScore[x] = c
        }
        val peaks = ArrayList<Int>()
        var x = 1
        while (x < w - 1) {
            if (colScore[x] >= colScore[x - 1] && colScore[x] > colScore[x + 1] && colScore[x] > bandHeight * 2) {
                peaks.add(x)
            }
            x++
        }
        val chosen = peaks.sortedByDescending { colScore[it] }.take(3).sorted()
        return if (chosen.size == 3 && chosen[1] - chosen[0] > w / 8 && chosen[2] - chosen[1] > w / 8) {
            intArrayOf(0, chosen[0], chosen[1], chosen[2], w)
        } else {
            intArrayOf(0, w / 4, w / 2, 3 * w / 4, w)
        }
    }

    /** Position verticale du tracé par colonne dans la cellule [x0,x1)×[y0,y1], en pixels. */
    private fun extractColumnTrace(m: Masks, x0: Int, x1: Int, y0: Int, y1: Int): DoubleArray {
        val w = m.w
        val n = x1 - x0
        val y = DoubleArray(n) { Double.NaN }
        for (xx in 0 until n) {
            val x = x0 + xx
            var count = 0
            var sum = 0
            for (yy in y0..y1) {
                if (m.dark[yy * w + x]) {
                    count++
                    sum += yy
                }
            }
            if (count >= 2) y[xx] = sum.toDouble() / count
        }
        val f = DoubleArray(n)
        for (i in 0 until n) {
            val vals = ArrayList<Double>(5)
            for (d in -2..2) {
                val j = i + d
                if (j in 0 until n && !y[j].isNaN()) vals.add(y[j])
            }
            f[i] = if (vals.isEmpty()) Double.NaN else vals.sorted()[vals.size / 2]
        }
        var first = 0
        while (first < n && f[first].isNaN()) first++
        var last = n - 1
        while (last >= 0 && f[last].isNaN()) last--
        val out = DoubleArray(n) { Double.NaN }
        if (last - first < 20) return out
        var prev = first
        var i = first
        while (i <= last) {
            if (!f[i].isNaN()) {
                out[i] = f[i]
                prev = i
                i++
            } else {
                var next = i
                while (next <= last && f[next].isNaN()) next++
                if (next > last) {
                    out[i] = out[prev]
                } else {
                    val t = (i - prev).toDouble() / (next - prev)
                    out[i] = f[prev] * (1.0 - t) + f[next] * t
                }
                i++
            }
        }
        for (i in 0 until first) out[i] = out[first]
        for (i in last + 1 until n) out[i] = out[last]
        return out
    }

    /** Conversion pixels -> mV ; si [mVperPx] est null, amplitude estimée (crête-crête ≈ 2 mV). */
    private fun toMv(y: DoubleArray, mVperPx: Double?): FloatArray {
        val valid = y.filter { !it.isNaN() }
        if (valid.isEmpty()) return FloatArray(y.size)
        val sorted = valid.sorted()
        val y0 = sorted[sorted.size / 2]
        val scale = mVperPx ?: run {
            val p5 = sorted[(sorted.size * 0.05).toInt().coerceIn(0, sorted.size - 1)]
            val p95 = sorted[(sorted.size * 0.95).toInt().coerceIn(0, sorted.size - 1)]
            2.0 / max(1.0, p95 - p5)
        }
        return FloatArray(y.size) { ((y0 - y[it]) * scale).toFloat() }
    }

    private fun gridSpacingX(m: Masks): Double? {
        val colGrid = IntArray(m.w)
        for (x in 0 until m.w) {
            var c = 0
            for (y in 0 until m.h) if (m.grid[y * m.w + x]) c++
            colGrid[x] = c
        }
        return estimateGridSpacing(colGrid)
    }

    private fun gridSpacingY(m: Masks): Double? {
        val rowGrid = IntArray(m.h)
        for (y in 0 until m.h) {
            var c = 0
            for (x in 0 until m.w) if (m.grid[y * m.w + x]) c++
            rowGrid[y] = c
        }
        return estimateGridSpacing(rowGrid)
    }

    /**
     * Espacement médian (en pixels) entre les pics d'un profil 1D, c'est-à-dire
     * le pas de la grille (1 mm). Retourne null si moins de 3 pics fiables.
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
