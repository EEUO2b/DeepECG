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
 * numérique, sans dépendance externe (pas d'OpenCV).
 *
 * Détection des bandes (améliorée) :
 *  1. cadrage sur la zone de contenu (les marges avec texte sont exclues) ;
 *  2. profil de lignes (nb de pixels sombres par ligne), lissé, seuillé à
 *     max(3, 0.4 × médiane des lignes non nulles) pour trouver les noyaux ;
 *  3. filtrage par couverture horizontale (≥ 40 % de la largeur) pour écarter
 *     le texte ;
 *  4. expansion de chaque noyau jusqu'au milieu entre noyaux voisins ;
 *  5. classement : ≥ 10 bandes → format 12×1, 3..9 → format 3×4 (la bande de
 *     rythme en bas, plus courte, est exclue), 1..2 → découpe en 3 bandes.
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

    private class Content(val x0: Int, val x1: Int, val y0: Int, val y1: Int)

    private class Band(val first: Int, val last: Int, val count: Int)

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
        val c = contentBox(m)
        val bands = detectBands(m, c)
        val pxX = gridSpacingX(m)
        val pxY = gridSpacingY(m)
        val mVperPx = if (pxY != null && pxY > 0) MV_PER_MM / pxY else null
        val cw = c.x1 - c.x0
        val ch = c.y1 - c.y0

        val leads = ArrayList<FloatArray>(12)
        val note: String
        val cellSec: Double
        val refWidth: Int
        if (bands.size >= 10) {
            // format 12×1 : chaque bande = une dérivation sur toute la largeur (10 s)
            val top = bands.sortedByDescending { it.count }.take(12).sortedBy { it.first }
            cellSec = 10.0
            refWidth = cw
            for (b in top) {
                leads.add(toMv(extractColumnTrace(m, c.x0, c.x1, c.y0 + b.first, c.y0 + b.last), mVperPx))
            }
            note = "format 12×1 (${top.size}/${bands.size} bandes)"
        } else {
            // format 3×4 : 3 bandes × 4 colonnes de 2,5 s
            cellSec = 2.5
            refWidth = cw / 4
            var cand = bands
            if (cand.size >= 4) {
                // exclure une éventuelle bande de rythme en bas (plus courte)
                val heights = cand.dropLast(1).map { it.last - it.first }.sorted()
                val medH = heights[heights.size / 2]
                val bottom = cand.last()
                if (bottom.last - bottom.first < 0.6 * medH) cand = cand.dropLast(1)
            }
            val rows: List<Band>
            if (cand.size >= 3) {
                rows = cand.sortedByDescending { it.count }.take(3).sortedBy { it.first }
                note = "format 3×4 (${rows.size}/${bands.size} bandes, 2,5 s/dérivation)"
            } else {
                rows = (0..2).map { Band(it * ch / 3, (it + 1) * ch / 3, 0) }
                note = "format 3×4 estimé (${bands.size} bande(s), découpe en 3)"
            }
            val cols = intArrayOf(0, cw / 4, cw / 2, 3 * cw / 4, cw)
            for (col in 0..3) {
                for (row in rows) {
                    leads.add(
                        toMv(
                            extractColumnTrace(m, c.x0 + cols[col], c.x0 + cols[col + 1], c.y0 + row.first, c.y0 + row.last),
                            mVperPx
                        )
                    )
                }
            }
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

    /** Zone de contenu : exclut les marges (texte, bords). */
    private fun contentBox(m: Masks): Content {
        val w = m.w
        val h = m.h
        val colThr = maxOf(2, h / 200)
        val rowThr = maxOf(2, w / 200)
        var x0 = -1
        var x1 = -1
        var y0 = -1
        var y1 = -1
        for (x in 0 until w) {
            var c = 0
            for (y in 0 until h) if (m.dark[y * w + x]) c++
            if (c >= colThr) {
                if (x0 < 0) x0 = x
                x1 = x
            }
        }
        for (y in 0 until h) {
            var c = 0
            val base = y * w
            for (x in 0 until w) if (m.dark[base + x]) c++
            if (c >= rowThr) {
                if (y0 < 0) y0 = y
                y1 = y
            }
        }
        if (x0 < 0 || y0 < 0) return Content(0, w, 0, h)
        return Content(x0, x1 + 1, y0, y1 + 1)
    }

    /** Bandes horizontales de tracé dans la zone de contenu (voir doc de la classe). */
    private fun detectBands(m: Masks, c: Content): List<Band> {
        val w = m.w
        val cw = c.x1 - c.x0
        val ch = c.y1 - c.y0
        if (cw <= 0 || ch <= 0) return emptyList()

        val rc = DoubleArray(ch)
        for (yy in 0 until ch) {
            val base = (c.y0 + yy) * w
            var s = 0
            for (xx in 0 until cw) if (m.dark[base + c.x0 + xx]) s++
            rc[yy] = s.toDouble()
        }
        val sm = DoubleArray(ch)
        for (y in 0 until ch) {
            var s = 0.0
            for (d in -1..1) {
                val yy = y + d
                if (yy in 0 until ch) s += rc[yy]
            }
            sm[y] = s / 3.0
        }
        val pos = sm.filter { it > 0.0 }
        val med = if (pos.isEmpty()) 3.0 else pos.sorted()[pos.size / 2]
        val tHi = maxOf(3.0, med * 0.4)

        val cores = ArrayList<Pair<Int, Int>>()
        var inBand = false
        var start = 0
        var gap = 0
        for (y in 0 until ch) {
            if (sm[y] >= tHi && !inBand) {
                inBand = true
                start = y
                gap = 0
            } else if (sm[y] < tHi && inBand) {
                gap++
                if (gap > 8) {
                    cores.add(start to (y - gap))
                    inBand = false
                    gap = 0
                }
            }
        }
        if (inBand) cores.add(start to ch)

        val filtered = ArrayList<Pair<Int, Int>>()
        for ((a, b) in cores) {
            if (b - a < 10) continue
            var cov = 0
            for (xx in 0 until cw) {
                var has = false
                for (yy in a until b) {
                    if (m.dark[(c.y0 + yy) * w + c.x0 + xx]) {
                        has = true
                        break
                    }
                }
                if (has) cov++
            }
            if (cov >= 0.4 * cw) filtered.add(a to b)
        }
        filtered.sortBy { it.first }

        val bands = ArrayList<Band>()
        for (i in filtered.indices) {
            val (a, b) = filtered[i]
            val lo = if (i == 0) 0 else (filtered[i - 1].second + a) / 2
            val hi = if (i == filtered.size - 1) ch else (b + filtered[i + 1].first) / 2
            var cnt = 0.0
            for (yy in lo until hi) cnt += rc[yy]
            bands.add(Band(lo, hi, cnt.toInt()))
        }
        return bands
    }

    /** Position verticale (médiane) du tracé par colonne dans la cellule [x0,x1)×[y0,y1), en pixels. */
    private fun extractColumnTrace(m: Masks, x0: Int, x1: Int, y0: Int, y1: Int): DoubleArray {
        val w = m.w
        val n = x1 - x0
        val y = DoubleArray(n) { Double.NaN }
        val tmp = ArrayList<Int>(y1 - y0 + 1)
        for (xx in 0 until n) {
            val x = x0 + xx
            tmp.clear()
            for (yy in y0..y1) {
                if (m.dark[yy * w + x]) tmp.add(yy)
            }
            if (tmp.size >= 2) {
                tmp.sort()
                y[xx] = tmp[tmp.size / 2].toDouble()
            }
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
