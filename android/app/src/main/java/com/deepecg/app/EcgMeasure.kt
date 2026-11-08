package com.deepecg.app

import kotlin.math.abs
import kotlin.math.atan2
import kotlin.math.sqrt

/**
 * Mesures ECG calculées sur le signal numérisé (estimations) :
 * fréquence cardiaque, rythme, intervalles PR / QRS / QT (QTc de Bazett)
 * et axe cardiaque (dérivations I et aVF si 12 dérivations).
 */
data class EcgMeasurements(
    val heartRate: Double,
    val rhythm: String, // "sinus" | "fa" | "irregulier"
    val rrCv: Double,   // coefficient de variation des intervalles RR
    val prMs: Double?,
    val qrsMs: Double,
    val qtMs: Double?,
    val qtcMs: Double?,
    val axisDeg: Double?,
    val beats: Int
)

object EcgMeasure {

    /**
     * [leads] en mV, échantillonnées à [samplesPerSec].
     * Les intervalles sont mesurés sur la dérivation II si elle existe, sinon sur la seule dérivation.
     */
    fun measure(leads: List<FloatArray>, samplesPerSec: Double): EcgMeasurements? {
        if (leads.isEmpty() || samplesPerSec <= 0.0) return null
        val fs = samplesPerSec
        val lead = if (leads.size > 1) leads[1] else leads[0]
        val x = DoubleArray(lead.size) { lead[it].toDouble() }
        val r = findRpeaks(x, fs)
        if (r.size < 3) return null

        val base = median(x)
        val rr = DoubleArray(r.size - 1) { (r[it + 1] - r[it]) / fs }
        val rrMed = median(rr)
        if (rrMed <= 0.0) return null
        val heartRate = 60.0 / rrMed
        val cv = std(rr) / rr.average()

        val prs = ArrayList<Double>()
        val qrss = ArrayList<Double>()
        val qts = ArrayList<Double>()
        var pFound = 0
        val n = (r.size - 2).coerceAtLeast(1)
        for (k in 1 until r.size - 1) {
            val p = r[k]
            val amp = abs(x[p] - base)
            // début/fin QRS : minima locaux de |x - base| de part et d'autre du pic R
            val llo = maxOf(0, p - (0.10 * fs).toInt())
            val lhi = maxOf(0, p - (0.02 * fs).toInt())
            val rlo = minOf(x.size, p + (0.02 * fs).toInt())
            val rhi = minOf(x.size, p + (0.10 * fs).toInt())
            if (lhi <= llo || rhi <= rlo) continue
            val on = llo + argminAbs(x, base, llo, lhi)
            val off = rlo + argminAbs(x, base, rlo, rhi)
            qrss.add((off - on) / fs * 1000.0)
            // onde P avant le QRS
            val plo = maxOf(0, on - (0.28 * fs).toInt())
            val phi = maxOf(0, on - (0.05 * fs).toInt())
            if (phi > plo) {
                var pk = plo
                var pv = -1e9
                for (j in plo until phi) {
                    if (x[j] - base > pv) {
                        pv = x[j] - base
                        pk = j
                    }
                }
                val pamp = x[pk] - base
                if (pamp > maxOf(0.03, 0.08 * amp)) {
                    val thr = base + 0.15 * pamp
                    var pon = pk
                    while (pon > plo && x[pon] > thr) pon--
                    prs.add((on - pon) / fs * 1000.0)
                    pFound++
                }
            }
            // fin de l'onde T après le QRS
            val tlo = off
            val thi = minOf(x.size, off + (0.50 * fs).toInt())
            if (thi > tlo + 2) {
                var tpk = tlo
                var tv = -1.0
                for (j in tlo until thi) {
                    val v = abs(x[j] - base)
                    if (v > tv) {
                        tv = v
                        tpk = j
                    }
                }
                val wlo = tpk
                val whi = minOf(x.size, tpk + (0.15 * fs).toInt())
                val tend = wlo + argminAbs(x, base, wlo, whi)
                qts.add((tend - on) / fs * 1000.0)
            }
        }
        if (qrss.isEmpty()) return null
        val qtMed = if (qts.isEmpty()) null else median(qts.toDoubleArray())
        val qtc = qtMed?.div(sqrt(rrMed))
        val prMed = if (prs.isNotEmpty() && prs.size >= 0.5 * n) median(prs.toDoubleArray()) else null
        // Rythme : la FA = RR "irrégulièrement irréguliers" (cv élevé). La
        // présence d'ondes P n'est pas fiable (les ondes f mimpent des P).
        val rhythm = when {
            cv >= 0.10 -> "fa"
            pFound >= 0.5 * n -> "sinus"
            else -> "irregulier"
        }
        val axis = if (leads.size >= 6) computeAxis(leads[0], leads[5], fs) else null
        return EcgMeasurements(heartRate, rhythm, cv, prMed, median(qrss.toDoubleArray()), qtMed, qtc, axis, r.size)
    }

    /** Axe frontal = atan2(aire nette QRS en aVF, aire nette QRS en I), en degrés. */
    private fun computeAxis(leadI: FloatArray, leadAvf: FloatArray, fs: Double): Double? {
        val xI = DoubleArray(leadI.size) { leadI[it].toDouble() }
        val xF = DoubleArray(leadAvf.size) { leadAvf[it].toDouble() }
        val r = findRpeaks(xI, fs)
        if (r.size < 3) return null
        val baseI = median(xI)
        val baseF = median(xF)
        val aI = ArrayList<Double>()
        val aF = ArrayList<Double>()
        for (k in 1 until r.size - 1) {
            val p = r[k]
            val lo = maxOf(0, p - (0.05 * fs).toInt())
            val hi = minOf(xI.size, p + (0.07 * fs).toInt())
            var sI = 0.0
            var sF = 0.0
            for (j in lo until hi) {
                sI += xI[j] - baseI
                sF += xF.getOrElse(j) { 0.0 } - baseF
            }
            aI.add(sI)
            aF.add(sF)
        }
        return Math.toDegrees(atan2(median(aF.toDoubleArray()), median(aI.toDoubleArray())))
    }

    /** Détection des pics R : passe-bande maison (haut 0,5 Hz + lissage), |gradient| intégré, seuil adaptatif. */
    private fun findRpeaks(x: DoubleArray, fs: Double): IntArray {
        val s = x.copyOf()
        val hp = movingAverage(s, (0.5 * fs).toInt())
        for (i in s.indices) s[i] -= hp[i]
        val sm = movingAverage(s, (0.02 * fs).toInt())
        val d = DoubleArray(sm.size)
        for (i in 1 until sm.size) d[i] = abs(sm[i] - sm[i - 1])
        val e = movingAverage(d, (0.08 * fs).toInt())
        var mx = 0.0
        for (v in e) if (v > mx) mx = v
        val thr = 0.4 * mx
        val peaks = ArrayList<Int>()
        var i = 1
        while (i < e.size - 1) {
            if (e[i] >= thr && e[i] >= e[i - 1] && e[i] > e[i + 1]) {
                if (peaks.isEmpty() || i - peaks.last() > (0.25 * fs).toInt()) {
                    peaks.add(i)
                } else if (e[i] > e[peaks.last()]) {
                    peaks[peaks.size - 1] = i
                }
            }
            i++
        }
        val base = median(x)
        return peaks.map { p ->
            val lo = maxOf(0, p - (0.05 * fs).toInt())
            val hi = minOf(x.size, p + (0.05 * fs).toInt() + 1)
            var best = p
            var bv = -1.0
            for (j in lo until hi) {
                val v = abs(x[j] - base)
                if (v > bv) {
                    bv = v
                    best = j
                }
            }
            best
        }.toIntArray()
    }

    private fun argminAbs(x: DoubleArray, base: Double, lo: Int, hi: Int): Int {
        var best = 0
        var bv = Double.MAX_VALUE
        for (j in lo until hi) {
            val v = abs(x[j] - base)
            if (v < bv) {
                bv = v
                best = j - lo
            }
        }
        return best
    }

    private fun movingAverage(x: DoubleArray, wIn: Int): DoubleArray {
        val w = wIn.coerceAtLeast(3)
        val out = DoubleArray(x.size)
        val half = w / 2
        for (i in x.indices) {
            val lo = maxOf(0, i - half)
            val hi = minOf(x.size, i + half + 1)
            var s = 0.0
            for (j in lo until hi) s += x[j]
            out[i] = s / (hi - lo)
        }
        return out
    }

    private fun median(x: DoubleArray): Double {
        val s = x.copyOf()
        s.sort()
        return if (s.size % 2 == 1) s[s.size / 2] else (s[s.size / 2 - 1] + s[s.size / 2]) / 2.0
    }

    private fun std(x: DoubleArray): Double {
        val m = x.average()
        var v = 0.0
        for (e in x) v += (e - m) * (e - m)
        return sqrt(v / x.size)
    }
}
