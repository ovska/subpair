from __future__ import annotations

import contextlib
import io
import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from subpair.cache import write_cache
from subpair.cli import _build_parser, main
from subpair.engine import SearchOptions, run_search
from subpair.modal import (
    ModalOptions,
    RoomMode,
    RoomModalSignature,
    T60_LN_RATIO,
    DIRECT_REFERENCE_WINDOW_SECONDS,
    _bandlimited_decimated_impulse,
    _cluster_candidates,
    _direct_reference_window_seconds,
    _eigenvalues_to_modes,
    _matrix_pencil_poles,
    _measurement_consensus_poles,
    _noise_floor_db,
    _prepare_segments,
    _windowed_segment,
    aggregate_modal_metrics,
    compute_pair_modal_metrics,
    estimate_room_poles,
    fit_mode_residues,
    mode_metrics,
    modal_robustness,
)


def _decaying_sinusoid(
    n: int, fs: float, frequency_hz: float, t60_s: float, amplitude: float, phase: float = 0.0
) -> np.ndarray:
    t = np.arange(n) / fs
    alpha = T60_LN_RATIO / t60_s
    return amplitude * np.exp(-alpha * t) * np.cos(2.0 * math.pi * frequency_hz * t + phase)


def _synthetic_modal_impulse(
    n: int,
    fs: float,
    modes: list[tuple[float, float, float]],
    noise_amplitude: float = 0.0,
    seed: int = 0,
) -> np.ndarray:
    """A causal impulse: unit spike at t=0 plus a sum of decaying sinusoids."""
    result = np.zeros(n, dtype=np.float64)
    result[0] = 1.0
    for frequency_hz, t60_s, amplitude in modes:
        result += _decaying_sinusoid(n, fs, frequency_hz, t60_s, amplitude)
    if noise_amplitude:
        rng = np.random.default_rng(seed)
        result += rng.normal(scale=noise_amplitude, size=n)
    return result


class MatrixPencilTests(unittest.TestCase):
    def test_recovers_a_single_known_mode_frequency_and_decay(self):
        fs = 500.0
        segment = _decaying_sinusoid(400, fs, 50.0, 0.4, 1.0)
        eigenvalues = _matrix_pencil_poles(segment, order=4, pencil_ratio=0.5)
        modes = _eigenvalues_to_modes(eigenvalues, fs, (18.0, 200.0))
        self.assertEqual(len(modes), 1)
        f, alpha = modes[0]
        self.assertAlmostEqual(f, 50.0, delta=0.5)
        expected_alpha = T60_LN_RATIO / 0.4
        self.assertAlmostEqual(alpha, expected_alpha, delta=expected_alpha * 0.1)

    def test_recovers_two_well_separated_modes(self):
        fs = 500.0
        segment = _decaying_sinusoid(400, fs, 35.0, 0.5, 1.0) + _decaying_sinusoid(
            400, fs, 90.0, 0.25, 0.6
        )
        eigenvalues = _matrix_pencil_poles(segment, order=8, pencil_ratio=0.5)
        modes = sorted(_eigenvalues_to_modes(eigenvalues, fs, (18.0, 200.0)))
        self.assertEqual(len(modes), 2)
        self.assertAlmostEqual(modes[0][0], 35.0, delta=0.5)
        self.assertAlmostEqual(modes[1][0], 90.0, delta=0.5)

    def test_filters_out_growing_or_unstable_poles(self):
        # A pole exactly on or outside the unit circle is not a decaying resonance.
        eigenvalues = np.array([1.0 + 0.0j, 1.2 * np.exp(1j * 0.5)])
        modes = _eigenvalues_to_modes(eigenvalues, 500.0, (18.0, 200.0))
        self.assertEqual(modes, [])

    def test_filters_out_of_band_poles(self):
        fs = 500.0
        # A mode centred well outside the analysis band should be dropped.
        segment = _decaying_sinusoid(400, fs, 5.0, 0.4, 1.0)
        eigenvalues = _matrix_pencil_poles(segment, order=4, pencil_ratio=0.5)
        modes = _eigenvalues_to_modes(eigenvalues, fs, (18.0, 200.0))
        self.assertEqual(modes, [])


class ClusteringTests(unittest.TestCase):
    def test_groups_close_candidates_and_separates_distinct_ones(self):
        candidates = [
            (0, 50.0, 10.0),
            (1, 50.2, 10.5),
            (2, 50.1, 9.8),
            (0, 90.0, 20.0),
            (1, 90.3, 21.0),
        ]
        groups = _cluster_candidates(candidates, freq_tolerance_hz=0.5, decay_tolerance_fraction=0.2)
        sizes = sorted(len(group) for group in groups)
        self.assertEqual(sizes, [2, 3])

    def test_does_not_merge_candidates_with_different_decay_rates(self):
        candidates = [(0, 50.0, 5.0), (1, 50.1, 50.0)]
        groups = _cluster_candidates(candidates, freq_tolerance_hz=0.5, decay_tolerance_fraction=0.2)
        self.assertEqual(len(groups), 2)


class PreprocessingTests(unittest.TestCase):
    def test_bandlimited_decimated_suppresses_out_of_band_content(self):
        source_fs = 4000.0
        n = 4096
        delay = 200  # leading silence, matching a real captured IR's loopback offset
        t = np.arange(n - delay) / source_fs
        impulse = np.zeros(n)
        # A strong 1 kHz tone, well outside the 18-200 Hz modal band.
        impulse[delay:] = 0.5 * np.sin(2.0 * math.pi * 1000.0 * t) * np.exp(-t / 0.05)
        decimated, actual_fs = _bandlimited_decimated_impulse(
            impulse, source_fs, (18.0, 200.0), 500.0
        )
        self.assertAlmostEqual(actual_fs, 500.0, delta=1.0)
        self.assertLess(np.max(np.abs(decimated)), 0.05)

    def test_bandlimiting_an_impulse_decays_quickly_rather_than_ringing(self):
        # A hard frequency-domain mask on a delta function rings almost
        # undamped (Gibbs); the smooth Butterworth bandpass used here should
        # not leave the whole window near its peak amplitude.
        source_fs = 4000.0
        impulse = np.zeros(4096)
        impulse[0] = 1.0
        decimated, actual_fs = _bandlimited_decimated_impulse(
            impulse, source_fs, (18.0, 200.0), 500.0
        )
        peak = float(np.max(np.abs(decimated)))
        tail = decimated[decimated.size // 2 :]
        self.assertLess(float(np.max(np.abs(tail))), 0.1 * peak)

    def test_windowed_segment_offsets_from_the_peak_and_reports_short_windows(self):
        fs = 500.0
        segment = np.zeros(100)
        segment[30] = 1.0
        windowed, achieved_seconds = _windowed_segment(segment, fs, peak_index=30, offset_seconds=0.0, duration_seconds=0.6)
        self.assertAlmostEqual(windowed[0], 1.0)
        self.assertLess(achieved_seconds, 0.6)
        offset_windowed, _ = _windowed_segment(segment, fs, peak_index=30, offset_seconds=0.02, duration_seconds=0.6)
        self.assertNotAlmostEqual(float(offset_windowed[0]) if offset_windowed.size else 0.0, 1.0)

    def test_noise_floor_is_low_for_a_clean_decaying_tone(self):
        fs = 500.0
        segment = _decaying_sinusoid(300, fs, 50.0, 0.4, 1.0)
        floor_db = _noise_floor_db(segment)
        self.assertLess(floor_db, -20.0)

    def test_direct_reference_window_spans_at_least_one_period_of_the_lowest_mode(self):
        # 1/18 Hz ~= 55.6 ms, longer than the 20 ms floor.
        self.assertAlmostEqual(
            _direct_reference_window_seconds((18.0, 200.0)), 1.0 / 18.0
        )
        # A higher band floor (e.g. 60 Hz, period ~16.7 ms) is already
        # shorter than the 20 ms floor, which must still win.
        self.assertAlmostEqual(
            _direct_reference_window_seconds((60.0, 200.0)),
            DIRECT_REFERENCE_WINDOW_SECONDS,
        )

    def test_prepare_segments_direct_window_widens_for_a_low_band_floor(self):
        fs = 4000.0
        impulse = _decaying_sinusoid(8000, fs, 30.0, 0.4, 1.0)
        options = ModalOptions(band=(18.0, 200.0), decimated_fs_hz=500.0)
        direct_segment, _, actual_fs, _ = _prepare_segments(impulse, fs, options)
        self.assertAlmostEqual(
            direct_segment.size / actual_fs, 1.0 / 18.0, delta=1.0 / actual_fs
        )


class ConsensusTests(unittest.TestCase):
    def test_order_sweep_retains_a_persistent_mode(self):
        fs = 500.0
        segment = _decaying_sinusoid(400, fs, 60.0, 0.3, 1.0)
        options = ModalOptions(order_min=4, order_max=20, order_step=2)
        retained, raw_count, kept_count = _measurement_consensus_poles(segment, fs, options)
        self.assertEqual(len(retained), 1)
        f, alpha, persistence = retained[0]
        self.assertAlmostEqual(f, 60.0, delta=0.5)
        self.assertGreaterEqual(persistence, options.order_persistence_fraction)
        self.assertLessEqual(kept_count, raw_count)

    def test_higher_orders_add_noise_poles_that_do_not_persist(self):
        fs = 500.0
        rng = np.random.default_rng(1)
        segment = _decaying_sinusoid(400, fs, 60.0, 0.3, 1.0) + rng.normal(
            scale=0.01, size=400
        )
        options = ModalOptions(order_min=4, order_max=40, order_step=4)
        retained, raw_count, kept_count = _measurement_consensus_poles(segment, fs, options)
        # The real mode should survive; spurious high-order noise poles should
        # not, so the discard fraction should be substantial.
        frequencies = [f for f, _, _ in retained]
        self.assertTrue(any(abs(f - 60.0) < 1.0 for f in frequencies))
        self.assertGreater(raw_count, kept_count)


class JointEstimationTests(unittest.TestCase):
    def _positions(
        self, fs: float, n: int, mode_amplitudes: list[float]
    ) -> list[tuple[str, np.ndarray]]:
        positions = []
        for index, amplitude in enumerate(mode_amplitudes, start=1):
            impulse = _synthetic_modal_impulse(n, fs, [(55.0, 0.35, amplitude)])
            positions.append((f"Position {index}", impulse))
        return positions

    def test_pools_a_pole_shared_by_every_measurement(self):
        fs = 500.0
        positions = self._positions(fs, 400, [1.0, 0.6, 0.8, 0.4])
        options = ModalOptions(order_min=4, order_max=20, order_step=2)
        signature = estimate_room_poles(positions, fs, options)
        self.assertTrue(signature.valid)
        self.assertEqual(len(signature.modes), 1)
        self.assertAlmostEqual(signature.modes[0].frequency_hz, 55.0, delta=0.5)
        self.assertEqual(len(signature.per_measurement), 4)
        for entry in signature.per_measurement:
            self.assertEqual(len(entry["modes"]), 1)
            self.assertEqual(entry["modes"][0]["pooled_index"], 0)

    def test_a_mode_present_in_only_one_measurement_is_not_pooled(self):
        fs = 500.0
        shared = _synthetic_modal_impulse(400, fs, [(55.0, 0.35, 1.0)])
        contaminated = _synthetic_modal_impulse(400, fs, [(55.0, 0.35, 1.0), (140.0, 0.35, 1.0)])
        positions = [
            ("Position 1", shared),
            ("Position 2", shared),
            ("Position 3", shared),
            ("Position 4", contaminated),
        ]
        options = ModalOptions(order_min=4, order_max=20, order_step=2)
        signature = estimate_room_poles(positions, fs, options)
        self.assertTrue(signature.valid)
        pooled_frequencies = [mode.frequency_hz for mode in signature.modes]
        self.assertTrue(any(abs(f - 55.0) < 1.0 for f in pooled_frequencies))
        self.assertFalse(any(abs(f - 140.0) < 1.0 for f in pooled_frequencies))

    def test_no_consistent_pole_marks_the_signature_invalid_with_a_warning(self):
        fs = 500.0
        rng = np.random.default_rng(3)
        positions = [
            (f"Position {index}", rng.normal(scale=0.02, size=200))
            for index in range(1, 4)
        ]
        options = ModalOptions(order_min=4, order_max=20, order_step=2)
        signature = estimate_room_poles(positions, fs, options)
        self.assertFalse(signature.valid)
        self.assertTrue(signature.warnings)

    def test_requires_at_least_two_measurements(self):
        with self.assertRaises(ValueError):
            estimate_room_poles([("Only one", np.zeros(200))], 500.0, ModalOptions())


class ResidueFitTests(unittest.TestCase):
    def test_recovers_known_amplitude_for_a_single_mode(self):
        fs = 500.0
        alpha = T60_LN_RATIO / 0.4
        segment = _decaying_sinusoid(400, fs, 55.0, 0.4, amplitude=2.0, phase=0.0)
        amplitudes, fit_r2 = fit_mode_residues(segment, fs, [(55.0, alpha)])
        self.assertAlmostEqual(amplitudes[0], 2.0, delta=0.05)
        self.assertGreater(fit_r2, 0.99)

    def test_recovers_amplitude_regardless_of_phase(self):
        fs = 500.0
        alpha = T60_LN_RATIO / 0.4
        segment = _decaying_sinusoid(400, fs, 55.0, 0.4, amplitude=1.5, phase=1.1)
        amplitudes, fit_r2 = fit_mode_residues(segment, fs, [(55.0, alpha)])
        self.assertAlmostEqual(amplitudes[0], 1.5, delta=0.05)
        self.assertGreater(fit_r2, 0.99)

    def test_poor_fit_for_the_wrong_pole_set(self):
        fs = 500.0
        alpha = T60_LN_RATIO / 0.4
        segment = _decaying_sinusoid(400, fs, 55.0, 0.4, amplitude=1.0)
        # Fit against a mode that was never in the signal at all.
        wrong_alpha = T60_LN_RATIO / 0.4
        _, fit_r2_wrong = fit_mode_residues(segment, fs, [(150.0, wrong_alpha)])
        _, fit_r2_right = fit_mode_residues(segment, fs, [(55.0, alpha)])
        self.assertLess(fit_r2_wrong, fit_r2_right)
        self.assertLess(fit_r2_wrong, 0.5)

    def test_empty_pole_set_returns_no_amplitudes(self):
        amplitudes, fit_r2 = fit_mode_residues(np.zeros(50), 500.0, [])
        self.assertEqual(amplitudes.size, 0)
        self.assertEqual(fit_r2, 0.0)


class ModeMetricsTests(unittest.TestCase):
    def test_hand_computed_t60_q_and_level(self):
        fs = 500.0
        t60 = 0.4
        alpha = T60_LN_RATIO / t60
        frequency_hz = 50.0
        amplitude = 1.0
        direct_reference = 2.0  # mode is 6.02 dB below the direct sound
        modes = mode_metrics(
            [(frequency_hz, alpha)],
            np.array([amplitude]),
            direct_reference,
            noise_floor_db=-60.0,
            options=ModalOptions(noise_floor_margin_db=10.0, audible_margin_db=20.0),
        )
        self.assertEqual(len(modes), 1)
        mode = modes[0]
        self.assertAlmostEqual(mode["t60_s"], t60, places=6)
        expected_q = math.pi * frequency_hz * t60 / T60_LN_RATIO
        self.assertAlmostEqual(mode["q"], expected_q, places=6)
        expected_level_db = 20.0 * math.log10(amplitude / direct_reference)
        self.assertAlmostEqual(mode["level_db"], expected_level_db, places=6)
        # t_audible: time for level_db - (20*alpha/ln10)*t to reach -20 dB.
        decline_db_per_s = 20.0 * alpha / math.log(10.0)
        expected_t_audible = (expected_level_db + 20.0) / decline_db_per_s
        self.assertAlmostEqual(mode["t_audible_s"], expected_t_audible, places=6)

    def test_a_mode_already_below_the_audible_margin_has_zero_audible_time(self):
        modes = mode_metrics(
            [(50.0, T60_LN_RATIO / 0.2)],
            np.array([0.001]),
            direct_reference=1.0,
            noise_floor_db=-80.0,
            options=ModalOptions(noise_floor_margin_db=10.0, audible_margin_db=20.0),
        )
        self.assertEqual(len(modes), 1)
        self.assertEqual(modes[0]["t_audible_s"], 0.0)

    def test_a_mode_near_the_noise_floor_is_discarded(self):
        modes = mode_metrics(
            [(50.0, T60_LN_RATIO / 0.3)],
            np.array([1.0]),
            direct_reference=1.0,
            noise_floor_db=-3.0,  # only 3 dB below the mode's own 0 dB level
            options=ModalOptions(noise_floor_margin_db=10.0),
        )
        self.assertEqual(modes, [])


class AggregateMetricsTests(unittest.TestCase):
    def _mode(self, frequency_hz, q, level_db, decay_rate_per_s, t_audible_s=0.0):
        return {
            "frequency_hz": frequency_hz,
            "decay_rate_per_s": decay_rate_per_s,
            "t60_s": T60_LN_RATIO / decay_rate_per_s,
            "q": q,
            "level_db": level_db,
            "t_audible_s": t_audible_s,
        }

    def test_counts_only_high_q_modes_above_the_level_gate(self):
        modes = [
            self._mode(50.0, 20.0, -10.0, 5.0),  # high Q, loud -> counted
            self._mode(60.0, 20.0, -30.0, 5.0),  # high Q, quiet -> not counted
            self._mode(70.0, 5.0, -5.0, 5.0),  # loud but low Q -> not counted
        ]
        options = ModalOptions(high_q_threshold=16.0, primary_gate_db=-20.0)
        aggregate = aggregate_modal_metrics(modes, options)
        self.assertEqual(aggregate["n_high_q"], 1)

    def test_reports_n_high_q_at_every_configured_gate(self):
        modes = [
            self._mode(50.0, 20.0, -18.0, 5.0),  # between -15 and -20
        ]
        options = ModalOptions(level_gates_db=(-15.0, -20.0), primary_gate_db=-20.0)
        aggregate = aggregate_modal_metrics(modes, options)
        self.assertEqual(aggregate["n_high_q_by_gate_db"]["-15"], 0)
        self.assertEqual(aggregate["n_high_q_by_gate_db"]["-20"], 1)

    def test_q_max_triple_identifies_the_worst_gated_offender(self):
        modes = [
            self._mode(50.0, 20.0, -5.0, 5.0),
            self._mode(60.0, 40.0, -5.0, 5.0),
        ]
        aggregate = aggregate_modal_metrics(modes, ModalOptions())
        self.assertAlmostEqual(aggregate["q_max"], 40.0)
        self.assertAlmostEqual(aggregate["q_max_triple"]["frequency_hz"], 60.0)

    def test_sum_modal_energy_increases_with_level_and_with_slower_decay(self):
        louder = [self._mode(50.0, 20.0, -5.0, 5.0)]
        quieter = [self._mode(50.0, 20.0, -15.0, 5.0)]
        slower_decay = [self._mode(50.0, 20.0, -5.0, 1.0)]
        options = ModalOptions()
        louder_energy = aggregate_modal_metrics(louder, options)["sum_modal_energy_db"]
        quieter_energy = aggregate_modal_metrics(quieter, options)["sum_modal_energy_db"]
        slower_energy = aggregate_modal_metrics(slower_decay, options)["sum_modal_energy_db"]
        self.assertGreater(louder_energy, quieter_energy)
        self.assertGreater(slower_energy, louder_energy)

    def test_no_gated_modes_yields_none_energy_and_zero_count(self):
        aggregate = aggregate_modal_metrics([], ModalOptions())
        self.assertEqual(aggregate["n_high_q"], 0)
        self.assertIsNone(aggregate["sum_modal_energy_db"])
        self.assertIsNone(aggregate["q_max_triple"])
        self.assertIsNone(aggregate["ringing_ms"])

    def test_ringing_ms_is_the_worst_case_time_audible_across_every_mode(self):
        modes = [
            self._mode(50.0, 20.0, -5.0, 5.0, t_audible_s=0.05),
            self._mode(60.0, 20.0, -5.0, 5.0, t_audible_s=0.30),
            self._mode(70.0, 20.0, -5.0, 5.0, t_audible_s=0.10),
        ]
        aggregate = aggregate_modal_metrics(modes, ModalOptions())
        self.assertAlmostEqual(aggregate["ringing_ms"], 300.0)

    def test_ringing_ms_counts_a_low_q_mode_not_gated_into_n_highq(self):
        # Q below the high-Q threshold, so it doesn't count toward n_highQ,
        # but it still rings audibly and must still set ringing_ms.
        modes = [self._mode(50.0, 5.0, -5.0, 5.0, t_audible_s=0.5)]
        aggregate = aggregate_modal_metrics(modes, ModalOptions(high_q_threshold=16.0))
        self.assertEqual(aggregate["n_high_q"], 0)
        self.assertAlmostEqual(aggregate["ringing_ms"], 500.0)

    def test_worst_mode_level_db_is_the_loudest_mode_regardless_of_audibility(self):
        # None of these cross the audibility margin (t_audible_s=0 for all),
        # so ringing_ms is 0 for every one of them -- worst_mode_level_db
        # must still distinguish the loudest (-22 dB) from the rest.
        modes = [
            self._mode(50.0, 20.0, -35.0, 5.0, t_audible_s=0.0),
            self._mode(60.0, 20.0, -22.0, 5.0, t_audible_s=0.0),
            self._mode(70.0, 20.0, -40.0, 5.0, t_audible_s=0.0),
        ]
        aggregate = aggregate_modal_metrics(modes, ModalOptions())
        self.assertEqual(aggregate["ringing_ms"], 0.0)
        self.assertAlmostEqual(aggregate["worst_mode_level_db"], -22.0)

    def test_worst_mode_level_db_is_none_when_no_modes_survive(self):
        aggregate = aggregate_modal_metrics([], ModalOptions())
        self.assertIsNone(aggregate["worst_mode_level_db"])


class PairMetricsAndRobustnessTests(unittest.TestCase):
    def _signature(self) -> RoomModalSignature:
        return RoomModalSignature(
            modes=(RoomMode(55.0, T60_LN_RATIO / 0.4, 1.0),),
            decimated_fs_hz=500.0,
            window_seconds=0.6,
            discard_fraction=0.0,
            per_measurement=(),
            warnings=(),
            valid=True,
        )

    def _impulse_for_amplitude(self, amplitude: float, fs: float, n: int) -> np.ndarray:
        return _synthetic_modal_impulse(n, fs, [(55.0, 0.4, amplitude)])

    def test_invalid_signature_short_circuits_pair_metrics(self):
        invalid = RoomModalSignature(
            modes=(),
            decimated_fs_hz=500.0,
            window_seconds=0.6,
            discard_fraction=1.0,
            per_measurement=(),
            warnings=("no poles",),
            valid=False,
        )
        impulse = self._impulse_for_amplitude(1.0, 4000.0, 2000)
        result = compute_pair_modal_metrics(invalid, impulse, 4000.0, ModalOptions())
        self.assertFalse(result["valid"])
        self.assertIn("no poles", result["warnings"])

    def test_pair_metrics_report_a_mode_at_the_fixed_pole(self):
        fs = 4000.0
        impulse = self._impulse_for_amplitude(1.0, fs, 3000)
        result = compute_pair_modal_metrics(self._signature(), impulse, fs, ModalOptions())
        self.assertTrue(result["valid"])
        self.assertEqual(len(result["modes"]), 1)
        self.assertAlmostEqual(result["modes"][0]["frequency_hz"], 55.0, delta=0.5)
        self.assertGreater(result["fixed_pole_fit_r2"], 0.9)
        self.assertFalse(result["fixed_pole_fit_flagged"])

    def test_level_db_uses_the_rms_of_the_direct_window_not_its_peak_sample(self):
        fs = 4000.0
        n = 3000
        signature = self._signature()
        options = ModalOptions()
        impulse = self._impulse_for_amplitude(1.0, fs, n)
        result = compute_pair_modal_metrics(signature, impulse, fs, options)

        direct_segment, fit_segment, actual_fs, _ = _prepare_segments(impulse, fs, options)
        poles = [(mode.frequency_hz, mode.decay_rate_per_s) for mode in signature.modes]
        amplitudes, _ = fit_mode_residues(fit_segment, actual_fs, poles)
        rms_reference = float(np.sqrt(np.mean(np.square(direct_segment))))
        peak_reference = float(np.max(np.abs(direct_segment)))
        expected_level_db = 20.0 * math.log10(float(amplitudes[0]) / rms_reference)
        peak_based_level_db = 20.0 * math.log10(float(amplitudes[0]) / peak_reference)

        self.assertAlmostEqual(result["modes"][0]["level_db"], expected_level_db, places=6)
        self.assertGreater(abs(result["modes"][0]["level_db"] - peak_based_level_db), 1.0)

    def test_robustness_reports_full_stability_for_a_flat_neighbourhood(self):
        fs = 4000.0
        n = 3000

        def sum_impulse(delay_ms: float, gain_db: float) -> np.ndarray:
            # A delay/gain-invariant excitation: the mode's level does not
            # meaningfully move, so n_highQ should be stable everywhere.
            return self._impulse_for_amplitude(1.0, fs, n)

        result = modal_robustness(
            self._signature(), sum_impulse, fs, nominal_delay_ms=0.0, nominal_gain_db=0.0,
            options=ModalOptions(),
        )
        self.assertTrue(result["valid"])
        self.assertEqual(result["neighbourhood_size"], 9)
        self.assertEqual(result["fraction_stable"], 1.0)

    def test_robustness_on_invalid_signature_is_invalid(self):
        invalid = RoomModalSignature(
            modes=(), decimated_fs_hz=500.0, window_seconds=0.6, discard_fraction=1.0,
            per_measurement=(), warnings=(), valid=False,
        )
        result = modal_robustness(
            invalid, lambda d, g: np.zeros(10), 4000.0, 0.0, 0.0, ModalOptions()
        )
        self.assertFalse(result["valid"])


class EngineIntegrationTests(unittest.TestCase):
    def _write_modal_cache(self, cache: "Path") -> None:
        sample_rate = 2000.0
        length = 3000
        delay = 200
        rows = []
        # A shared 50 Hz room mode at every position, at varying excitation
        # levels (what delay/gain/polarity are supposed to control), plus a
        # per-pair-distinguishing higher mode so search has something to rank.
        definitions = [
            (0.30, [(85.0, 0.15, 0.05)]),
            (0.55, [(95.0, 0.15, 0.05)]),
            (0.20, [(78.0, 0.15, 0.05)]),
            (0.45, [(105.0, 0.15, 0.05)]),
        ]
        for index, (mode_50_amplitude, extra_modes) in enumerate(definitions, start=1):
            impulse = _synthetic_modal_impulse(
                length, sample_rate, [(50.0, 0.3, mode_50_amplitude), *extra_modes]
            )
            rows.append(
                {
                    "source_index": index,
                    "title": f"Position {index}",
                    "uuid": f"uuid-{index}",
                    "sample_rate": sample_rate,
                    "start_time_seconds": -0.05,
                    "impulse": impulse,
                }
            )
        write_cache(cache, rows, {"test": True})

    def test_search_with_modal_enabled_serializes_a_valid_signature_and_per_pair_metrics(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cache = root / "cache"
            self._write_modal_cache(cache)
            results_path = cache / "search-results.json"
            result = run_search(
                cache,
                results_path,
                SearchOptions(
                    band=(25.0, 150.0),
                    delay_range_ms=(-1.0, 1.0, 1.0),
                    gain_range_db=(0.0, 0.0, 1.0),
                    ppo=24,
                    eq_bands=0,
                    modal=True,
                    modal_tiebreak=True,
                ),
            )
            self.assertEqual(result["format_version"], 24)
            signature = result["modal_signature"]
            self.assertIsNotNone(signature)
            self.assertTrue(signature["valid"])
            self.assertTrue(any(abs(mode["frequency_hz"] - 50.0) < 1.0 for mode in signature["modes"]))
            self.assertEqual(len(signature["per_measurement"]), 4)
            for pair in result["pairs"]:
                self.assertIn("modal", pair)
                self.assertTrue(pair["modal"]["valid"])
                self.assertIn("n_high_q", pair["modal"])
                self.assertIn("ringing_ms", pair["modal"])
                self.assertIn("robustness", pair["modal"])
                self.assertTrue(pair["modal"]["robustness"]["valid"])
                self.assertIn("post_eq_modal", pair)
                self.assertTrue(pair["post_eq_modal"]["valid"])
                # eq_bands=0 means no filters were fitted, so the post-EQ
                # impulse is just the raw sum plus headroom; ringing_ms is a
                # level-relative metric so headroom cancels out of it exactly.
                self.assertAlmostEqual(
                    pair["post_eq_modal"]["ringing_ms"], pair["modal"]["ringing_ms"]
                )
                self.assertTrue(pair["effective_tail_is_modal"])
                self.assertTrue(pair["post_eq_effective_tail_is_modal"])
                self.assertAlmostEqual(pair["effective_tail_ms"], pair["modal"]["ringing_ms"])
                self.assertAlmostEqual(
                    pair["post_eq_effective_tail_ms"], pair["post_eq_modal"]["ringing_ms"]
                )
                # The report/CLI "Tail" column reads *_effective_tail_db, not
                # the ms fields, whenever the source is modal.
                self.assertIn("worst_mode_level_db", pair["modal"])
                self.assertAlmostEqual(
                    pair["effective_tail_db"], pair["modal"]["worst_mode_level_db"]
                )
                self.assertAlmostEqual(
                    pair["post_eq_effective_tail_db"],
                    pair["post_eq_modal"]["worst_mode_level_db"],
                )
            # write_json (allow_nan=False) already succeeded above without
            # raising; reloading confirms every modal field round-trips.
            reloaded = json.loads(results_path.read_text())
            self.assertEqual(
                reloaded["pairs"][0]["modal"]["n_high_q"],
                result["pairs"][0]["modal"]["n_high_q"],
            )
            self.assertTrue(reloaded["settings"]["modal"]["enabled"])
            self.assertTrue(reloaded["settings"]["modal"]["tiebreak"])

    def test_effective_tail_falls_back_to_csd_tail_when_modal_is_off(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cache = root / "cache"
            self._write_modal_cache(cache)
            results_path = cache / "search-results.json"
            result = run_search(
                cache,
                results_path,
                SearchOptions(
                    band=(25.0, 150.0),
                    delay_range_ms=(-1.0, 1.0, 1.0),
                    gain_range_db=(0.0, 0.0, 1.0),
                    ppo=24,
                    eq_bands=0,
                ),
            )
            for pair in result["pairs"]:
                self.assertNotIn("modal", pair)
                self.assertFalse(pair["effective_tail_is_modal"])
                self.assertFalse(pair["post_eq_effective_tail_is_modal"])
                self.assertAlmostEqual(pair["effective_tail_ms"], pair["raw_tail_ms"])
                self.assertAlmostEqual(
                    pair["post_eq_effective_tail_ms"], pair["post_eq_tail_ms"]
                )
                self.assertIsNone(pair["effective_tail_db"])
                self.assertIsNone(pair["post_eq_effective_tail_db"])

    def test_post_eq_modal_reflects_the_fitted_eq_bank(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cache = root / "cache"
            self._write_modal_cache(cache)
            results_path = cache / "search-results.json"
            result = run_search(
                cache,
                results_path,
                SearchOptions(
                    band=(25.0, 150.0),
                    delay_range_ms=(-1.0, 1.0, 1.0),
                    gain_range_db=(0.0, 0.0, 1.0),
                    ppo=24,
                    eq_bands=4,
                    max_cut_db=18.0,
                    modal=True,
                ),
            )
            for pair in result["pairs"]:
                self.assertIn("post_eq_modal", pair)
                # Whether or not the fitter happened to touch the 50 Hz mode
                # for this pair, the post-EQ fit must independently succeed
                # (not crash, and report a well-defined fit quality) against
                # the same fixed room poles -- EQ can legitimately change how
                # well those fixed poles explain the post-EQ signal, so this
                # does not assert a fit-quality floor.
                self.assertTrue(pair["post_eq_modal"]["valid"])
                self.assertTrue(math.isfinite(pair["post_eq_modal"]["fixed_pole_fit_r2"]))

    def test_search_with_modal_disabled_omits_modal_fields(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cache = root / "cache"
            self._write_modal_cache(cache)
            results_path = cache / "search-results.json"
            result = run_search(
                cache,
                results_path,
                SearchOptions(
                    band=(25.0, 150.0),
                    delay_range_ms=(-1.0, 1.0, 1.0),
                    gain_range_db=(0.0, 0.0, 1.0),
                    ppo=24,
                    eq_bands=0,
                ),
            )
            self.assertIsNone(result["modal_signature"])
            self.assertFalse(result["settings"]["modal"]["enabled"])
            for pair in result["pairs"]:
                self.assertNotIn("modal", pair)

    def test_modal_tiebreak_requires_modal_enabled(self):
        with self.assertRaises(ValueError):
            SearchOptions(modal=False, modal_tiebreak=True)

    def test_report_renders_a_modal_section_when_present(self):
        from subpair.html_report import build_report

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cache = root / "cache"
            self._write_modal_cache(cache)
            results_path = cache / "search-results.json"
            run_search(
                cache,
                results_path,
                SearchOptions(
                    band=(25.0, 150.0),
                    delay_range_ms=(-1.0, 1.0, 1.0),
                    gain_range_db=(0.0, 0.0, 1.0),
                    ppo=24,
                    eq_bands=0,
                    modal=True,
                ),
            )
            output = root / "report.html"
            build_report(cache, results_path, output, top=2, limit=4, raw=True)
            html_text = output.read_text()
            self.assertIn("Modal analysis", html_text)
            self.assertIn("modal-pole-map", html_text)
            self.assertIn("modal-invariance-frequency", html_text)

    def test_report_without_modal_data_has_no_modal_section(self):
        from subpair.html_report import build_report

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cache = root / "cache"
            self._write_modal_cache(cache)
            results_path = cache / "search-results.json"
            run_search(
                cache,
                results_path,
                SearchOptions(
                    band=(25.0, 150.0),
                    delay_range_ms=(-1.0, 1.0, 1.0),
                    gain_range_db=(0.0, 0.0, 1.0),
                    ppo=24,
                    eq_bands=0,
                ),
            )
            output = root / "report.html"
            build_report(cache, results_path, output, top=2, limit=4, raw=True)
            html_text = output.read_text()
            self.assertNotIn("Modal analysis", html_text)


class CliIntegrationTests(unittest.TestCase):
    def test_modal_flags_default_off_and_validate_choices(self):
        parser = _build_parser()
        defaults = parser.parse_args(["search"])
        self.assertEqual(defaults.modal, "off")
        self.assertEqual(defaults.modal_tiebreak, "off")
        parsed = parser.parse_args(["search", "--modal", "on", "--modal-tiebreak", "on"])
        self.assertEqual(parsed.modal, "on")
        self.assertEqual(parsed.modal_tiebreak, "on")
        with self.assertRaises(SystemExit):
            parser.parse_args(["search", "--modal", "maybe"])

    def test_main_rejects_modal_tiebreak_without_modal(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cache = root / "cache"
            rows = []
            sample_rate = 2000.0
            length = 2000
            for index in range(1, 4):
                impulse = np.zeros(length)
                impulse[100 + index] = 1.0
                rows.append(
                    {
                        "source_index": index,
                        "title": f"Position {index}",
                        "uuid": f"uuid-{index}",
                        "sample_rate": sample_rate,
                        "start_time_seconds": 0.0,
                        "impulse": impulse,
                    }
                )
            write_cache(cache, rows, {"test": True})
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                exit_code = main(
                    [
                        "search",
                        "--cache",
                        str(cache),
                        "--modal-tiebreak",
                        "on",
                    ]
                )
            self.assertEqual(exit_code, 2)
            self.assertIn("modal_tiebreak requires modal", stderr.getvalue())

    def test_print_ranking_shows_db_tail_for_modal_sourced_pairs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cache = root / "cache"
            sample_rate = 2000.0
            length = 3000
            rows = []
            for index, amplitude in enumerate([0.30, 0.55, 0.20, 0.45], start=1):
                impulse = _synthetic_modal_impulse(
                    length, sample_rate, [(50.0, 0.3, amplitude)]
                )
                rows.append(
                    {
                        "source_index": index,
                        "title": f"Position {index}",
                        "uuid": f"uuid-{index}",
                        "sample_rate": sample_rate,
                        "start_time_seconds": -0.05,
                        "impulse": impulse,
                    }
                )
            write_cache(cache, rows, {"test": True})
            results_path = cache / "search-results.json"
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                exit_code = main(
                    [
                        "search",
                        "--cache",
                        str(cache),
                        "--band",
                        "25",
                        "150",
                        "--delay-range",
                        "-1",
                        "1",
                        "1",
                        "--gain-range",
                        "0",
                        "0",
                        "1",
                        "--eq-bands",
                        "0",
                        "--modal",
                        "on",
                        "--results",
                        str(results_path),
                    ]
                )
            self.assertEqual(exit_code, 0)
            output = stdout.getvalue()
            self.assertIn("dB", output)
            self.assertNotIn("Tail ms", output)


if __name__ == "__main__":
    unittest.main()
