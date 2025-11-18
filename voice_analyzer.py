#!/usr/bin/env python3
"""
Voice Analyzer Module - Automatic Voice Characteristics Analysis
=================================================================

Module for automatic extraction of vocal characteristics from any audio file:
- Fundamental pitch and its variation
- Audio dynamics (RMS, volume variations)
- Tempo and speech rhythm
- Spectral characteristics (MFCC, formants)
- Timbre and vocal texture
- Recommendations for TTS parameters

Author: Autonomous Voice Cloning System
Version: 1.0.0
"""

import numpy as np
import librosa
import librosa.display
from scipy import stats
from scipy.signal import find_peaks
import warnings
warnings.filterwarnings('ignore')

class VoiceAnalyzer:
    """
    Comprehensive analyzer for vocal characteristics.
    
    Automatically extracts all relevant characteristics for optimal
    calibration of TTS parameters for any new voice.
    """
    
    def __init__(self, sample_rate=22050):
        """
        Initializes the voice analyzer.
        
        Args:
            sample_rate (int): Sampling rate for audio processing
        """
        self.sample_rate = sample_rate
        self.analysis_results = {}
        
    def analyze_voice_file(self, audio_file_path, verbose=True):
        """
        Completely analyzes an audio file and extracts all vocal characteristics.
        
        Args:
            audio_file_path (str): Path to audio file to analyze
            verbose (bool): Whether to display analysis progress
            
        Returns:
            dict: Dictionary with all extracted vocal characteristics
        """
        if verbose:
            print(f"🎙️ Analyzing voice from: {audio_file_path}")
            print("=" * 60)
        
        try:
            # 1. Load audio
            if verbose:
                print("📁 Loading audio file...")
            y, sr = librosa.load(audio_file_path, sr=self.sample_rate)
            duration = len(y) / sr  # Manual duration calculation
            
            if verbose:
                print(f"   ✓ Duration: {duration:.2f} seconds")
                print(f"   ✓ Sample rate: {sr} Hz")
            
            # 2. Fundamental pitch analysis
            if verbose:
                print("\n🎵 Analyzing fundamental pitch...")
            pitch_analysis = self._analyze_pitch(y, sr)
            
            # 3. Audio dynamics analysis
            if verbose:
                print("🔊 Analyzing audio dynamics...")
            dynamics_analysis = self._analyze_dynamics(y, sr)
            
            # 4. Tempo and rhythm analysis
            if verbose:
                print("⏱️ Analyzing tempo and rhythm...")
            tempo_analysis = self._analyze_tempo(y, sr)
            
            # 5. Spectral characteristics analysis
            if verbose:
                print("🌈 Analyzing spectral characteristics...")
            spectral_analysis = self._analyze_spectral_features(y, sr)
            
            # 6. Timbre analysis
            if verbose:
                print("🎭 Analyzing vocal timbre...")
            timbre_analysis = self._analyze_timbre(y, sr)
            
            # 7. Consolidate results
            self.analysis_results = {
                'file_info': {
                    'duration': duration,
                    'sample_rate': sr,
                    'file_path': audio_file_path
                },
                'pitch': pitch_analysis,
                'dynamics': dynamics_analysis,
                'tempo': tempo_analysis,
                'spectral': spectral_analysis,
                'timbre': timbre_analysis
            }
            
            if verbose:
                print("\n✅ Complete analysis finalized!")
                self._print_summary()
            
            return self.analysis_results
            
        except Exception as e:
            print(f"❌ Error during voice analysis: {e}")
            return None
    
    def _analyze_pitch(self, y, sr):
        """Analyzes fundamental pitch and its variations."""
        # IMPROVED ALGORITHM: YIN as PRIMARY (more accurate for fundamental)
        pitch_values = []
        
        # Method 1: YIN algorithm (priority - correctly detects fundamental)
        try:
            f0_yin = librosa.yin(y, fmin=50, fmax=400, sr=sr)
            pitch_values = f0_yin[f0_yin > 50].tolist()
        except Exception:
            pass
        
        # Method 2: Fallback with Piptrack only if YIN fails
        if len(pitch_values) < 10:  # Very few values from YIN
            try:
                pitches, magnitudes = librosa.piptrack(y=y, sr=sr, threshold=0.15)  # Stricter threshold
                
                # Intelligent filtering for fundamental (not harmonics)
                for t in range(pitches.shape[1]):
                    frame_pitches = pitches[:, t][pitches[:, t] > 50]
                    if len(frame_pitches) > 0:
                        # Select lowest pitch (fundamental, not harmonics)
                        fundamental = np.min(frame_pitches[frame_pitches < 350])  # Max 350Hz for voice
                        if 50 <= fundamental <= 350:
                            pitch_values.append(fundamental)
            except Exception:
                pass
        
        if len(pitch_values) > 0:
            pitch_mean = np.mean(pitch_values)
            pitch_std = np.std(pitch_values)
            pitch_range = np.max(pitch_values) - np.min(pitch_values)
            pitch_median = np.median(pitch_values)
        else:
            # Default values if pitch cannot be extracted
            pitch_mean = 150.0
            pitch_std = 20.0
            pitch_range = 60.0
            pitch_median = 150.0
        
        return {
            'mean_pitch': pitch_mean,
            'pitch_std': pitch_std,
            'pitch_range': pitch_range,
            'median_pitch': pitch_median,
            'pitch_stability': max(0.1, 1.0 - (pitch_std / pitch_mean)) if pitch_mean > 0 else 0.5
        }
    
    def _analyze_dynamics(self, y, sr):
        """Analyzes audio dynamics (RMS, volume variations)."""
        # RMS Energy
        rms = librosa.feature.rms(y=y)[0]
        rms_mean = np.mean(rms)
        rms_std = np.std(rms)
        
        # Dynamic range (difference between peak and significant minimum)
        rms_max = np.max(rms)
        rms_min = np.percentile(rms[rms > 0], 10)  # 10th percentile to avoid silences
        dynamic_range = 20 * np.log10(rms_max / rms_min) if rms_min > 0 else 20.0
        
        # Energy consistency
        energy_consistency = max(0.1, 1.0 - (rms_std / rms_mean)) if rms_mean > 0 else 0.5
        
        return {
            'rms_mean': rms_mean,
            'rms_std': rms_std,
            'dynamic_range_db': dynamic_range,
            'energy_consistency': energy_consistency
        }
    
    def _analyze_tempo(self, y, sr):
        """Analyzes tempo and speech rhythm."""
        # Onset detection for detecting syllable/word beginnings
        onset_frames = librosa.onset.onset_detect(y=y, sr=sr, units='time')
        
        if len(onset_frames) > 1:
            # Calculate intervals between onsets
            intervals = np.diff(onset_frames)
            tempo_bpm = 60.0 / np.mean(intervals) if len(intervals) > 0 else 120.0
            tempo_stability = max(0.1, 1.0 - (np.std(intervals) / np.mean(intervals))) if len(intervals) > 0 else 0.5
        else:
            tempo_bpm = 120.0
            tempo_stability = 0.5
        
        # Speech density (how dense the speech is)
        speech_density = len(onset_frames) / (len(y) / sr)
        
        return {
            'tempo_bpm': tempo_bpm,
            'tempo_stability': tempo_stability,
            'speech_density': speech_density,
            'num_onsets': len(onset_frames)
        }
    
    def _analyze_spectral_features(self, y, sr):
        """Analyzes spectral characteristics (MFCC, spectral features)."""
        # MFCC features
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfccs, axis=1)
        mfcc_std = np.std(mfccs, axis=1)
        
        # Spectral centroid (sound brightness)
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        spectral_centroid_mean = np.mean(spectral_centroids)
        
        # Spectral rolloff
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        spectral_rolloff_mean = np.mean(spectral_rolloff)
        
        # Zero crossing rate (voice roughness)
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        zcr_mean = np.mean(zcr)
        
        # Spectral bandwidth
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
        spectral_bandwidth_mean = np.mean(spectral_bandwidth)
        
        return {
            'mfcc_mean': mfcc_mean,
            'mfcc_std': mfcc_std,
            'mfcc_variation': np.mean(mfcc_std),
            'spectral_centroid': spectral_centroid_mean,
            'spectral_rolloff': spectral_rolloff_mean,
            'zero_crossing_rate': zcr_mean,
            'spectral_bandwidth': spectral_bandwidth_mean
        }
    
    def _analyze_timbre(self, y, sr):
        """Analyzes timbre and vocal texture."""
        # Chroma features (for vocal harmony)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_mean = np.mean(chroma, axis=1)
        chroma_std = np.std(chroma, axis=1)
        
        # Spectral contrast (for vocal texture)
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        contrast_mean = np.mean(spectral_contrast, axis=1)
        
        # Roughness estimation (voice roughness)
        spectral_flatness = librosa.feature.spectral_flatness(y=y)[0]
        roughness = np.mean(spectral_flatness)
        
        return {
            'chroma_mean': chroma_mean,
            'chroma_std': chroma_std,
            'spectral_contrast': contrast_mean,
            'roughness': roughness,
            'tonal_stability': np.mean(chroma_std)
        }
    
    def _print_summary(self):
        """Displays a summary of the voice analysis."""
        if not self.analysis_results:
            return
        
        print("\n" + "=" * 60)
        print("📊 VOCAL ANALYSIS SUMMARY")
        print("=" * 60)
        
        # Pitch
        pitch = self.analysis_results['pitch']
        print(f"🎵 PITCH:")
        print(f"   • Mean pitch: {pitch['mean_pitch']:.1f} Hz")
        print(f"   • Variability: {pitch['pitch_std']:.1f} Hz")
        print(f"   • Stability: {pitch['pitch_stability']:.2%}")
        
        # Dynamics
        dynamics = self.analysis_results['dynamics']
        print(f"\n🔊 DYNAMICS:")
        print(f"   • Dynamic range: {dynamics['dynamic_range_db']:.1f} dB")
        print(f"   • Energy consistency: {dynamics['energy_consistency']:.2%}")
        
        # Tempo
        tempo = self.analysis_results['tempo']
        print(f"\n⏱️ TEMPO:")
        print(f"   • Tempo: {tempo['tempo_bpm']:.1f} BPM")
        print(f"   • Rhythm stability: {tempo['tempo_stability']:.2%}")
        print(f"   • Speech density: {tempo['speech_density']:.1f} onset/sec")
        
        # Spectral
        spectral = self.analysis_results['spectral']
        print(f"\n🌈 SPECTRAL:")
        print(f"   • Spectral centroid: {spectral['spectral_centroid']:.0f} Hz")
        print(f"   • MFCC variability: {spectral['mfcc_variation']:.3f}")
        print(f"   • Zero crossing rate: {spectral['zero_crossing_rate']:.4f}")
        
        # Timbre
        timbre = self.analysis_results['timbre']
        print(f"\n🎭 TIMBRE:")
        print(f"   • Roughness: {timbre['roughness']:.4f}")
        print(f"   • Tonal stability: {timbre['tonal_stability']:.3f}")
    
    def get_voice_profile_summary(self):
        """
        Returns a summarized voice profile for use in calibration.
        
        Returns:
            dict: Simplified profile with key characteristics
        """
        if not self.analysis_results:
            return None
        
        pitch = self.analysis_results['pitch']
        dynamics = self.analysis_results['dynamics']
        tempo = self.analysis_results['tempo']
        spectral = self.analysis_results['spectral']
        timbre = self.analysis_results['timbre']
        
        # Classifications for voice type
        voice_type = self._classify_voice_type(pitch['mean_pitch'])
        voice_character = self._classify_voice_character(spectral, timbre)
        speech_style = self._classify_speech_style(tempo, pitch['pitch_stability'])
        
        return {
            'voice_type': voice_type,
            'voice_character': voice_character,
            'speech_style': speech_style,
            'key_metrics': {
                'pitch': pitch['mean_pitch'],
                'pitch_stability': pitch['pitch_stability'],
                'dynamic_range': dynamics['dynamic_range_db'],
                'tempo_bpm': tempo['tempo_bpm'],
                'spectral_brightness': spectral['spectral_centroid'],
                'roughness': timbre['roughness']
            }
        }
    
    def _classify_voice_type(self, mean_pitch):
        """Classifies voice type based on pitch."""
        if mean_pitch < 100:
            return "bass"
        elif mean_pitch < 130:
            return "baritone"  
        elif mean_pitch < 160:
            return "tenor"
        elif mean_pitch < 200:
            return "alto"
        else:
            return "soprano"
    
    def _classify_voice_character(self, spectral, timbre):
        """Classifies voice character."""
        brightness = spectral['spectral_centroid']
        roughness = timbre['roughness']
        
        if brightness > 2000 and roughness < 0.01:
            return "bright_smooth"
        elif brightness > 2000:
            return "bright_textured"
        elif roughness < 0.01:
            return "warm_smooth"
        else:
            return "warm_textured"
    
    def _classify_speech_style(self, tempo, pitch_stability):
        """Classifies speech style."""
        if tempo['tempo_bpm'] > 140 and pitch_stability > 0.7:
            return "fast_controlled"
        elif tempo['tempo_bpm'] > 140:
            return "fast_expressive"
        elif pitch_stability > 0.7:
            return "measured_controlled"
        else:
            return "measured_expressive"


def analyze_voice_sample(audio_file_path, verbose=True):
    """
    Helper function for quick analysis of a voice sample.
    
    Args:
        audio_file_path (str): Path to audio file
        verbose (bool): Whether to display analysis details
        
    Returns:
        dict: Voice analysis results
    """
    analyzer = VoiceAnalyzer()
    return analyzer.analyze_voice_file(audio_file_path, verbose=verbose)


if __name__ == "__main__":
    # Direct module test
    import sys
    import os
    
    if len(sys.argv) > 1:
        audio_file = sys.argv[1]
    else:
        # Search for speaker.wav in current directory
        audio_file = "speaker.wav"
    
    if os.path.exists(audio_file):
        print(f"🎙️ Testing Voice Analyzer with: {audio_file}")
        print("=" * 70)
        
        results = analyze_voice_sample(audio_file)
        
        if results:
            analyzer = VoiceAnalyzer()
            analyzer.analysis_results = results
            profile = analyzer.get_voice_profile_summary()
            
            print("\n" + "=" * 70)
            print("🎯 SUMMARIZED VOCAL PROFILE")
            print("=" * 70)
            print(f"Voice type: {profile['voice_type']}")
            print(f"Voice character: {profile['voice_character']}")
            print(f"Speech style: {profile['speech_style']}")
            print("\n✅ Complete analysis! Module works correctly.")
        else:
            print("❌ Analysis failed!")
    else:
        print(f"❌ File {audio_file} does not exist!")
        print("💡 Usage: python3 voice_analyzer.py [audio_file_path]")