#!/usr/bin/env python3
"""
Voice cloning script using Coqui XTTS V2
Universal GPU support: NVIDIA (CUDA), AMD (ROCm), Apple Silicon (MPS)
With intelligent segmentation customized according to linguistic criteria
"""

import os
import re
import warnings
import torch
import numpy as np
import argparse
import sys
from TTS.api import TTS
from pydub import AudioSegment

# Import for automatic universal calibration
try:
    from universal_voice_calibrator import calibrate_voice_parameters
    UNIVERSAL_CALIBRATION_AVAILABLE = True
except ImportError:
    UNIVERSAL_CALIBRATION_AVAILABLE = False
    print("⚠️ Universal Voice Calibrator not available - default parameters will be used")

# Suppress torchaudio warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio")

def clean_text_from_random_characters(text):
    """
    Intelligent text cleaning from random characters while preserving valid punctuation.
    
    ✅ CLEANING STRATEGY:
    - Detects and removes random special characters between/in words
    - Preserves valid punctuation at the end of sentences and after commas
    - Corrects punctuation placed incorrectly in the middle of words
    - Normalizes multiple spaces and tabs
    
    Args:
        text: Original text with random characters
    
    Returns:
        str: Cleaned and corrected text
    """
    
    print("🧹 Cleaning text from random characters...")
    
    # Valid punctuation that can appear in normal text
    valid_punctuation = {'.', ',', ';', ':', '!', '?', '-', "'", '"', '(', ')', '[', ']'}
    
    # Completely invalid characters that should never exist
    completely_invalid = {'#', '$', '@', '%', '^', '*', '&', '§', '±', '~', '`', '|', '\\', '<', '>', '{', '}'}
    
    lines = text.split('\n')
    cleaned_lines = []
    
    for line in lines:
        if not line.strip():
            cleaned_lines.append(line)
            continue
        
        # Step 1: Remove completely invalid characters BUT preserve attached words
        # Ex: "*&may" becomes "may", "computer$#" becomes "computer"
        
        # Use regex to replace groups of invalid characters with space
        invalid_pattern = '[' + re.escape(''.join(completely_invalid)) + ']+'
        line = re.sub(invalid_pattern, ' ', line)
            
        # Step 2: Character by character processing for suspicious punctuation
        cleaned_chars = []
        words = re.split(r'(\s+)', line)  # Preserve spaces
        
        for word_or_space in words:
            if word_or_space.isspace():
                cleaned_chars.append(word_or_space)
                continue
                
            if not word_or_space.strip():
                continue
                
            # Word analysis for invalid punctuation
            cleaned_word = ""
            for i, char in enumerate(word_or_space):
                
                if char.isalnum():  # Letter or digit - preserve
                    cleaned_word += char
                    
                elif char in valid_punctuation:
                    # Check if punctuation is in its place
                    
                    # Case 1: Period, comma, etc at word end (OK)
                    if i == len(word_or_space) - 1 and char in {'.', ',', ';', ':', '!', '?'}:
                        cleaned_word += char
                        
                    # Case 2: Apostrophe in word (OK for contractions)
                    elif char == "'" and i > 0 and i < len(word_or_space) - 1:
                        cleaned_word += char
                        
                    # Case 3: Hyphen between words (OK)
                    elif char == '-' and i > 0 and i < len(word_or_space) - 1:
                        cleaned_word += char
                        
                    # Case 4: Parentheses and quotes (preserve if balanced)
                    elif char in {'(', ')', '[', ']', '"'}:
                        cleaned_word += char
                        
                    # Otherwise: punctuation in suspicious position - remove
                    # Ex: "!technology" becomes "technology"
                    else:
                        # If at beginning and seems to be incorrect punctuation, ignore
                        if i == 0 and char in {'.', ',', ';', ':', '!', '?'}:
                            pass  # Remove
                        # If in middle of word and not apostrophe/hyphen, remove
                        elif i > 0 and i < len(word_or_space) - 1 and char not in {"'", '-'}:
                            pass  # Remove
                        else:
                            cleaned_word += char
                            
                else:
                    # Unknown character - remove
                    pass
                    
            if cleaned_word:
                cleaned_chars.append(cleaned_word)
                
        # Line reconstruction
        cleaned_line = ''.join(cleaned_chars)
        
        # Step 3: Space normalization
        cleaned_line = re.sub(r'\s+', ' ', cleaned_line)  # Multiple spaces → 1 space
        cleaned_line = cleaned_line.strip()
        
        # Step 4: Basic punctuation correction
        # Ensure space after period, comma etc.
        cleaned_line = re.sub(r'([.!?])([A-Z])', r'\1 \2', cleaned_line)
        cleaned_line = re.sub(r'([,;:])([A-Za-z])', r'\1 \2', cleaned_line)
        
        if cleaned_line:
            cleaned_lines.append(cleaned_line)
    
    result = '\n'.join(cleaned_lines)
    
    return result

def analyze_text_sentiment_and_emotion(text):
    """
    Analyzes text sentiment and emotions for vocal expressiveness adjustment.
    
    ✅ SOPHISTICATED MULTI-CRITERIA ALGORITHM:
    - Contextual and combinatorial analysis, not just isolated keywords
    - Detects tone and intentions: urgent vs technical vs explanatory vs positive
    - Stratified criteria with prioritization and mutual exclusions
    - Evaluates grammatical structure and contextual indicators
    
    Args:
        text: Text to analyze
    
    Returns:
        dict: {
            'sentiment': 'urgent'|'positive'|'technical'|'explanatory'|'neutral',
            'temperature_adjust': float (+/-0.03),
            'length_penalty_adjust': float (+/-0.01), 
            'speed_adjust': float (+/-0.01),
            'confidence': float (0-1)
        }
    """
    
    text_lower = text.lower()
    words = text_lower.split()
    word_count = len(words)
    
    if word_count == 0:
        return {'sentiment': 'neutral', 'confidence': 0.0, 'temperature_adjust': 0.0, 'length_penalty_adjust': 0.0, 'speed_adjust': 0.0}
    
    # ================================================================================================
    # 🔴 CRITERION 1: URGENT/ALERT - Requires IMMEDIATE ACTION or ACTIVE CRITICAL SITUATION
    # ================================================================================================
    
    # Primary urgency indicators (immediate action required)
    urgent_primary = {
        'immediate', 'urgent', 'emergency', 'critical alert', 'action required', 
        'must act', 'now', 'immediately', 'asap', 'right away', 'quickly',
        'stop', 'cease', 'halt', 'prevent', 'block', 'disable'
    }
    
    # Active dangerous situation indicators (in progress)  
    urgent_active = {
        'under attack', 'being attacked', 'active threat', 'ongoing attack',
        'breached', 'compromised', 'infected', 'hacked', 'infiltrated',
        'detected intrusion', 'malware found', 'virus detected'
    }
    
    # Urgent action verbs
    urgent_verbs = {
        'respond', 'react', 'counter', 'mitigate', 'contain', 'isolate',
        'evacuate', 'shutdown', 'terminate', 'kill', 'abort'
    }
    
    # Temporal urgency indicators
    urgent_time = {
        'within minutes', 'within hours', 'before', 'deadline', 'expires',
        'limited time', 'time critical', 'time sensitive'
    }
    
    # ================================================================================================
    # 🟢 CRITERION 2: POSITIVE/CONFIDENT - Solutions, successes, assured safety
    # ================================================================================================
    
    # Success and achievement indicators
    positive_success = {
        'success', 'successful', 'achieved', 'accomplished', 'completed',
        'resolved', 'fixed', 'solved', 'works', 'working', 'effective'
    }
    
    # Safety and protection indicators
    positive_security = {
        'secure', 'secured', 'protected', 'safe', 'safety', 'reliable',
        'trusted', 'verified', 'authenticated', 'authorized', 'validated'
    }
    
    # Improvement and benefit indicators
    positive_improvement = {
        'improve', 'enhanced', 'optimized', 'upgraded', 'advanced',
        'excellent', 'superior', 'robust', 'strong', 'solid'
    }
    
    # Positive assurance verbs
    positive_assurance = {
        'ensure', 'guarantee', 'provide', 'deliver', 'maintain',
        'support', 'enable', 'facilitate', 'strengthen'
    }
    
    # ================================================================================================
    # 🔵 CRITERION 3: TECHNICAL/INFORMATIVE - System descriptions, processes, concepts
    # ================================================================================================
    
    # Technical system terms
    technical_systems = {
        'system', 'systems', 'architecture', 'infrastructure', 'platform',
        'framework', 'environment', 'network', 'server', 'database'
    }
    
    # Processes and methodologies
    technical_processes = {
        'protocol', 'algorithm', 'methodology', 'procedure', 'process',
        'implementation', 'configuration', 'setup', 'deployment', 'operation'
    }
    
    # Concepts and standards
    technical_concepts = {
        'specification', 'standard', 'compliance', 'requirement', 'criteria',
        'parameter', 'variable', 'function', 'module', 'component'
    }
    
    # Technical analysis and evaluation
    technical_analysis = {
        'analysis', 'evaluation', 'assessment', 'measurement', 'monitoring',
        'testing', 'validation', 'verification', 'audit', 'review'
    }
    
    # ================================================================================================
    # 🟡 CRITERION 4: EXPLANATORY/EDUCATIONAL - Explanations, examples, clarifications
    # ================================================================================================
    
    # Explanatory connectors
    explanatory_connectors = {
        'for example', 'such as', 'including', 'namely', 'specifically',
        'in particular', 'that is', 'i.e.', 'e.g.', 'for instance'
    }
    
    # Idea development connectors
    explanatory_development = {
        'furthermore', 'moreover', 'in addition', 'additionally', 'also',
        'besides', 'plus', 'as well as', 'along with', 'together with'
    }
    
    # Clarification connectors
    explanatory_clarification = {
        'essentially', 'basically', 'fundamentally', 'primarily', 'mainly',
        'generally', 'typically', 'usually', 'commonly', 'often'
    }
    
    # Logical connectors
    explanatory_logical = {
        'therefore', 'thus', 'consequently', 'as a result', 'hence',
        'because', 'since', 'due to', 'owing to', 'given that'
    }
    
    # ================================================================================================
    # ADVANCED SCORE CALCULATION WITH CONTEXTUAL ANALYSIS
    # ================================================================================================
    
    # Helper function for multi-word expression detection
    def count_multi_word_expressions(text_lower, expressions):
        count = 0
        for expr in expressions:
            if expr in text_lower:
                count += 1
        return count
    
    # Helper function for simple words (removes punctuation)
    def count_single_words(words, word_set):
        import string
        return sum(1 for word in words if word.strip(string.punctuation) in word_set)
    
    # URGENT SCORES - with advanced contextual analysis
    urgent_primary_score = count_multi_word_expressions(text_lower, urgent_primary) * 3.0  # Maximum weight
    urgent_active_score = count_multi_word_expressions(text_lower, urgent_active) * 2.5
    urgent_verbs_score = count_single_words(words, urgent_verbs) * 1.5
    urgent_time_score = count_multi_word_expressions(text_lower, urgent_time) * 2.0
    
    urgent_total = urgent_primary_score + urgent_active_score + urgent_verbs_score + urgent_time_score
    
    # POSITIVE SCORES
    positive_success_score = count_single_words(words, positive_success) * 2.0
    positive_security_score = count_single_words(words, positive_security) * 1.5
    positive_improvement_score = count_single_words(words, positive_improvement) * 1.3
    positive_assurance_score = count_single_words(words, positive_assurance) * 1.8
    
    positive_total = positive_success_score + positive_security_score + positive_improvement_score + positive_assurance_score
    
    # TECHNICAL SCORES
    technical_systems_score = count_single_words(words, technical_systems) * 1.5
    technical_processes_score = count_single_words(words, technical_processes) * 1.8
    technical_concepts_score = count_single_words(words, technical_concepts) * 1.3
    technical_analysis_score = count_single_words(words, technical_analysis) * 1.4
    
    technical_total = technical_systems_score + technical_processes_score + technical_concepts_score + technical_analysis_score
    
    # EXPLANATORY SCORES - combine multi-word detection and simple words
    explanatory_connectors_score = count_multi_word_expressions(text_lower, explanatory_connectors) * 2.5
    explanatory_development_score = count_multi_word_expressions(text_lower, explanatory_development) * 1.8
    explanatory_clarification_score = count_single_words(words, explanatory_clarification) * 1.5
    explanatory_logical_score = count_multi_word_expressions(text_lower, explanatory_logical) * 2.0
    
    # Also add simple word detection for logical and development (for 'therefore', 'furthermore' etc.)
    explanatory_logical_single = {'therefore', 'thus', 'consequently', 'hence', 'because', 'since'}
    explanatory_development_single = {'furthermore', 'moreover', 'additionally', 'also', 'besides', 'plus'}
    explanatory_logical_score += count_single_words(words, explanatory_logical_single) * 2.0
    explanatory_development_score += count_single_words(words, explanatory_development_single) * 1.8
    
    explanatory_total = explanatory_connectors_score + explanatory_development_score + explanatory_clarification_score + explanatory_logical_score
    
    # Normalization to text length
    urgent_score = urgent_total / word_count if word_count > 0 else 0
    positive_score = positive_total / word_count if word_count > 0 else 0
    technical_score = technical_total / word_count if word_count > 0 else 0
    explanatory_score = explanatory_total / word_count if word_count > 0 else 0
    
    # ================================================================================================
    # SOPHISTICATED DECISION LOGIC WITH MUTUAL EXCLUSIONS
    # ================================================================================================
    
    # Score dictionary creation
    scores = {
        'urgent': urgent_score,
        'positive': positive_score,
        'technical': technical_score,
        'explanatory': explanatory_score
    }
    
    # Find dominant sentiment
    dominant_sentiment = max(scores, key=scores.get)
    confidence = scores[dominant_sentiment]
    
    # ✅ SOPHISTICATED CRITERION: Hierarchical decision logic with contextual priority
    
    # PRIORITY 1: URGENT - only situations with explicit urgency/immediate action indicators
    if urgent_primary_score > 0 or urgent_active_score > 0:
        dominant_sentiment = 'urgent'
        confidence = urgent_score
    
    # PRIORITY 2: EXPLANATORY - explanatory connectors have high priority
    elif (explanatory_connectors_score > 0 or explanatory_logical_score > 0 or 
          explanatory_development_score > 0):
        dominant_sentiment = 'explanatory'  
        confidence = explanatory_score
    
    # PRIORITY 3: Compare positive vs technical for the rest
    elif positive_score > technical_score and positive_score >= 0.06:
        dominant_sentiment = 'positive'
        confidence = positive_score
    
    # PRIORITY 4: TECHNICAL - only if there are no explanatory connectors
    elif technical_score >= 0.08:
        dominant_sentiment = 'technical'
        confidence = technical_score
    
    # PRIORITY 5: NEUTRAL - if no criterion is met
    else:
        dominant_sentiment = 'neutral'
        confidence = 0.0
    
    # ================================================================================================
    # 🎭 SOPHISTICATED SENTIMENT → TTS PARAMETERS MAPPING
    # ================================================================================================
    
    # ✅ ADVANCED STRATEGY: Proportional adjustments with confidence + optimized parameters per sentiment
    
    # Calculate adjustment intensity based on confidence (0.0 - 1.0)
    # The higher the confidence, the more pronounced the adjustment
    confidence_factor = min(confidence, 0.8)  # Cap at 0.8 to avoid extremes
    
    def get_sentiment_adjustments(sentiment_type, confidence_factor):
        """
        Calculates dynamic TTS adjustments based on sentiment type and detection confidence.
        
        OPTIMIZED PRINCIPLES:
        - Proportional adjustments with confidence (more confidence = larger adjustments)
        - Calibrated parameters for each content type
        - Includes top_p and repetition_penalty for complete control
        - Maintains naturalness and avoids exaggerations
        """
        
        # Define base adjustments for each sentiment - CALIBRATED for speaker.wav
        # BASED ON: Pitch 128Hz, dynamics 12.8dB, MFCC variation 28.5, tempo 143.6 BPM
        base_adjustments = {
            
            # 🚨 URGENT: Expressive, alert - CALIBRATED for moderate masculine voice
            'urgent': {
                'temperature_base': +0.03,     # REDUCED: moderate dynamics from speaker.wav
                'length_penalty_base': -0.02,  # More alert and direct rhythm
                'speed_base': +0.002,         # MINIMALIST: almost imperceptible
                'top_p_base': +0.02,          # REDUCED: more consistent timbre
                'repetition_penalty_base': +0.15  # REDUCED: more natural articulation
            },
            
            # ✅ POSITIVE: Expressive but balanced - CALIBRATED for vocal control
            'positive': {
                'temperature_base': +0.03,     # REDUCED: expressive but controlled
                'length_penalty_base': -0.005, # Slightly more fluid
                'speed_base': +0.001,         # MINIMALIST: practically imperceptible
                'top_p_base': +0.02,          # REDUCED: more spectrally focused
                'repetition_penalty_base': +0.10  # REDUCED: naturalness
            },
            
            # 🔧 TECHNICAL: Controlled, clear - CALIBRATED for pitch stability
            'technical': {
                'temperature_base': -0.03,     # REDUCED: controlled without extremism
                'length_penalty_base': +0.015, # More constant and meticulous rhythm
                'speed_base': -0.002,         # MINIMALIST: subtly slower
                'top_p_base': -0.02,          # REDUCED: predictable but not rigid
                'repetition_penalty_base': -0.10  # REDUCED: consistent without too much variation
            },
            
            # 📖 EXPLANATORY: Educational expressive - CALIBRATED for natural rhythm
            'explanatory': {
                'temperature_base': +0.03,     # REDUCED: expressive but not dramatic
                'length_penalty_base': +0.02,  # Calmer, pedagogical rhythm
                'speed_base': -0.002,         # MINIMALIST: subtly slower
                'top_p_base': +0.015,         # REDUCED: controlled variation
                'repetition_penalty_base': +0.12  # REDUCED: variation for interest, not forcing
            },
            
            # ⚪ NEUTRAL: Base, natural parameters
            'neutral': {
                'temperature_base': 0.0,
                'length_penalty_base': 0.0,
                'speed_base': 0.0,
                'top_p_base': 0.0,
                'repetition_penalty_base': 0.0
            }
        }
        
        # Get base adjustments for detected sentiment
        if sentiment_type not in base_adjustments:
            sentiment_type = 'neutral'
        
        base = base_adjustments[sentiment_type]
        
        # Apply confidence factor for proportional adjustments
        # Use non-linear scaling function for more nuance
        scaling_factor = confidence_factor ** 0.7  # Root for smoother transition
        
        return {
            'temperature_adjust': base['temperature_base'] * scaling_factor,
            'length_penalty_adjust': base['length_penalty_base'] * scaling_factor,
            'speed_adjust': base['speed_base'] * scaling_factor,
            'top_p_adjust': base['top_p_base'] * scaling_factor,
            'repetition_penalty_adjust': base['repetition_penalty_base'] * scaling_factor
        }
    
    # Calculate final adjustments
    adjustments = get_sentiment_adjustments(dominant_sentiment, confidence_factor)
    
    # If confidence is too low, force neutral
    if confidence < 0.05:  # Minimum threshold for detection
        dominant_sentiment = 'neutral'
        adjustments = get_sentiment_adjustments('neutral', 0.0)
    
    result = {
        'sentiment': dominant_sentiment,
        'confidence': confidence,
        **adjustments
    }
    
    return result

def move_linking_words_to_next_segment(segments):
    """
    Moves prepositions and conjunctions from segment end to next segment beginning.
    
    ✅ INTONATION IMPROVEMENT:
    - Detects linking words at segment ends
    - Moves them to beginning of next segment for natural intonation
    - Avoids "suspended" intonation when sentence ends with preposition
    
    Args:
        segments: List of tuples [(segment_text, is_complete_sentence), ...]
    
    Returns:
        Optimized list of segments with linking words moved
    """
    
    # ✅ COMPOUND EXPRESSIONS (MAXIMUM priority - must be kept together)
    compound_expressions = [
        # Cause and effect expressions
        'due to', 'because of', 'as a result of', 'on account of', 'owing to',
        # Addition expressions
        'as well as', 'in addition to', 'along with', 'together with',
        # Purpose expressions
        'in order to', 'so as to', 'with a view to',
        # Time expressions
        'as soon as', 'as long as', 'by the time', 'at the same time',
        # Location expressions
        'in front of', 'on top of', 'at the back of', 'in the middle of',
        # Comparison expressions
        'as opposed to', 'in contrast to', 'compared to', 'in comparison with',
        # Other common expressions
        'such as', 'rather than', 'instead of', 'apart from', 'aside from'
    ]
    
    # Individual linking words (normal priority)
    single_linking_words = {
        'due', 'of', 'to', 'for', 'with', 'from', 'by', 'at', 'in', 'on', 'into', 'onto',
        'through', 'over', 'under', 'between', 'among', 'during', 'before', 'after',
        'since', 'until', 'within', 'without', 'beneath', 'beside', 'beyond', 'toward',
        'towards', 'upon', 'across', 'along', 'around', 'behind', 'below', 'above',
        # Common conjunctions
        'and', 'or', 'but', 'nor', 'so', 'yet', 'because', 'although', 'though',
        'while', 'whereas', 'since', 'unless', 'until', 'when', 'where', 'if',
        'that', 'which', 'who', 'whom', 'whose', 'what', 'how', 'why',
        # Articles and determiners
        'the', 'a', 'an', 'this', 'that', 'these', 'those', 'my', 'your', 'his',
        'her', 'its', 'our', 'their', 'some', 'any', 'each', 'every', 'all', 'both'
    }
    
    optimized_segments = []
    moves_made = 0
    
    for i in range(len(segments)):
        segment_text, is_complete = segments[i]
        words = segment_text.strip().split()
        
        if len(words) > 0 and i < len(segments) - 1:  # Not for the last segment
            
            # ✅ PRIORITY 1: Check CROSS-SEGMENT compound expressions
            # Check if current segment + beginning of next segment form a compound expression
            compound_found = False
            next_text, next_complete = segments[i + 1]
            next_words = next_text.strip().split()
            
            for expr in compound_expressions:
                expr_words = expr.split()
                
                # Check all possible cross-segment combinations
                for split_point in range(1, len(expr_words)):
                    # Part 1 of expression should be at end of current segment
                    part1 = " ".join(expr_words[:split_point])
                    # Part 2 of expression should be at beginning of next segment
                    part2 = " ".join(expr_words[split_point:])
                    
                    if (len(words) >= split_point and 
                        len(next_words) >= len(expr_words) - split_point):
                        
                        # Check if they match
                        current_ending = " ".join(words[-split_point:]).lower()
                        next_beginning = " ".join(next_words[:len(expr_words) - split_point]).lower()
                        
                        if current_ending == part1 and next_beginning == part2:
                            # Found! Move the part from current segment to next
                            moved_part = " ".join(words[-split_point:])
                            current_segment = " ".join(words[:-split_point])
                            
                            # Update segments
                            optimized_segments.append((current_segment, False))
                            enhanced_next = moved_part + " " + next_text
                            segments[i + 1] = (enhanced_next, next_complete)
                            
                            print(f"  🔥 Cross-segment compound expression reunited: '{moved_part}' + '{part2}' = '{expr}' (segment {i+1} → {i+2})")
                            moves_made += 1
                            compound_found = True
                            break
                
                if compound_found:
                    break
            
            # ✅ PRIORITY 2: Check COMPLETE compound expressions in current segment
            if not compound_found and len(words) > 1:
                for expr in compound_expressions:
                    expr_words = expr.split()
                    if len(words) >= len(expr_words):
                        # Check if segment ends with this compound expression
                        segment_ending = " ".join(words[-len(expr_words):]).lower()
                        if segment_ending == expr:
                            # Move the entire compound expression
                            moved_expression = " ".join(words[-len(expr_words):])
                            current_segment = " ".join(words[:-len(expr_words)])
                            
                            # Update segments
                            optimized_segments.append((current_segment, False))
                            enhanced_next = moved_expression + " " + next_text
                            segments[i + 1] = (enhanced_next, next_complete)
                            
                            print(f"  🔄 Complete compound expression moved: '{moved_expression}' (segment {i+1} → {i+2})")
                            moves_made += 1
                            compound_found = True
                            break
            
            # ✅ PRIORITY 3: If no compound expression found, check individual words
            if not compound_found and len(words) > 1:
                last_word = words[-1].lower().rstrip('.,;:!?')  # Remove punctuation
                
                if last_word in single_linking_words:
                    # Move last word to next segment
                    moved_word = words[-1]  # Keep original punctuation if present
                    current_segment = " ".join(words[:-1])  # Segment without last word
                    
                    # Update current segment
                    optimized_segments.append((current_segment, False))  # No longer a complete sentence
                    
                    # Update next segment
                    enhanced_next = moved_word + " " + next_text
                    segments[i + 1] = (enhanced_next, next_complete)  # Modify directly in original list
                    
                    print(f"  🔄 Individual word moved: '{moved_word}' (segment {i+1} → {i+2})")
                    moves_made += 1
                    
                else:
                    optimized_segments.append((segment_text, is_complete))
            elif not compound_found:
                optimized_segments.append((segment_text, is_complete))
            
        else:
            optimized_segments.append((segment_text, is_complete))
    
    if moves_made == 0:
        print(f"  ✓ No prepositions/conjunctions found for moving in current segmentation")
    
    return optimized_segments


def intelligent_text_segmentation(text, max_chars=150, min_chars=50):
    """
    NEW STRATEGY - Segmentation on COMPLETE SENTENCES with preposition/conjunction optimization.
    
    PROBLEM SOLVED:
    - Intonation changes between segments because each segment is generated INDEPENDENTLY
    - Even within the same sentence, intonation is not consistent
    - Prepositions and conjunctions at segment ends create suspended intonation
    
    SOLUTION:
    - Split ONLY on complete sentences (. ! ?)
    - Increase max_chars to allow longer sentences
    - XTTS generates each sentence with UNIFIED intonation
    - ✅ NEW: Move prepositions/conjunctions to beginning of next segment
    - Result: consistent intonation + natural fluidity
    
    Returns:
        List of tuples: [(segment_text, is_complete_sentence), ...]
        - segment_text: actual text to generate
        - is_complete_sentence: True if it's a complete sentence
    """
    
    # Step 1: Split on complete sentences
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    segments = []  # List of (segment_text, is_complete_sentence) tuples
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        
        # Check sentence length
        if len(sentence) <= max_chars:
            # Complete sentence fits - IDEAL for consistent intonation
            segments.append((sentence, True))
        else:
            # Sentence too long - must split
            # Try splitting on comma/semicolon to preserve meaning
            pattern = r'(?<=[,;])\s+'
            chunks = re.split(pattern, sentence)
            
            current = ""
            for chunk in chunks:
                chunk = chunk.strip()
                if not chunk:
                    continue
                
                test = (current + " " + chunk).strip() if current else chunk
                
                if len(test) <= max_chars:
                    current = test
                else:
                    if current:
                        segments.append((current, False))  # Not a complete sentence
                        current = chunk
                    else:
                        # Single chunk too long - forced split
                        words = chunk.split()
                        mid = len(words) // 2
                        segments.append((" ".join(words[:mid]), False))
                        current = " ".join(words[mid:])
            
            if current:
                segments.append((current, False))
    
    # Post-processing: combine small segments
    i = 0
    while i < len(segments):
        segment_text, is_complete = segments[i]
        
        if len(segment_text) >= min_chars:
            i += 1
            continue
        
        # Segment too small
        if i + 1 < len(segments):
            next_text, next_complete = segments[i+1]
            segments[i] = (segment_text + " " + next_text, is_complete or next_complete)
            segments.pop(i+1)
        elif i > 0:
            prev_text, prev_complete = segments[i-1]
            segments[i-1] = (prev_text + " " + segment_text, prev_complete or is_complete)
            segments.pop(i)
            i = 0
        else:
            i += 1
    
    # ✅ POST-PROCESSING: Optimize prepositions/conjunctions for natural intonation
    print("  🔄 Optimizing prepositions/conjunctions...")
    optimized_segments = move_linking_words_to_next_segment(segments)
    
    return optimized_segments


def generate_segment_audio(tts, text_segment, is_complete_sentence, speaker_wav, language, device, temp_file, 
                          custom_speed=1.03, custom_temperature=0.78, custom_top_p=0.89, 
                          custom_repetition_penalty=2.7, custom_top_k=45, voice_analysis=None):
    """
    Generates audio for a text segment with ADAPTIVE EXPRESSIVENESS and naturalness.
    
    INTONATION STRATEGY:
    - For COMPLETE SENTENCES: enable split_sentences=True
      → XTTS manages intonation internally for maximum consistency
    - For incomplete fragments: split_sentences=False
      → Standard generation for fragments
    
    ADAPTIVE SPEED STRATEGY:
    - Short segments (<80 chars): lower speed (more natural, less rushed)
    - Medium segments (80-120 chars): normal speed
    - Long segments (>120 chars): slightly higher speed (for consistency)
    
    ✅ NATURAL EXPRESSIVENESS:
    - Sentiment analysis for TTS parameter adjustment
    - Urgent content → more expressive, alert rhythm
    - Technical content → more controlled, constant rhythm
    - Positive content → slightly more expressive
    - Subtle variations, not exaggerated
    
    ✅ ANTI-ROBOTIZATION:
    - Subtle micro-variations in temperature, speed, top_p
    - Variations imperceptible individually but cumulatively create naturalness
    - Avoid robotic monotony of identical parameters
    
    Args:
        tts: Loaded TTS model
        text_segment: Actual text segment to generate
        is_complete_sentence: True if it's a complete sentence (for intonation)
        speaker_wav: Reference audio file
        language: Language (e.g., "en")
        device: Computing device ("mps", "cpu", etc.)
        temp_file: Temporary file path for saving
    
    Returns:
        True if generation was successful, False otherwise
    """
    try:
        import random
        
        # 🎙️ ADAPTIVE PROCESSING BASED ON VOCAL ANALYSIS
        if voice_analysis:
            # Extract vocal features for fine adjustments
            pitch_info = voice_analysis['pitch']
            dynamics_info = voice_analysis['dynamics']
            spectral_info = voice_analysis['spectral']
            timbre_info = voice_analysis['timbre']
            
            # Fine adjustments based on voice characteristics
            # 1. Temperature adjustment based on vocal roughness (voice texture)
            if timbre_info['roughness'] > 0.05:  # Textured voice
                custom_temperature = min(1.2, custom_temperature + 0.05)  # More expressive for textures
            elif timbre_info['roughness'] < 0.02:  # Very smooth voice
                custom_temperature = max(0.7, custom_temperature - 0.02)  # More controlled
            
            # 2. top_p adjustment based on spectral brightness
            if spectral_info['spectral_centroid'] > 2500:  # Very bright voice
                custom_top_p = min(0.95, custom_top_p + 0.02)  # More diversity
            elif spectral_info['spectral_centroid'] < 1800:  # Very warm voice
                custom_top_p = max(0.85, custom_top_p - 0.02)  # More focused
                
            # 3. repetition_penalty adjustment based on pitch stability
            if pitch_info['pitch_stability'] > 0.15:  # Very stable voice
                custom_repetition_penalty = max(2.5, custom_repetition_penalty - 0.2)  # More natural
            elif pitch_info['pitch_stability'] < 0.08:  # Very variable voice
                custom_repetition_penalty = min(3.8, custom_repetition_penalty + 0.3)  # Clearer
                
            # 4. top_k adjustment based on vocal dynamics
            if dynamics_info['dynamic_range_db'] > 25:  # Very dynamic voice
                custom_top_k = min(70, custom_top_k + 8)  # More vocabulary
            elif dynamics_info['dynamic_range_db'] < 15:  # More monotone voice
                custom_top_k = max(35, custom_top_k - 5)  # More focused
        
        # STRATEGY: Use split_sentences=True for complete sentences
        # This allows XTTS to manage intonation internally
        
        # 🎯 UNIFORM + ULTRA-CONSISTENT SPEED for ANY TEXT TYPE
        # PROBLEM SOLVED: Descriptive texts (long sentences) sounded faster
        # SOLUTION: IDENTICAL base speed for all lengths → zero variation by text type
        segment_length = len(text_segment)
        
        # UNIFORM base from user arguments - ZERO variation by length!
        # Motivation: Eliminates speed differences between descriptive vs technical texts
        base_speed = custom_speed  # From command line arguments
        
        # 🎯 NATURAL TIMBRE OPTIMIZATION: UNIFORM SPEED + ANTI-SYNTHETIC PARAMETERS
        # PROBLEM: Variable speed → audible speed differences
        # PROBLEM: TTS parameters → artificial synthetic timbre
        # SOLUTION: Fixed speed + optimized parameters for natural timbre
        
        # Speed: COMPLETELY FIXED for uniform speed
        speed_adjust = 0.0  # ZERO variation in speed - this parameter remains fixed
        
        # NATURAL TIMBRE OPTIMIZATION - adjusted parameters to eliminate synthetic timbre
        import random
        if '--disable-stability' not in sys.argv:
            # NATURAL TIMBRE: specific adjustments to eliminate synthetic effect
            # Lower temperature for improved timbral consistency
            temperature_adjust = random.uniform(-0.015, -0.005)  # Tendency toward lower values
            # Optimized top_p for improved phonemic naturalness
            top_p_adjust = random.uniform(-0.010, 0.005)         # Slightly more restrictive
            # Reduced rep_penalty for natural fluidity
            repetition_penalty_adjust = random.uniform(-0.15, -0.05)  # Tendency toward lower values
        else:
            # MAXIMUM NATURALNESS: larger range for organic timbre
            temperature_adjust = random.uniform(-0.025, -0.010)  # Lower for naturalness
            top_p_adjust = random.uniform(-0.015, 0.000)         # Restrictive for clarity
            repetition_penalty_adjust = random.uniform(-0.25, -0.10)  # Much lower for fluidity
            
        sentiment = "natural_timbre_optimized"
        confidence = 0.3
        
        # ✅ CONFIGURABLE STABILITY: Variations based on user preferences
        # PROBLEM SOLVED: Random variations caused different results on each run
        # SOLUTION: Complete control over consistency vs naturalness
        
        # ✅ CALIBRATED PARAMETERS for voice in speaker.wav
        # BASED ON: Pitch 128Hz, dynamics 12.8dB, MFCC variation 28.5, tempo 143.6 BPM
        
        # 🎯 NATURAL TIMBRE OPTIMIZATION: FIXED SPEED + ANTI-SYNTHETIC PARAMETERS
        # SPEED: Completely fixed to eliminate speed differences
        # TTS PARAMETERS: Optimized specifically to eliminate synthetic timbre
        
        # Speed: COMPLETELY FIXED - main source of speed differences
        speed = custom_speed  # ZERO variation - speed remains constant
        
        # Temperature: REDUCED for natural timbral consistency (eliminates artificial variation)
        base_temperature = custom_temperature + temperature_adjust
        temperature = max(0.65, min(0.82, base_temperature))  # Narrow range for naturalness
        
        # Top_p: OPTIMIZED for phonemic clarity without artificiality
        base_top_p = custom_top_p + top_p_adjust  
        top_p = max(0.82, min(0.92, base_top_p))  # Adjusted range for natural timbre
        
        # Repetition penalty: REDUCED for natural fluidity (eliminates rigidity)
        base_repetition_penalty = custom_repetition_penalty + repetition_penalty_adjust
        repetition_penalty = max(2.2, min(2.6, base_repetition_penalty))  # Range for fluidity
        
        # NOTE: length_penalty is not supported by XTTS V2 and has been removed
        
        # STABILITY: No longer resetting seed to preserve consistency
        
        # 📊 Optimized logging - natural timbre + uniform speed
        emotion_info = f"NATURAL_TIMBRE (anti-synthetic optimization, conf={confidence:.1f})"
        param_info = f"temp={temperature:.3f}↓, speed={speed:.3f}(FIX), top_p={top_p:.3f}~, rep_pen={repetition_penalty:.1f}↓"
        
        # 🎭 OPTIMIZED TTS PARAMETERS for voice cloning + clear pronunciation
        # BASED ON: Pitch 128Hz, moderate dynamics, tempo 143.6 BPM + articulatory improvements
        # STRATEGY: Maximum fidelity + improved clarity for pronunciation (+1% to +12%)
        # NOTE: length_penalty not supported by XTTS V2 - removed to avoid warnings
        tts.tts_to_file(
            text=text_segment,
            speaker_wav=speaker_wav,
            language=language,
            file_path=temp_file,
            split_sentences=is_complete_sentence,  # ✅ TRUE for complete sentences
            temperature=temperature,               # ✅ CALIBRATED: 0.75 base for moderate dynamics
            repetition_penalty=repetition_penalty, # ✅ CALIBRATED: 2.5 base for natural articulation
            top_k=custom_top_k,                   # ✅ From user arguments for phonic diversity
            top_p=top_p,                          # ✅ CALIBRATED: 0.88 base for consistent timbre
            speed=speed                           # ✅ CALIBRATED: 1.05 perfect for tempo 143.6 BPM
        )
        
        # Debug info for natural timbre optimization (disabled)
        # print(f"    💭 DEBUG TIMBRE: {emotion_info}, temp={temperature:.3f}, speed={speed:.3f}, top_p={top_p:.3f}, rep_pen={repetition_penalty:.1f}")
        
        return True
    except Exception as e:
        print(f"  ✗ Error generating segment: {e}")
        return False


def concatenate_audio_segments(segment_files, output_file):
    """
    Concatenates audio segments with MICRO PAUSE and soft normalization for consistency.
    
    ✅ ANTI-ROBOTIZATION + INTONATION CONTINUITY:
    - 50ms micro pause between segments (vs 100ms crossfade which created intonation disconnect)
    - Soft normalization (50% adaptive + 50% original) to preserve natural dynamics
    - Allows intonation to "breathe" naturally without "new sentence" effect
    
    Args:
        segment_files: List of audio files to concatenate
        output_file: Final output file
    
    Returns:
        True if concatenation was successful, False otherwise
    """
    try:
        import librosa
        import numpy as np
        import soundfile as sf
        
        combined_audio = []
        target_rms = None  # Will be set at first segment
        sample_rate = None  # Track sample rate
        
        for i, segment_file in enumerate(segment_files):
            if os.path.exists(segment_file):
                # Load segment
                audio, sr = librosa.load(segment_file, sr=None, mono=True)
                
                # Save sample rate from first segment
                if sample_rate is None:
                    sample_rate = sr
                
                # Calculate RMS
                rms = np.sqrt(np.mean(audio**2))
                
                # Set target RMS at first segment
                if target_rms is None:
                    target_rms = rms
                
                # ✅ SOFT NORMALIZATION - preserves more natural dynamics
                # STRATEGY: 50% adaptive + 50% original (vs 70-30 previously)
                # → Less compression = more natural expressiveness
                if rms > 0:
                    normalization_factor = (target_rms / rms)
                    # Blend 50% normalized + 50% original for MAXIMUM naturalness
                    normalization_factor = 0.5 * normalization_factor + 0.5 * 1.0
                    audio = audio * normalization_factor
                
                # ✅ MICRO PAUSE between segments for intonation continuity
                # NEW STRATEGY: 50ms pause instead of crossfade
                # MOTIVATION: Crossfade mixes audio but doesn't solve the problem
                #             of intonation ending/starting as separate sentences
                # SOLUTION: Very short pause (50ms) allows intonation to "breathe"
                #           naturally without sounding disconnected
                if len(combined_audio) > 0:
                    # Add a 50ms pause (1200 samples at 24kHz)
                    pause_samples = int(0.05 * sample_rate)  # 50ms micro pause
                    silence = np.zeros(pause_samples)
                    combined_audio.append(silence)
                
                # Add current segment
                combined_audio.append(audio)
                
                # Delete temporary file after processing
                os.remove(segment_file)
        
        # Concatenate all segments
        if combined_audio and sample_rate is not None:
            final_audio = np.concatenate(combined_audio)
            
            # SOFT final normalization to prevent clipping
            # Conservative target 0.90 (vs 0.95) for more headroom
            peak = np.max(np.abs(final_audio))
            if peak > 0.90:
                final_audio = final_audio * (0.90 / peak)
            
            # Save
            sf.write(output_file, final_audio, sample_rate, subtype='PCM_16')
            return True
        else:
            return False
            
    except Exception as e:
        print(f"✗ Error concatenating segments: {e}")
        return False


def main():
    # Parse arguments to check stability option
    import sys
    
    # Quick check for --disable-stability option
    stability_disabled = '--disable-stability' in sys.argv
    
    if not stability_disabled:
        # 🎯 COMPLETE STABILITY: Set fixed seeds for perfect reproducibility
        # PROBLEM SOLVED: Different results on each run due to randomness
        # SOLUTION: Deterministic seeds → identical results for same input
        
        import random
        import numpy as np
        
        # Fixed seeds for all randomness sources
        STABILITY_SEED = 42  # Deterministic seed for consistency
        
        random.seed(STABILITY_SEED)           # Python random
        np.random.seed(STABILITY_SEED)        # NumPy random
        torch.manual_seed(STABILITY_SEED)     # PyTorch CPU
        
        if torch.cuda.is_available():
            torch.cuda.manual_seed(STABILITY_SEED)         # PyTorch CUDA
            torch.cuda.manual_seed_all(STABILITY_SEED)     # All GPUs
            
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            torch.mps.manual_seed(STABILITY_SEED)  # PyTorch MPS (Apple Silicon)
        
        # Settings for complete reproducibility
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        print("🎯 Stability enabled (reproducible results)")
    else:
        print("🎲 Stability disabled (natural variations)")
    print()
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="XTTS V2 Voice Cloning with custom intelligent segmentation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Usage examples:
  python3 generate_audio.py                                    # Auto-calibration active by default (recommended)
  python3 generate_audio.py my_voice.wav my_text.txt          # Custom voice + text with automatic calibration
  python3 generate_audio.py voice.wav text.txt -o clone.wav   # With custom output name + automatic calibration
  python3 generate_audio.py voice.wav text.txt --no-auto-calibrate # Disable calibration (manual parameters)
  python3 generate_audio.py --speed 1.3 --temperature 0.9 voice.wav text.txt # Manual parameters + automatic calibration
        """
    )
    
    # Main arguments
    parser.add_argument(
        'speaker_file',
        nargs='?',
        default='speaker.wav',
        help='Audio file with voice to clone (default: speaker.wav)'
    )
    
    parser.add_argument(
        'text_file',
        nargs='?',
        default='sample_text_for_clone.txt',
        help='File with text to clone (default: sample_text_for_clone.txt)'
    )
    
    parser.add_argument('--output', '-o', metavar='OUTPUT',
                       help='Output file name (default: auto-generated from text_file_path name)')
    
    # Speed argument
    parser.add_argument('--speed', '-s', type=float, default=1.03, metavar='SPEED',
                       help='Speech speed (0.5-2.0, default: 1.03 - fixed, not modified by auto-calibration)')
    
    # Arguments for intonation and expressiveness - SET BY AUTO-CALIBRATION OR MANUALLY
    parser.add_argument('--temperature', '-t', type=float, metavar='TEMP',
                       help='Vocal creativity (0.1-1.5) - auto-calibrated or set manually')
    
    parser.add_argument('--top_p', '-p', type=float, metavar='TOP_P',
                       help='Phonemic diversity (0.1-1.0) - auto-calibrated or set manually')
    
    parser.add_argument('--repetition_penalty', '-r', type=float, metavar='REP',
                       help='Repetition penalty (1.0-5.0) - auto-calibrated or set manually')
    
    parser.add_argument('--top_k', '-k', type=int, metavar='TOP_K',
                       help='Active vocabulary (10-100) - auto-calibrated or set manually')
    
    # Advanced options
    parser.add_argument('--no-auto-calibrate', action='store_true',
                       help='🔧 Disable automatic calibration (use manual/default parameters instead of vocal analysis)')
    
    parser.add_argument('--disable-stability', action='store_true',
                       help='🎲 Disable stability (allow random variations for naturalness, but different results on each run)')
    
    args = parser.parse_args()
    
    # 🎯 AUTO-CALIBRATION BY DEFAULT for new users
    # PROBLEM SOLVED: New users get generic unoptimized parameters
    # SOLUTION: Automatic calibration active by default for maximum fidelity
    auto_calibrate_enabled = not args.no_auto_calibrate and UNIVERSAL_CALIBRATION_AVAILABLE
    
    # Vocal analysis for intelligent audio processing
    voice_analysis_data = None
    if auto_calibrate_enabled:
        print("🎯 Auto-calibration active (analyzing voice...)")
        
        try:
            # MANDATORY AUTOMATIC CALIBRATION
            calibration_result = calibrate_voice_parameters(args.speaker_file, verbose=True)
            
            if calibration_result and calibration_result['confidence'] > 0.5:
                calibrated_params = calibration_result['parameters']
                
                # Detect parameters explicitly set through arguments (args.param is not None)
                manual_temperature = args.temperature is not None
                manual_top_p = args.top_p is not None
                manual_repetition_penalty = args.repetition_penalty is not None
                manual_top_k = args.top_k is not None
                manual_speed = args.speed != 1.03  # Speed keeps default 1.03
                
                # Apply calibration for parameters that were NOT explicitly set
                if not manual_temperature:
                    args.temperature = calibrated_params['temperature']
                if not manual_top_p:
                    args.top_p = calibrated_params['top_p'] 
                if not manual_repetition_penalty:
                    args.repetition_penalty = calibrated_params['repetition_penalty']
                if not manual_top_k:
                    args.top_k = calibrated_params['top_k']
                # Speed always remains manual (not auto-calibrated)
                
                voice_analysis_data = calibration_result['voice_analysis']
                
                print()
                print(f"✓ Voice type: {calibration_result['voice_type']} (confidence: {calibration_result['confidence']:.0%})")
                
                # Show what was calibrated vs what was kept manual
                calibrated_list = []
                manual_list = []
                
                if not manual_temperature:
                    calibrated_list.append("temp")
                else:
                    manual_list.append(f"temp={args.temperature}")
                    
                if not manual_top_p:
                    calibrated_list.append("top_p")
                else:
                    manual_list.append(f"top_p={args.top_p}")
                    
                if not manual_repetition_penalty:
                    calibrated_list.append("rep_penalty")
                else:
                    manual_list.append(f"rep_penalty={args.repetition_penalty}")
                    
                if not manual_top_k:
                    calibrated_list.append("top_k")
                else:
                    manual_list.append(f"top_k={args.top_k}")
                
                manual_list.append(f"speed={args.speed}")  # Speed is always manual
                
                if calibrated_list:
                    print(f"  Auto-calibrated: {', '.join(calibrated_list)}")
                if manual_list:
                    print(f"  Manual: {', '.join(manual_list)}")
                print()
            else:
                print("⚠️ Auto-calibration failed - using manual parameters")
                
        except Exception as e:
            print(f"❌ Calibration error: {e}")
            
            # Check if all TTS parameters are manually set
            all_manual = all([
                args.temperature is not None,
                args.top_p is not None,
                args.repetition_penalty is not None,
                args.top_k is not None
            ])
            
            if all_manual:
                print("✅ All TTS parameters manually set - continuing")
                print(f"   Temperature: {args.temperature}")
                print(f"   Top_p: {args.top_p}")
                print(f"   Repetition_penalty: {args.repetition_penalty}")
                print(f"   Top_k: {args.top_k}")
            else:
                print("🔄 EROARE CRITICĂ: Fără auto-calibrare nu pot seta parametrii TTS!")
                print("💡 Auto-calibrarea este OBLIGATORIE sau setați TOȚI parametrii manual:")
                missing = []
                if args.temperature is None: missing.append("--temperature")
                if args.top_p is None: missing.append("--top_p")
                if args.repetition_penalty is None: missing.append("--repetition_penalty")
                if args.top_k is None: missing.append("--top_k")
                print(f"   Parametri lipsă: {', '.join(missing)}")
                sys.exit(1)
    else:
        # User has explicitly disabled auto-calibration - ERROR!
        print("� === EROARE: CALIBRARE AUTOMATĂ DEZACTIVATĂ ===")
        print("❌ Auto-calibrarea este OBLIGATORIE în noua versiune!")
        print("💡 Parametrii TTS nu mai au valori default - trebuie calibrați automat")
        print("🔄 Eliminați --no-auto-calibrate sau setați parametrii manual explicit")
        print("   Exemplu: --temperature 0.8 --top_p 0.9 --repetition_penalty 2.5 --top_k 50")
        sys.exit(1)
    
    
    # Validate arguments (only for parameters that have been set)
    if not (0.5 <= args.speed <= 2.0):
        print(f"❌ Eroare: Viteza ({args.speed}) trebuie să fie între 0.5 și 2.0")
        print("💡 Sugestie: Folosește valori între 0.8 (foarte lent) și 1.5 (foarte rapid)")
        sys.exit(1)
    
    if args.temperature is not None and not (0.1 <= args.temperature <= 1.5):
        print(f"❌ Eroare: Temperature ({args.temperature}) trebuie să fie între 0.1 și 1.5")
        print("💡 Sugestie: 0.7-0.9 pentru voce naturală, 0.5-0.7 pentru claritate maximă")
        sys.exit(1)
    
    if args.top_p is not None and not (0.1 <= args.top_p <= 1.0):
        print(f"❌ Eroare: Top_p ({args.top_p}) trebuie să fie între 0.1 și 1.0")
        print("💡 Sugestie: 0.85-0.95 pentru diversitate fonemică optimă")
        sys.exit(1)
    
    if args.repetition_penalty is not None and not (1.0 <= args.repetition_penalty <= 5.0):
        print(f"❌ Eroare: Repetition_penalty ({args.repetition_penalty}) trebuie să fie între 1.0 și 5.0")
        print("💡 Sugestie: 2.0-3.5 pentru evitarea repetițiilor fără afectarea fluenței")
        sys.exit(1)
    
    if args.top_k is not None and not (10 <= args.top_k <= 100):
        print(f"❌ Eroare: Top_k ({args.top_k}) trebuie să fie între 10 și 100")
        print("💡 Sugestie: 40-60 pentru vocabular diversificat fără pierderea consistenței")
        sys.exit(1)
    
    # 0. Define source and destination files BEFORE display
    speaker_wav = args.speaker_file  # USE ARGUMENT FROM COMMAND LINE
    text_file_path = args.text_file  # USE ARGUMENT FROM COMMAND LINE
    
    # Generate output file name
    if args.output:
        output_file = args.output
    else:
        # Auto-generate based on text file name
        base_name = os.path.splitext(os.path.basename(text_file_path))[0]
        output_file = f"cloned_{base_name}.wav"
    
    print("=" * 80)
    print("🎙️  XTTS V2 Voice Cloning")
    print(f"📂 {speaker_wav} → {text_file_path} → {output_file}")
    print(f"⚙️  Speed={args.speed}, Temp={args.temperature}, Top_p={args.top_p}, Rep_penalty={args.repetition_penalty}, Top_k={args.top_k}")
    print("=" * 80)
    
    # 1. Detect and set optimal computing device (universal GPU support)
    # Priority: CUDA (NVIDIA) → ROCm (AMD) → MPS (Apple Silicon) → CPU
    if torch.cuda.is_available():
        device = "cuda"
        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else "Unknown"
        print(f"✓ NVIDIA GPU detected: {gpu_name}")
        print(f"✓ Running on: {device}")
    elif hasattr(torch.version, 'hip') and torch.version.hip is not None:
        # AMD ROCm support (requires PyTorch built with ROCm)
        device = "cuda"  # ROCm uses CUDA API in PyTorch
        print(f"✓ AMD GPU detected (ROCm)")
        print(f"✓ Running on: {device}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = "mps"
        print(f"✓ Apple Silicon GPU detected (MPS)")
        print(f"✓ Running on: {device}")
    else:
        device = "cpu"
        print(f"⚠ No GPU detected, using CPU")
        print(f"  (Note: CPU processing will be significantly slower)")
    
    print("-" * 80)
    
    # 2. Continue setup
    
    temp_dir = "temp_segments"
    
    # Create temporary directory for segments
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    
    # Check file existence
    if not os.path.exists(speaker_wav):
        print(f"✗ Voice file not found: {speaker_wav}")
        return
    
    if not os.path.exists(text_file_path):
        print(f"✗ Text file not found: {text_file_path}")
        return
    
    print(f"✓ Files: {speaker_wav}, {text_file_path} → {output_file}")
    print("-" * 80)
    
    # 3. Load XTTS V2 model
    print("📥 Loading XTTS V2 model...")
    
    try:
        tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2").to(device)
        print("✓ Model loaded")
    except Exception as e:
        print(f"✗ Model loading error: {e}")
        return
    
    print("-" * 80)
    
    # 4. Read text from file
    print(f"📄 Reading text from {text_file_path}...")
    try:
        with open(text_file_path, 'r', encoding='utf-8') as f:
            raw_text_content = f.read().strip()
        
        print(f"✓ Loaded {len(raw_text_content)} characters")
        
        # ✅ INTELLIGENT CLEANING: Removes random characters, preserves valid punctuation
        text_content = clean_text_from_random_characters(raw_text_content)
        
        if len(text_content) != len(raw_text_content):
            print(f"✓ Cleaned to {len(text_content)} characters")
        
    except Exception as e:
        print(f"✗ Error reading text: {e}")
        return
    
    print("-" * 80)
    
    # 5. Intelligent text segmentation
    print("📝 Intelligent text segmentation...")
    
    segments = intelligent_text_segmentation(text_content, max_chars=150, min_chars=50)
    
    print(f"✓ {len(segments)} segments (min={min(len(s[0]) for s in segments)}, "
          f"max={max(len(s[0]) for s in segments)}, "
          f"avg={sum(len(s[0]) for s in segments)//len(segments)} chars)")
    print("-" * 80)
    
    # 6. Generate audio segment by segment
    print(f"🎵 Generating audio for {len(segments)} segments...")
    print()
    
    segment_files = []
    
    for i, (segment_text, is_complete_sentence) in enumerate(segments, 1):
        segment_preview = segment_text[:60] + "..." if len(segment_text) > 60 else segment_text
        sentence_marker = "✓" if is_complete_sentence else "~"
        
        print(f"[{i}/{len(segments)}] {sentence_marker} {segment_preview}")
        
        temp_file = os.path.join(temp_dir, f"segment_{i:04d}.wav")
        
        if generate_segment_audio(tts, segment_text, is_complete_sentence, speaker_wav, "en", device, temp_file,
                                 custom_speed=args.speed, custom_temperature=args.temperature,
                                 custom_top_p=args.top_p, custom_repetition_penalty=args.repetition_penalty,
                                 custom_top_k=args.top_k, voice_analysis=voice_analysis_data):
            segment_files.append(temp_file)
        else:
            print(f"  ✗ Segment {i} FAILED")
    
    print()
    print("-" * 80)
    
    # 7. Concatenate audio segments
    print(f"Concatenating {len(segment_files)} audio segments...")
    
    if concatenate_audio_segments(segment_files, output_file):
        print("✓ Segments concatenated successfully")
        
        # Clean temporary directory
        try:
            os.rmdir(temp_dir)
        except:
            pass
        
        print("=" * 80)
        print("✓ SUCCESS! Audio generated successfully!")
        print(f"✓ File saved: {output_file}")
        
        # Display generated file size
        if os.path.exists(output_file):
            file_size = os.path.getsize(output_file)
            file_size_mb = file_size / (1024 * 1024)
            print(f"✓ File size: {file_size_mb:.2f} MB")
        
        print("=" * 80)
    else:
        print("✗ ERROR concatenating segments")
        print("=" * 80)


if __name__ == "__main__":
    main()
