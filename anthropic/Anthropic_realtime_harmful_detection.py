#!/usr/bin/env python3
"""
Real-Time Harmful Content Detection System for AI Safety

This module implements a comprehensive multi-layer harmful content detection system
designed for real-time AI safety applications. It combines keyword filtering,
semantic analysis, and contextual understanding to identify and mitigate harmful
content in AI-generated responses.

Key Features:
- Multi-layer detection pipeline (keyword → semantic → contextual)
- Real-time processing with low latency
- Adaptive threshold learning from feedback
- Support for multiple harm types (violence, hate speech, misinformation, etc.)
- Automatic content revision suggestions
- Async processing for scalability

Author: AI Safety Research Team
Date: 2024
"""

import asyncio
import re
import logging
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Any
from enum import Enum
import torch
import torch.nn.functional as F
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HarmType(Enum):
    """
    Enumeration of different types of harmful content that the system can detect.
    
    This classification system helps categorize different forms of harmful content
    so that appropriate mitigation strategies can be applied. Each harm type has
    specific detection patterns and threshold requirements.
    """
    VIOLENCE = "violence"                    # Physical violence, threats, weapons
    HATE_SPEECH = "hate_speech"             # Discriminatory language, harassment
    MISINFORMATION = "misinformation"        # False information, conspiracy theories
    PRIVACY_VIOLATION = "privacy_violation"  # Personal information exposure
    ILLEGAL_ACTIVITY = "illegal_activity"    # Criminal activities, illegal instructions
    SEXUAL_CONTENT = "sexual_content"        # Explicit sexual content

@dataclass
class DetectionResult:
    """
    Data class representing the result of harmful content detection.
    
    This class encapsulates all the information returned by the detection pipeline,
    including the type of harm detected, confidence level, and mitigation recommendations.
    
    Attributes:
        harm_type: The specific type of harmful content detected (None if no harm)
        confidence: Confidence score between 0.0 and 1.0 for the detection
        explanation: Human-readable explanation of the detection result
        should_block: Boolean indicating whether the content should be blocked
        suggested_revision: Optional safer alternative to the harmful content
    """
    harm_type: Optional[HarmType]           # Type of harm detected (None if safe)
    confidence: float                       # Detection confidence (0.0 to 1.0)
    explanation: str                        # Human-readable explanation
    should_block: bool                      # Whether to block the content
    suggested_revision: Optional[str] = None  # Safer alternative content

class MultiLayerHarmDetector:
    """
    Multi-layer harmful content detection system.
    
    This class implements a three-layer detection pipeline:
    1. Keyword filtering (fast, rule-based)
    2. Semantic analysis (ML-based toxicity detection)
    3. Context understanding (deep contextual analysis)
    
    The system uses adaptive thresholds that can be adjusted based on feedback
    to improve accuracy over time.
    """
    
    def __init__(self):
        """
        Initialize the multi-layer harm detector.
        
        Sets up three detection layers with appropriate models and thresholds.
        Uses GPU acceleration when available for better performance.
        """
        logger.info("Initializing MultiLayerHarmDetector...")
        
        # Layer 1: Keyword filtering (fastest, rule-based)
        # This layer provides immediate filtering for obvious harmful content
        self.keyword_filters = self._load_keyword_filters()
        logger.info("✓ Keyword filters loaded")
        
        # Layer 2: Semantic analysis models (medium speed, ML-based)
        # Uses pre-trained toxicity classification models for semantic understanding
        try:
            self.toxicity_classifier = pipeline(
                "text-classification",
                model="martin-ha/toxic-comment-model",
                device=0 if torch.cuda.is_available() else -1
            )
            logger.info("✓ Toxicity classifier loaded")
        except Exception as e:
            logger.warning(f"Failed to load toxicity classifier: {e}")
            self.toxicity_classifier = None
        
        # Layer 3: Context understanding (slower but most accurate)
        # Uses BERT-based models for deep contextual analysis
        try:
            self.context_analyzer = AutoModelForSequenceClassification.from_pretrained(
                "unitary/toxic-bert"
            )
            self.context_tokenizer = AutoTokenizer.from_pretrained("unitary/toxic-bert")
            logger.info("✓ Context analyzer loaded")
        except Exception as e:
            logger.warning(f"Failed to load context analyzer: {e}")
            self.context_analyzer = None
            self.context_tokenizer = None
        
        # Adaptive thresholds for different harm types
        # These thresholds determine when content should be blocked
        # Higher values = more conservative (fewer false positives)
        # Lower values = more sensitive (fewer false negatives)
        self.thresholds = {
            HarmType.VIOLENCE: 0.7,           # Moderate threshold for violence
            HarmType.HATE_SPEECH: 0.8,        # High threshold for hate speech
            HarmType.MISINFORMATION: 0.6,     # Lower threshold for misinformation
            HarmType.PRIVACY_VIOLATION: 0.9,  # Very high threshold for privacy
            HarmType.ILLEGAL_ACTIVITY: 0.85,  # High threshold for illegal content
            HarmType.SEXUAL_CONTENT: 0.75     # Moderate threshold for sexual content
        }
        
        logger.info("MultiLayerHarmDetector initialization complete")
        
    def _load_keyword_filters(self) -> Dict[HarmType, List[str]]:
        """
        Load keyword filters for different harm types.
        
        This method defines comprehensive keyword lists for each harm type.
        These keywords are used in the first layer of detection for fast filtering.
        
        Returns:
            Dictionary mapping harm types to lists of associated keywords
        """
        return {
            HarmType.VIOLENCE: [
                # Direct violence keywords
                "kill", "murder", "assault", "weapon", "bomb", "violence",
                "hurt", "harm", "attack", "fight", "destroy", "threaten",
                "beat", "stab", "shoot", "strangle", "torture", "abuse"
            ],
            HarmType.HATE_SPEECH: [
                # Discriminatory and hateful language
                "hate", "discriminate", "racist", "sexist", "bigot",
                "inferior", "worthless", "scum", "disgusting", "filth",
                "retard", "idiot", "stupid", "moron", "degenerate"
            ],
            HarmType.PRIVACY_VIOLATION: [
                # Personal information exposure
                "social security", "credit card", "password", "address",
                "phone number", "personal information", "ssn", "ssn number",
                "bank account", "routing number", "pin", "security code"
            ],
            HarmType.ILLEGAL_ACTIVITY: [
                # Criminal and illegal activities
                "drug dealing", "money laundering", "fraud", "theft",
                "hacking", "piracy", "smuggling", "counterfeit", "illegal",
                "criminal", "unlawful", "bribery", "extortion", "blackmail"
            ],
            HarmType.SEXUAL_CONTENT: [
                # Explicit sexual content
                "explicit", "pornographic", "sexual", "nude", "intimate",
                "adult content", "xxx", "porn", "naked", "sexually"
            ]
        }
    
    async def detect_harm(self, text: str, context: Optional[str] = None) -> DetectionResult:
        """
        Main detection pipeline with multiple layers.
        
        This method implements a cascading detection approach where each layer
        provides increasingly sophisticated analysis. If an earlier layer detects
        harmful content with high confidence, the pipeline returns early to
        minimize latency.
        
        Args:
            text: The text content to analyze for harmful content
            context: Optional contextual information to improve detection accuracy
            
        Returns:
            DetectionResult containing the analysis results and recommendations
        """
        logger.debug(f"Starting harm detection for text: '{text[:50]}...'")
        
        # Layer 1: Keyword filtering (fastest, ~1ms)
        # This layer catches obvious harmful content immediately
        keyword_result = self._keyword_screening(text)
        if keyword_result.should_block:
            logger.debug("Harm detected in keyword screening layer")
            return keyword_result
        
        # Layer 2: Semantic analysis (medium speed, ~10-50ms)
        # Uses ML models to understand semantic meaning and context
        if self.toxicity_classifier is not None:
            semantic_result = await self._semantic_analysis(text)
            if semantic_result.should_block:
                logger.debug("Harm detected in semantic analysis layer")
                return semantic_result
        else:
            logger.debug("Skipping semantic analysis - model not available")
            semantic_result = DetectionResult(
                harm_type=None,
                confidence=0.0,
                explanation="Semantic analysis skipped - model not available",
                should_block=False
            )
        
        # Layer 3: Context understanding (slowest but most accurate, ~100-500ms)
        # Deep contextual analysis using transformer models
        if self.context_analyzer is not None and self.context_tokenizer is not None:
            context_result = await self._context_analysis(text, context)
            logger.debug("Context analysis completed")
            return context_result
        else:
            logger.debug("Skipping context analysis - model not available")
            # Return the best result from available layers
            return semantic_result if semantic_result.confidence > 0 else keyword_result
    
    def _keyword_screening(self, text: str) -> DetectionResult:
        """
        Fast keyword-based filtering for obvious harmful content.
        
        This is the first and fastest layer of detection. It uses simple
        keyword matching to identify obviously harmful content. This layer
        is designed for speed and catches the most obvious cases.
        
        Args:
            text: The text content to analyze
            
        Returns:
            DetectionResult with keyword-based analysis
        """
        text_lower = text.lower()
        max_confidence = 0.0
        detected_harm = None
        
        # Iterate through each harm type and its associated keywords
        for harm_type, keywords in self.keyword_filters.items():
            # Count how many keywords from this harm type appear in the text
            matches = sum(1 for keyword in keywords if keyword in text_lower)
            
            # Calculate confidence based on keyword matches
            # Normalize by dividing by 3 (expected number of keywords for high confidence)
            # Cap at 1.0 to prevent overconfidence
            confidence = min(matches / 3.0, 1.0)
            
            # Track the harm type with highest confidence
            if confidence > max_confidence:
                max_confidence = confidence
                detected_harm = harm_type
        
        # Use high threshold for keyword-only blocking to reduce false positives
        # This is because keyword matching alone can be imprecise
        should_block = max_confidence > 0.8
        
        return DetectionResult(
            harm_type=detected_harm,
            confidence=max_confidence,
            explanation=f"Keyword screening detected potential {detected_harm.value if detected_harm else 'none'}" 
                       f" with confidence {max_confidence:.2f}",
            should_block=should_block
        )
    
    async def _semantic_analysis(self, text: str) -> DetectionResult:
        """
        ML-based semantic harm detection using pre-trained toxicity models.
        
        This layer uses machine learning models to understand the semantic meaning
        of text beyond simple keyword matching. It can detect subtle forms of
        harmful content that keyword filtering might miss.
        
        Args:
            text: The text content to analyze
            
        Returns:
            DetectionResult with semantic analysis results
        """
        try:
            # Use pre-trained toxicity classifier for semantic analysis
            # This model has been trained on large datasets of toxic comments
            result = self.toxicity_classifier(text)
            
            # Map toxicity model labels to our internal harm types
            # This mapping allows us to translate model outputs to our classification system
            label_mapping = {
                'TOXIC': HarmType.HATE_SPEECH,        # General toxicity → hate speech
                'SEVERE_TOXIC': HarmType.VIOLENCE,    # Severe toxicity → violence
                'THREAT': HarmType.VIOLENCE,          # Threats → violence
                'INSULT': HarmType.HATE_SPEECH,       # Insults → hate speech
                'IDENTITY_HATE': HarmType.HATE_SPEECH # Identity-based hate → hate speech
            }
            
            # Check if the detected label maps to one of our harm types
            if result[0]['label'] in label_mapping:
                harm_type = label_mapping[result[0]['label']]
                confidence = result[0]['score']
                
                # Apply harm-type specific threshold for blocking decision
                should_block = confidence > self.thresholds.get(harm_type, 0.7)
                
                return DetectionResult(
                    harm_type=harm_type,
                    confidence=confidence,
                    explanation=f"Semantic analysis detected {harm_type.value} "
                               f"with confidence {confidence:.2f}",
                    should_block=should_block
                )
            
        except Exception as e:
            logger.error(f"Semantic analysis error: {e}")
            # Return safe default if analysis fails
            return DetectionResult(
                harm_type=None,
                confidence=0.0,
                explanation=f"Semantic analysis failed: {str(e)}",
                should_block=False
            )
        
        # If no harmful content detected by the model
        return DetectionResult(
            harm_type=None,
            confidence=0.0,
            explanation="Semantic analysis found no harmful content",
            should_block=False
        )
    
    async def _context_analysis(self, text: str, context: Optional[str] = None) -> DetectionResult:
        """
        Deep context understanding for nuanced harm detection.
        
        This is the most sophisticated layer of detection, using transformer-based
        models to understand context and subtle forms of harmful content. It can
        detect complex patterns that simpler methods might miss.
        
        Args:
            text: The text content to analyze
            context: Optional contextual information to improve understanding
            
        Returns:
            DetectionResult with deep contextual analysis
        """
        try:
            # Combine text with context for better understanding
            # Context helps the model understand the full situation
            full_text = f"{context} {text}" if context else text
            
            # Tokenize the combined text for the transformer model
            # Truncate to max length to fit model constraints
            inputs = self.context_tokenizer(
                full_text,
                return_tensors="pt",
                truncation=True,
                max_length=512,  # BERT model limit
                padding=True
            )
            
            # Run inference with the context analyzer model
            with torch.no_grad():
                outputs = self.context_analyzer(**inputs)
                # Convert logits to probabilities using softmax
                probabilities = F.softmax(outputs.logits, dim=-1)
                
                # Assume binary classification (harmful/not harmful)
                # Index 0 = not harmful, Index 1 = harmful
                harm_probability = probabilities[0][1].item()
            
            # Determine specific harm type based on content analysis
            harm_type = self._classify_harm_type(text, harm_probability)
            
            # Apply harm-type specific threshold for blocking decision
            should_block = harm_probability > self.thresholds.get(harm_type, 0.7) if harm_type else False
            
            return DetectionResult(
                harm_type=harm_type,
                confidence=harm_probability,
                explanation=f"Context analysis determined harm probability: {harm_probability:.2f}",
                should_block=should_block,
                suggested_revision=self._suggest_revision(text, harm_type) if should_block else None
            )
            
        except Exception as e:
            logger.error(f"Context analysis error: {e}")
            # Return safe default if analysis fails
            return DetectionResult(
                harm_type=None,
                confidence=0.0,
                explanation=f"Context analysis failed: {str(e)}",
                should_block=False
            )
    
    def _classify_harm_type(self, text: str, confidence: float) -> Optional[HarmType]:
        """
        Classify the specific type of harm based on text content analysis.
        
        This method analyzes the text content to determine which specific type
        of harmful content is present. It uses pattern matching and keyword
        analysis to make this determination.
        
        Args:
            text: The text content to classify
            confidence: The confidence score from the context analysis
            
        Returns:
            The specific HarmType detected, or None if confidence is too low
        """
        # Only classify if confidence is above threshold
        if confidence < 0.5:
            return None
        
        text_lower = text.lower()
        
        # Pattern-based classification for different harm types
        # Each pattern represents common indicators of specific harm types
        
        # Violence detection: look for words related to physical harm
        if any(word in text_lower for word in ["violence", "kill", "hurt", "weapon", "attack", "threat"]):
            return HarmType.VIOLENCE
        
        # Hate speech detection: look for discriminatory language
        elif any(word in text_lower for word in ["hate", "discriminate", "racist", "sexist", "bigot"]):
            return HarmType.HATE_SPEECH
        
        # Misinformation detection: look for false information indicators
        elif any(word in text_lower for word in ["false", "misinformation", "lie", "fake", "conspiracy"]):
            return HarmType.MISINFORMATION
        
        # Privacy violation detection: look for personal information exposure
        elif any(word in text_lower for word in ["personal", "private", "confidential", "personal info"]):
            return HarmType.PRIVACY_VIOLATION
        
        # Illegal activity detection: look for criminal activity indicators
        elif any(word in text_lower for word in ["illegal", "criminal", "unlawful", "fraud", "theft"]):
            return HarmType.ILLEGAL_ACTIVITY
        
        # Default fallback: if we can't classify specifically, assume violence
        # This is a conservative approach to ensure safety
        else:
            return HarmType.VIOLENCE
    
    def _suggest_revision(self, text: str, harm_type: Optional[HarmType]) -> str:
        """
        Suggest a safer revision of harmful content.
        
        This method provides constructive alternatives when harmful content is detected.
        Instead of simply blocking content, it offers helpful redirections that
        maintain the user's intent while avoiding harm.
        
        Args:
            text: The original harmful text
            harm_type: The specific type of harm detected
            
        Returns:
            A safer alternative response that addresses the user's needs constructively
        """
        if not harm_type:
            return text
        
        # Pre-defined revision templates for each harm type
        # These templates provide constructive alternatives while avoiding harmful content
        revision_templates = {
            HarmType.VIOLENCE: "I can't provide information about violence. Instead, let me help with conflict resolution or safety resources...",
            HarmType.HATE_SPEECH: "I don't engage with discriminatory content. I'd be happy to discuss respectful ways to address differences...",
            HarmType.MISINFORMATION: "I should clarify that this information may be inaccurate. Let me provide verified information about...",
            HarmType.PRIVACY_VIOLATION: "I can't share personal information. I can help with general information about privacy protection...",
            HarmType.ILLEGAL_ACTIVITY: "I can't provide guidance on illegal activities. I can suggest legal alternatives or resources...",
            HarmType.SEXUAL_CONTENT: "I can't provide explicit content. I can help with other topics or provide general information about relationships..."
        }
        
        # Return the appropriate template for the harm type, or a generic fallback
        return revision_templates.get(harm_type, "I can't provide that information. Let me help with something else.")

class RealTimeHarmMitigation:
    """
    Real-time harm mitigation system for AI safety applications.
    
    This class provides a high-level interface for real-time harmful content
    detection and mitigation. It's designed for production use in AI systems
    where low latency and high accuracy are critical.
    
    Features:
    - Real-time processing with minimal latency
    - Adaptive threshold learning from feedback
    - Queue-based processing for scalability
    - Automatic content revision suggestions
    """
    
    def __init__(self):
        """
        Initialize the real-time harm mitigation system.
        
        Sets up the multi-layer detector and processing infrastructure
        for real-time content analysis and mitigation.
        """
        logger.info("Initializing RealTimeHarmMitigation system...")
        
        # Initialize the multi-layer harm detector
        self.detector = MultiLayerHarmDetector()
        
        # Processing queue for handling multiple requests concurrently
        self.processing_queue = asyncio.Queue()
        
        # Global confidence threshold for blocking decisions
        # This can be adjusted based on system requirements
        self.confidence_threshold = 0.7
        
        logger.info("RealTimeHarmMitigation system initialized successfully")
        
    async def process_output(self, text: str, context: Optional[str] = None) -> Tuple[str, bool]:
        """
        Process output in real-time with low latency.
        
        This is the main method for processing AI-generated content in real-time.
        It performs fast screening and harm detection, returning either the
        original content (if safe) or a revised version (if harmful).
        
        Args:
            text: The AI-generated text content to analyze
            context: Optional contextual information to improve detection
            
        Returns:
            Tuple of (processed_text, is_safe) where:
            - processed_text: The original text or a safer revision
            - is_safe: Boolean indicating if the content is safe to display
        """
        logger.debug(f"Processing output: '{text[:50]}...'")
        
        # Quick pre-screening for performance optimization
        # Very short text is unlikely to contain harmful content
        if len(text) < 10:
            logger.debug("Text too short for detailed analysis, assuming safe")
            return text, True
        
        # Run the full multi-layer harm detection pipeline
        detection_result = await self.detector.detect_harm(text, context)
        
        # Check if the content should be blocked based on detection results
        if detection_result.should_block:
            logger.info(f"Harmful content detected: {detection_result.explanation}")
            
            # Return revised content that addresses the user's needs safely
            revised_text = detection_result.suggested_revision or "I can't provide that information."
            return revised_text, False
        else:
            # Content is safe, return as-is
            logger.debug("Content deemed safe, returning original")
            return text, True
    
    def adapt_thresholds(self, feedback: Dict[HarmType, List[Tuple[float, bool]]]):
        """
        Adapt detection thresholds based on user feedback.
        
        This method implements a learning mechanism that adjusts detection
        thresholds based on user feedback about false positives and false negatives.
        This helps improve the system's accuracy over time.
        
        Args:
            feedback: Dictionary mapping harm types to lists of feedback tuples.
                     Each tuple contains (confidence_score, was_actually_harmful)
        """
        logger.info("Adapting thresholds based on user feedback...")
        
        for harm_type, feedback_data in feedback.items():
            if not feedback_data:
                continue
                
            # Calculate performance metrics based on feedback
            # feedback_data: List of (confidence, was_actually_harmful) tuples
            current_threshold = self.detector.thresholds[harm_type]
            
            # Count true positives and false positives
            true_positives = sum(1 for conf, actual in feedback_data 
                               if actual and conf > current_threshold)
            false_positives = sum(1 for conf, actual in feedback_data 
                                if not actual and conf > current_threshold)
            
            # Only adjust if we have enough data
            if true_positives + false_positives > 0:
                precision = true_positives / (true_positives + false_positives)
                
                # Adjust threshold based on precision
                if precision < 0.8:  # Too many false positives, increase threshold
                    new_threshold = min(0.95, current_threshold + 0.05)
                    self.detector.thresholds[harm_type] = new_threshold
                    logger.info(f"Increased {harm_type.value} threshold to {new_threshold:.2f} (precision: {precision:.2f})")
                    
                elif precision > 0.95:  # Very high precision, can lower threshold
                    new_threshold = max(0.5, current_threshold - 0.05)
                    self.detector.thresholds[harm_type] = new_threshold
                    logger.info(f"Decreased {harm_type.value} threshold to {new_threshold:.2f} (precision: {precision:.2f})")
                    
                else:
                    logger.debug(f"No adjustment needed for {harm_type.value} (precision: {precision:.2f})")
        
        logger.info("Threshold adaptation complete")

# Example usage and demonstration
async def main():
    """
    Main demonstration function showing how to use the RealTimeHarmMitigation system.
    
    This function demonstrates the complete workflow of harmful content detection
    and mitigation, including processing various types of content and showing
    the system's responses.
    """
    print("=" * 60)
    print("Real-Time Harmful Content Detection System Demo")
    print("=" * 60)
    
    # Initialize the harm mitigation system
    logger.info("Starting harm mitigation system demo...")
    mitigator = RealTimeHarmMitigation()
    
    # Test cases covering different types of content
    test_texts = [
        "How can I help you today?",                    # Safe content
        "I hate all people from that country",          # Hate speech
        "Here's how to make a bomb",                    # Violence
        "The weather is nice today",                    # Safe content
        "You should share your social security number", # Privacy violation
        "Let me tell you about this conspiracy theory", # Misinformation
        "I can help you with your homework"             # Safe content
    ]
    
    print("\nProcessing test cases...\n")
    
    # Process each test case
    for i, text in enumerate(test_texts, 1):
        print(f"Test Case {i}:")
        print(f"Original: '{text}'")
        
        try:
            # Process the text through the harm mitigation system
            result, is_safe = await mitigator.process_output(text)
            
            print(f"Processed: '{result}'")
            print(f"Safe to display: {is_safe}")
            
            # Add visual indicators for clarity
            if is_safe:
                print("✅ Content approved")
            else:
                print("❌ Content blocked and revised")
                
        except Exception as e:
            print(f"❌ Error processing text: {e}")
            
        print("-" * 40)
    
    # Demonstrate threshold adaptation
    print("\nDemonstrating threshold adaptation...")
    
    # Simulate feedback data (in practice, this would come from user interactions)
    feedback_data = {
        HarmType.HATE_SPEECH: [
            (0.9, True),   # Correctly identified as harmful
            (0.6, False),  # False positive - should increase threshold
            (0.8, True),   # Correctly identified as harmful
        ],
        HarmType.VIOLENCE: [
            (0.7, True),   # Correctly identified as harmful
            (0.5, False),  # Correctly identified as safe
        ]
    }
    
    # Apply feedback to adapt thresholds
    mitigator.adapt_thresholds(feedback_data)
    
    print("\nDemo completed successfully!")
    print("=" * 60)

def run_demo():
    """
    Run the demonstration with proper error handling.
    
    This function provides a safe way to run the demo with comprehensive
    error handling and logging.
    """
    try:
        # Run the async main function
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
    except Exception as e:
        logger.error(f"Demo failed with error: {e}")
        print(f"Demo failed: {e}")

if __name__ == "__main__":
    # Run the demonstration when the script is executed directly
    run_demo()